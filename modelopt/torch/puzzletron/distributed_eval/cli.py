"""Command-line interface for optional distributed Puzzletron evaluation."""

from __future__ import annotations

import argparse
import asyncio
import faulthandler
import json
import os
import signal
import socket
import sys
from datetime import timedelta
from pathlib import Path
from typing import Any

from .campaign import Campaign
from .config import (
    _replacement_scoring_config,
    build_campaign_manifest,
    distributed_stage_config,
    load_plain_pipeline_config,
    parallelism_from_config,
)
from .identity import canonicalize
from .storage import atomic_write_json, read_json


def _json_print(value: Any) -> None:
    print(json.dumps(canonicalize(value), indent=2, sort_keys=True, allow_nan=False))


def _parse_overrides(args) -> list[str]:
    return list(getattr(args, "override", None) or [])


def command_init(args) -> int:
    manifest = build_campaign_manifest(
        args.config,
        world_size=args.world_size,
        name=args.name,
        evaluator_revision=args.evaluator_revision,
        overrides=_parse_overrides(args),
        stage=args.stage,
    )
    campaign = Campaign.create(args.campaign_dir, manifest)
    _json_print(
        {
            "campaign_dir": str(campaign.root),
            "campaign_id": campaign.campaign_id,
            "manifest": manifest,
        }
    )
    return 0


async def _run_coordinator(args) -> int:
    from .client import AsyncEvaluationClient
    from .replace_block import build_replace_block_requests

    campaign = Campaign.open(args.campaign_dir)
    requests = build_replace_block_requests(
        campaign,
        solutions_path=args.solutions,
        solution_ids=args.solution_id,
        sort_solutions_by=args.sort_solutions_by,
        bigger_is_better=args.bigger_is_better,
    )
    output_dir = Path(args.output_dir).resolve()
    candidate_dir = output_dir / "candidates"
    candidate_dir.mkdir(parents=True, exist_ok=True)
    compatibility_dir = (
        Path(args.compatibility_output_dir).resolve()
        if args.compatibility_output_dir
        else None
    )
    if compatibility_dir is not None:
        compatibility_dir.mkdir(parents=True, exist_ok=True)

    request_by_id = {request.request_id: request for request in requests}
    client = AsyncEvaluationClient.from_campaign(
        str(campaign.root),
        stale_seconds=args.stale_seconds,
        connect_timeout_seconds=args.connect_timeout_seconds,
        task_timeout_seconds=args.task_timeout_seconds,
        retry_initial_seconds=args.retry_initial_seconds,
        retry_max_seconds=args.retry_max_seconds,
    )
    completed = 0
    try:
        async with client:
            handles = await client.submit_many(requests)
            print(
                f"[distributed-eval] campaign={campaign.campaign_id} "
                f"requests={len(handles)} output={output_dir}",
                flush=True,
            )
            async for result in client.as_completed(handles):
                request = request_by_id[result.request_id]
                candidate_id = request.payload["candidate_id"]
                payload = {
                    "request": request.to_wire(),
                    "result": result,
                }
                atomic_write_json(candidate_dir / f"{candidate_id}.json", payload)
                if compatibility_dir is not None:
                    solution_id = int(request.payload["solution_id"])
                    scoring_data = campaign.manifest.data.get("scoring") or campaign.manifest.data
                    compatibility = {
                        **result.metrics,
                        "hidden_width": request.payload.get("hidden_width"),
                        "sliced_teacher_baseline": result.provenance.get(
                            "sliced_teacher_baseline"
                        ),
                        "observability": result.provenance.get("observability"),
                        "args": {
                            "teacher_dir": campaign.manifest.model["checkpoint_dir"],
                            "descriptor": campaign.manifest.descriptor,
                            "replacement_library_path": str(
                                Path(args.solutions).resolve().parent
                                / "replacement_library.json"
                            ),
                            "eval_samples": scoring_data.get("eval_samples"),
                            "micro_batch_size": result.provenance.get(
                                "micro_batch_size", scoring_data.get("micro_batch_size")
                            ),
                            "block_size": scoring_data.get("block_size"),
                        },
                        "i_solution": solution_id,
                        "puzzle_solution": request.payload.get("puzzle_solution"),
                        "distributed_evaluation": {
                            "campaign_id": campaign.campaign_id,
                            "request_id": request.request_id,
                            "candidate_id": candidate_id,
                        },
                    }
                    atomic_write_json(
                        compatibility_dir / f"solution_{solution_id}.json",
                        compatibility,
                    )
                completed += 1
                print(
                    f"[distributed-eval] completed={completed}/{len(handles)} "
                    f"request={result.request_id}",
                    flush=True,
                )
    finally:
        campaign.storage.rebuild_summary()
    return 0


def command_coordinator(args) -> int:
    return asyncio.run(_run_coordinator(args))


async def _run_depth_coordinator(args) -> int:
    from .client import AsyncEvaluationClient
    from .config import load_runtime_config
    from .depth import depth_rpc_context_from_config, run_iterative_depth_rpc

    campaign = Campaign.open(args.campaign_dir)
    evaluation_stage = campaign.manifest.metadata.get("evaluation_stage")
    if evaluation_stage != "depth":
        raise ValueError(
            f"depth coordinator requires a depth campaign, got {evaluation_stage!r}"
        )
    runtime_cfg = load_runtime_config(args.config, overrides=_parse_overrides(args))
    context = depth_rpc_context_from_config(runtime_cfg)
    if args.output_dir is not None:
        context["output_dir"] = Path(args.output_dir).resolve()
    client = AsyncEvaluationClient.from_campaign(
        str(campaign.root),
        stale_seconds=args.stale_seconds,
        connect_timeout_seconds=args.connect_timeout_seconds,
        task_timeout_seconds=args.task_timeout_seconds,
        retry_initial_seconds=args.retry_initial_seconds,
        retry_max_seconds=args.retry_max_seconds,
    )
    try:
        async with client:
            result = await run_iterative_depth_rpc(campaign, client=client, **context)
    finally:
        campaign.storage.rebuild_summary()
    _json_print(result)
    return 0


def command_depth_coordinator(args) -> int:
    return asyncio.run(_run_depth_coordinator(args))


def _default_worker_id() -> str:
    pieces = [socket.gethostname()]
    for name in ("SLURM_JOB_ID", "SLURM_NODEID", "LOCAL_RANK"):
        value = os.environ.get(name)
        if value is not None:
            pieces.append(value)
    return "-".join(pieces)


def command_worker(args) -> int:
    os.environ.setdefault("TORCH_NCCL_ASYNC_ERROR_HANDLING", "1")
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    campaign = Campaign.open(args.campaign_dir)
    stack_log = None
    if hasattr(signal, "SIGUSR1"):
        log_dir = campaign.root.parent / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        stack_log = (
            log_dir
            / f"faulthandler_{os.environ.get('SLURM_JOB_ID', 'local')}_"
            f"node{os.environ.get('SLURM_NODEID', 'local')}_"
            f"rank{os.environ.get('RANK', '0')}.log"
        ).open("a")
        faulthandler.register(signal.SIGUSR1, file=stack_log, all_threads=True)
    evaluation_stage = str(campaign.manifest.metadata.get("evaluation_stage", "replace_block"))
    plain_cfg = distributed_stage_config(
        load_plain_pipeline_config(args.config, overrides=_parse_overrides(args)),
        stage=evaluation_stage,
    )
    actual_parallelism = parallelism_from_config(plain_cfg, world_size=world_size)
    if actual_parallelism != campaign.manifest.parallelism:
        raise ValueError(
            "Worker topology does not match campaign topology:\n"
            f"worker={actual_parallelism.model_dump()}\n"
            f"campaign={campaign.manifest.parallelism.model_dump()}"
        )
    scoring = _replacement_scoring_config(plain_cfg)
    automodel = dict(scoring.get("automodel") or {})
    force_hf = bool(automodel.get("force_hf", (plain_cfg.get("model") or {}).get("force_hf", True)))
    if force_hf != campaign.manifest.force_hf:
        raise ValueError(
            f"Worker force_hf={force_hf} does not match campaign "
            f"force_hf={campaign.manifest.force_hf}"
        )

    import modelopt.torch.utils.distributed as dist

    from .automodel_executor import AutoModelReplaceBlockExecutor
    from .config import load_runtime_config
    from .worker import DistributedEvaluationWorker

    runtime_cfg = load_runtime_config(args.config, overrides=_parse_overrides(args))
    if evaluation_stage == "depth":
        from ..depth.iterative import _depth_scoring_config

        runtime_cfg = _depth_scoring_config(runtime_cfg)
    timeout_minutes = int(runtime_cfg.get("nccl_timeout_minutes", 120))
    dist.setup(timeout=timedelta(minutes=timeout_minutes))
    try:
        executor = AutoModelReplaceBlockExecutor(runtime_cfg)
        worker = DistributedEvaluationWorker(
            campaign,
            executor,
            host=args.host,
            port=args.port,
            worker_id=args.worker_id or _default_worker_id(),
            heartbeat_seconds=args.heartbeat_seconds,
        )
        worker.run()
    finally:
        dist.cleanup()
        if stack_log is not None:
            faulthandler.unregister(signal.SIGUSR1)
            stack_log.close()
    return 0


def command_status(args) -> int:
    campaign = Campaign.open(args.campaign_dir)
    active = campaign.registry.list_workers(
        campaign.manifest,
        stale_seconds=args.stale_seconds,
    )
    all_workers = campaign.registry.list_workers(
        campaign.manifest,
        stale_seconds=args.stale_seconds,
        include_stale=True,
    )
    _json_print(
        {
            "campaign_id": campaign.campaign_id,
            "storage": campaign.storage.summary(),
            "active_workers": active,
            "stale_workers": [worker for worker in all_workers if worker not in active],
        }
    )
    return 0


def command_cleanup_stale(args) -> int:
    campaign = Campaign.open(args.campaign_dir)
    removed = campaign.registry.cleanup_stale(stale_seconds=args.stale_seconds)
    _json_print({"removed": [str(path) for path in removed]})
    return 0


def command_rebuild_summary(args) -> int:
    campaign = Campaign.open(args.campaign_dir)
    path = campaign.storage.rebuild_summary()
    _json_print({"summary": read_json(path), "path": str(path)})
    return 0


def command_lookup(args) -> int:
    campaign = Campaign.open(args.campaign_dir)
    result = campaign.storage.get_result(args.request_id)
    if result is None:
        terminal = campaign.storage.get_terminal_error(args.request_id)
        _json_print({"request_id": args.request_id, "result": None, "terminal": terminal})
        return 1
    _json_print(result)
    return 0


async def _drain_workers(args) -> int:
    from .http_transport import AsyncHttpClient

    campaign = Campaign.open(args.campaign_dir)
    workers = campaign.registry.list_workers(
        campaign.manifest,
        stale_seconds=args.stale_seconds,
    )
    http = AsyncHttpClient(token=campaign.storage.read_token())
    outcomes = []
    try:
        for worker in workers:
            try:
                response = await http.request("POST", f"{worker.endpoint}/v1/drain", payload={})
                outcomes.append({"worker": worker.worker_id, "response": response})
            except Exception as error:
                outcomes.append({"worker": worker.worker_id, "error": str(error)})
    finally:
        await http.close()
    _json_print({"workers": outcomes})
    return 0


def command_drain(args) -> int:
    return asyncio.run(_drain_workers(args))


def command_import_legacy(args) -> int:
    from .legacy_cache import import_legacy_cache

    campaign = Campaign.open(args.campaign_dir)
    _json_print(import_legacy_cache(campaign, args.cache_file, handler=args.handler))
    return 0


def _add_campaign_argument(parser) -> None:
    parser.add_argument("--campaign-dir", required=True)


def _add_registry_timing(parser) -> None:
    parser.add_argument("--stale-seconds", type=float, default=45.0)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    init = subparsers.add_parser("init", help="initialize or validate a campaign")
    _add_campaign_argument(init)
    init.add_argument("--config", required=True)
    init.add_argument("--world-size", type=int, required=True)
    init.add_argument("--name")
    init.add_argument("--stage", choices=("replace_block", "depth"), default="replace_block")
    init.add_argument("--evaluator-revision")
    init.add_argument("--override", action="append", default=[])
    init.set_defaults(func=command_init)

    coordinator = subparsers.add_parser("coordinator", help="score one-block solutions")
    _add_campaign_argument(coordinator)
    _add_registry_timing(coordinator)
    coordinator.add_argument("--solutions", required=True)
    coordinator.add_argument("--solution-id", action="append", type=int)
    coordinator.add_argument("--sort-solutions-by")
    coordinator.add_argument("--bigger-is-better", action="store_true")
    coordinator.add_argument("--output-dir", required=True)
    coordinator.add_argument("--compatibility-output-dir")
    coordinator.add_argument("--connect-timeout-seconds", type=float, default=10.0)
    coordinator.add_argument("--task-timeout-seconds", type=float, default=7200.0)
    coordinator.add_argument("--retry-initial-seconds", type=float, default=5.0)
    coordinator.add_argument("--retry-max-seconds", type=float, default=60.0)
    coordinator.set_defaults(func=command_coordinator)

    depth_coordinator = subparsers.add_parser(
        "depth-coordinator",
        help="score an iterative depth trajectory across persistent workers",
    )
    _add_campaign_argument(depth_coordinator)
    _add_registry_timing(depth_coordinator)
    depth_coordinator.add_argument("--config", required=True)
    depth_coordinator.add_argument("--override", action="append", default=[])
    depth_coordinator.add_argument("--output-dir")
    depth_coordinator.add_argument("--connect-timeout-seconds", type=float, default=10.0)
    depth_coordinator.add_argument("--task-timeout-seconds", type=float, default=7200.0)
    depth_coordinator.add_argument("--retry-initial-seconds", type=float, default=5.0)
    depth_coordinator.add_argument("--retry-max-seconds", type=float, default=60.0)
    depth_coordinator.set_defaults(func=command_depth_coordinator)

    worker = subparsers.add_parser("worker", help="run one torchrun worker group")
    _add_campaign_argument(worker)
    worker.add_argument("--config", required=True)
    worker.add_argument("--override", action="append", default=[])
    worker.add_argument("--host", default=socket.getfqdn())
    worker.add_argument("--port", type=int, default=5010)
    worker.add_argument("--worker-id")
    worker.add_argument("--heartbeat-seconds", type=float, default=10.0)
    worker.set_defaults(func=command_worker)

    status = subparsers.add_parser("status", help="show campaign and worker status")
    _add_campaign_argument(status)
    _add_registry_timing(status)
    status.set_defaults(func=command_status)

    cleanup = subparsers.add_parser("cleanup-stale", help="remove stale heartbeat files")
    _add_campaign_argument(cleanup)
    _add_registry_timing(cleanup)
    cleanup.set_defaults(func=command_cleanup_stale)

    rebuild = subparsers.add_parser("rebuild-summary", help="rebuild cache summary")
    _add_campaign_argument(rebuild)
    rebuild.set_defaults(func=command_rebuild_summary)

    lookup = subparsers.add_parser("lookup", help="read one cached result")
    _add_campaign_argument(lookup)
    lookup.add_argument("request_id")
    lookup.set_defaults(func=command_lookup)

    drain = subparsers.add_parser("drain", help="gracefully drain active workers")
    _add_campaign_argument(drain)
    _add_registry_timing(drain)
    drain.set_defaults(func=command_drain)

    legacy = subparsers.add_parser("import-legacy", help="import an old JSON result cache")
    _add_campaign_argument(legacy)
    legacy.add_argument("--cache-file", required=True)
    legacy.add_argument("--handler", default="replace_block")
    legacy.set_defaults(func=command_import_legacy)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
