"""Common accuracy harness for the QAT experiment series.

Corpus-level (token-weighted) loss / ppl / token-accuracy on the
HuggingFaceH4/Multilingual-Thinking test split -- the same 100 examples the
ModelOpt example evaluates on. Token-weighted so the number does NOT depend on
batch size or world size, unlike Trainer's batch-averaged eval_loss.

Usage:
  python qeval.py --model <path|hub-id> [--label NAME]
                  [--quantize-recipe general/ptq/mxfp4_mlp_weight_only]
                  [--attn eager|kernels-community/vllm-flash-attn3]
                  [--batch 4] [--json out.json]

--quantize-recipe applies ModelOpt PTQ in-memory before evaluating (this is how
the "PTQ reference" numbers are produced). Omit it to evaluate the checkpoint
as-is; an existing modelopt_state.pth in the model dir is auto-restored.
"""
import argparse, json, os, sys, time
import torch

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--label", default=None)
ap.add_argument("--quantize-recipe", default=None)
ap.add_argument("--attn", default="kernels-community/vllm-flash-attn3")
ap.add_argument("--batch", type=int, default=4)
ap.add_argument("--dataset", default="HuggingFaceH4/Multilingual-Thinking",
                help="Hub id or a load_from_disk path. A DatasetDict with an "
                     "explicit test split is used as-is; otherwise train is "
                     "split 90/10 with seed 42 (matching the example).")
ap.add_argument("--max-length", type=int, default=4096)
ap.add_argument("--json", default=None)
ap.add_argument("--csx-bake", action="store_true",
                help="For PTQ eval: write the fake-quantized values straight into "
                     "the expert weights and evaluate a plain model. Exactly "
                     "equivalent to a frozen quantizer (PTQ is static) but avoids "
                     "ModelOpt's per-forward weight cache and all temporaries.")
ap.add_argument("--csx-artifacts", default=None,
                help="Path to csx_Nbit_artifacts.pt; installs CSX quantizers "
                     "(implies the MXFP4 recipe as a carrier for the wrappers).")
args = ap.parse_args()

import modelopt.torch.opt as mto
mto.enable_huggingface_checkpointing()
import modelopt.torch.quantization as mtq
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig, Mxfp4Config
from datasets import load_dataset
from trl import SFTConfig, SFTTrainer

def orig_quant_method(p):
    c = AutoConfig.from_pretrained(p)
    qc = getattr(c, "quantization_config", None)
    return qc.get("quant_method") if qc else None

# Pin to one device rather than device_map="auto": when other tenants occupy the
# card, "auto" silently offloads part of the model to CPU and installs accelerate
# hooks, which wrap forward in a functools.partial and break callers that
# introspect model.forward.__func__. Better to fail loudly than measure a
# half-offloaded model.
kw = dict(dtype=torch.bfloat16, attn_implementation=args.attn,
          device_map={"": 0}, use_cache=False)
if orig_quant_method(args.model) == "mxfp4":
    kw["quantization_config"] = Mxfp4Config(dequantize=True)

t0 = time.time()
model = AutoModelForCausalLM.from_pretrained(args.model, **kw).eval()
tok = AutoTokenizer.from_pretrained(args.model)
def _mem(tag):
    if torch.cuda.is_available():
        tot = sum(torch.cuda.memory_allocated(i) for i in range(torch.cuda.device_count()))
        print(f"[qeval] mem after {tag}: {tot/2**30:.1f} GiB across "
              f"{torch.cuda.device_count()} GPU(s)", flush=True)

print(f"[qeval] loaded in {time.time()-t0:.0f}s", flush=True)
_mem("load")

def _load_eval_ds(name):
    from datasets import load_from_disk
    try:
        d = load_from_disk(name)
    except Exception:
        d = load_dataset(name)
    if "test" in d:
        return d
    return d["train"].train_test_split(test_size=0.1, seed=42)

ds = _load_eval_ds(args.dataset)
print(f"[qeval] eval split: {args.dataset} n={len(ds['test'])}", flush=True)

if args.csx_artifacts and args.csx_bake:
    # Static PTQ: bake fake-quantized weights in place, no wrappers needed.
    from qlab.qat import csx_fake_quantize
    blob = torch.load(args.csx_artifacts, map_location="cpu", weights_only=False)
    geom, arts = blob["geometry"], blob["artifacts"]
    baked = 0
    with torch.no_grad():
        for mname, mod in model.named_modules():
            for proj, (dF, dH) in geom.items():
                if not hasattr(mod, proj):
                    continue
                key = f"{mname}.{proj}"
                if key not in arts:
                    continue
                w = getattr(mod, proj)
                wt = w.detach().transpose(-1, -2).contiguous()
                cb = arts[key]["codebook"].to(w.device)
                qs = arts[key]["qscale"].to(w.device)
                q = csx_fake_quantize(wt, cb, qs, dF, dH)
                w.data.copy_(q.transpose(-1, -2))
                del wt, cb, qs, q
                baked += 1
    sq = blob.get("sqnr_db", {})
    gu = [v for k, v in sq.items() if "gate_up" in k]
    dp = [v for k, v in sq.items() if "down_proj" in k]
    print(f"[qeval] baked {baked} CSX {blob['num_bits']}-bit expert weights; "
          f"fit SQNR gate_up={sum(gu)/max(len(gu),1):.2f} dB "
          f"down_proj={sum(dp)/max(len(dp),1):.2f} dB", flush=True)
    assert baked == 48, f"expected 48 baked tensors, got {baked}"
    del blob, arts
    _mem("CSX bake")
    args.csx_artifacts = None      # don't also install live quantizers

if args.quantize_recipe or args.csx_artifacts:
    # PTQ reference: apply the recipe in-memory, no training. For CSX the MXFP4
    # recipe is only a carrier -- it builds the _QuantGptOssExperts wrappers and
    # selects exactly the expert weights, then we swap in CSX quantizers.
    from modelopt.recipe import load_recipe
    recipe = args.quantize_recipe or "general/ptq/mxfp4_mlp_weight_only"
    cfg = load_recipe(recipe).quantize
    model = mtq.quantize(model, cfg, None)   # weight-only + dynamic -> no calib
    print(f"[qeval] applied PTQ recipe {recipe}", flush=True)
    _mem("mtq.quantize")

if args.csx_artifacts:
    from qlab.qat import install_csx_quantizers
    blob = torch.load(args.csx_artifacts, map_location="cpu", weights_only=False)
    n = install_csx_quantizers(model, blob["artifacts"], blob["geometry"],
                               trainable_scale=False, num_bits=blob["num_bits"])
    sq = blob.get("sqnr_db", {})
    gu = [v for k, v in sq.items() if "gate_up" in k]
    dp = [v for k, v in sq.items() if "down_proj" in k]
    print(f"[qeval] installed {n} CSX {blob['num_bits']}-bit quantizers; "
          f"fit SQNR gate_up={sum(gu)/max(len(gu),1):.2f} dB "
          f"down_proj={sum(dp)/max(len(dp),1):.2f} dB", flush=True)
    del blob
    _mem("CSX install")

nq = sum(1 for n, _ in model.named_modules() if n.endswith("_weight_quantizer"))
print(f"[qeval] weight quantizers active: {nq}", flush=True)

sft = SFTConfig(output_dir="/tmp/qeval", per_device_eval_batch_size=args.batch,
                max_length=args.max_length, bf16=True, report_to=[],
                dataset_num_proc=8, eval_strategy="no", seed=42)
tr = SFTTrainer(model=model, args=sft, train_dataset=ds["train"],
                eval_dataset=ds["test"], processing_class=tok)

tot_nll = torch.zeros((), dtype=torch.float64); tot_tok = 0; correct = 0
with torch.no_grad():
    for i, b in enumerate(tr.get_eval_dataloader()):
        b = {k: v.to(model.device) for k, v in b.items() if isinstance(v, torch.Tensor)}
        labels = b.pop("labels")
        logits = model(**b).logits.float()
        sl, sg = labels[:, 1:], logits[:, :-1]
        m = sl != -100
        tot_nll += torch.nn.functional.cross_entropy(sg[m], sl[m], reduction="sum").double().cpu()
        tot_tok += int(m.sum()); correct += int((sg[m].argmax(-1) == sl[m]).sum())

loss = tot_nll.item() / tot_tok
res = dict(label=args.label or args.model, model=args.model, dataset=args.dataset,
           recipe=args.quantize_recipe, csx=args.csx_artifacts, quantizers=nq, tokens=tot_tok,
           loss=round(loss, 4), ppl=round(float(torch.tensor(loss).exp()), 4),
           token_acc=round(correct / tot_tok, 4))
print("[qeval] RESULT " + json.dumps(res), flush=True)
if args.json:
    with open(args.json, "w") as f: json.dump(res, f, indent=2)
