#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Generate multimodal SFT data from video prompts using SGLang native video input."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import sys
import traceback
from pathlib import Path
from typing import Any
from urllib.parse import quote

import tqdm

QWEN_IMAGE_TOKEN = "<|vision_start|><|image_pad|><|vision_end|>"
_UNRESOLVED_MEDIA_PATHS: set[str] = set()


def _load_json_or_jsonl(path: str) -> list[dict[str, Any]]:
    if path.endswith("jsonl"):
        with open(path, encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected list data in {path}")
    return data


def _first_user_message(sample: dict[str, Any]) -> dict[str, Any]:
    messages = sample.get("messages") or sample.get("conversations") or sample.get("conversation")
    if not isinstance(messages, list):
        raise ValueError(f"Sample has no messages/conversations list: keys={sorted(sample)}")
    for message in messages:
        role = (message.get("role") or message.get("from") or "").lower()
        if role in ("user", "human"):
            return message
    raise ValueError("Sample has no user message")


def _extract_message_text_and_media(sample: dict[str, Any]) -> tuple[str, str | None, str | None]:
    video_path = sample.get("video_path")
    image_path = sample.get("image_path") or sample.get("image")

    message = _first_user_message(sample)
    content = message.get("content") or message.get("value")
    text_parts: list[str] = []

    if isinstance(content, str):
        text_parts.append(content)
    elif isinstance(content, list):
        for item in content:
            if not isinstance(item, dict):
                continue
            item_type = item.get("type")
            if item_type == "text" and isinstance(item.get("text"), str):
                text_parts.append(item["text"])
            elif item_type == "video":
                video_path = item.get("video") or video_path
            elif item_type == "image":
                image_path = item.get("image") or image_path
    else:
        raise ValueError(f"Unsupported user content: {content!r}")

    prompt = "\n".join(part.strip() for part in text_parts if part.strip())
    if not prompt:
        raise ValueError("Could not extract text prompt from sample")
    return prompt, video_path, image_path


def _extract_text_and_media(sample: dict[str, Any]) -> tuple[str, str | None, str | None]:
    prompt = sample.get("prompt")
    text, video_path, image_path = _extract_message_text_and_media(sample)
    if isinstance(prompt, str) and prompt.strip():
        text = prompt.strip()
    image_path = sample.get("image_path") or sample.get("image") or image_path
    if image_path:
        return text, None, image_path
    return text, sample.get("video_path") or video_path, None


def _resolve_media_path(
    path: str | None, media_root: str | None, input_root: str | None
) -> str | None:
    if not path:
        return None
    if path.startswith(("http://", "https://", "data:")):
        return path
    candidate = Path(path)
    if candidate.is_absolute() and candidate.exists():
        return str(candidate)
    if input_root:
        rooted = Path(input_root) / path
        if rooted.exists():
            return str(rooted)
    if media_root:
        rooted = Path(media_root) / path
        if rooted.exists():
            return str(rooted)
        # If the record stores an absolute host path, preserve its suffix under media_root.
        parts = candidate.parts
        if "videos" in parts:
            suffix = Path(*parts[parts.index("videos") :])
            rooted = Path(media_root) / suffix
            if rooted.exists():
                return str(rooted)
    if candidate.exists():
        return str(candidate)
    if path not in _UNRESOLVED_MEDIA_PATHS:
        print(f"WARNING: could not resolve media path: {path}")
        _UNRESOLVED_MEDIA_PATHS.add(path)
    return None


def _as_openai_media_value(
    path: str, media_url_base: str | None, media_root: str | None, input_root: str | None
) -> str:
    if path.startswith(("http://", "https://", "data:")):
        return path
    if media_url_base:
        candidate = Path(path)
        if candidate.is_absolute():
            if media_root:
                try:
                    path = str(candidate.relative_to(media_root))
                except ValueError:
                    # The local HTTP server deliberately exposes only media_root.
                    # Leave paths outside it local for SGLang to resolve directly.
                    return path
            else:
                return f"{media_url_base.rstrip('/')}{quote(path, safe='/')}"
        return f"{media_url_base.rstrip('/')}/{quote(path, safe='/')}"
    # Do not convert local paths to file://. This SGLang build falls through to
    # the base64 loader for file:// videos and raises "Incorrect padding".
    return path


def _openai_chat_url(url: str) -> str:
    url = url.rstrip("/")
    if url.endswith("/v1/chat/completions"):
        return url
    if url.endswith("/v1"):
        return f"{url}/chat/completions"
    return f"{url}/v1/chat/completions"


def _messages_for_output(sample: dict[str, Any], answer: str) -> list[dict[str, Any]]:
    messages = sample.get("messages")
    if isinstance(messages, list):
        output_messages = list(messages)
    else:
        prompt, video_path, image_path = _extract_text_and_media(sample)
        content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
        if image_path:
            content.append({"type": "image", "image": image_path})
        elif video_path:
            content.append({"type": "video", "video": video_path, "fps": 4})
        output_messages = [{"role": "user", "content": content}]

    output_messages.append({"role": "assistant", "content": answer})
    return output_messages


def _coerce_text(value: Any) -> str:
    if value is None:
        return ""
    if type(value).__name__ == "ProgramState":
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, dict):
        for key in ("answer", "text", "value", "content"):
            text = _coerce_text(value.get(key))
            if text:
                return text
        return ""
    for attr_name in ("text", "value", "content"):
        attr = getattr(value, attr_name, None)
        if callable(attr):
            try:
                text = _coerce_text(attr())
                if text:
                    return text
            except Exception:
                pass
        else:
            text = _coerce_text(attr)
            if text:
                return text
    return ""


def _answer_from_messages(messages: Any) -> str:
    if callable(messages):
        try:
            messages = messages()
        except Exception:
            return ""
    if not isinstance(messages, list):
        return ""
    for message in reversed(messages):
        if not isinstance(message, dict):
            continue
        role = (message.get("role") or "").lower()
        if role != "assistant":
            continue
        return _coerce_text(message.get("content"))
    return ""


def _state_answer(state: Any) -> str:
    # SGLang ProgramState normally supports state["answer"]. Some versions or
    # failure paths expose variables through helper methods/attributes instead.
    try:
        text = _coerce_text(state["answer"])
        if text:
            return text
    except Exception:
        pass

    if isinstance(state, dict):
        return _coerce_text(state.get("answer") or state.get("value") or state.get("text"))

    for method_name in ("get", "get_var", "get_variable", "var"):
        method = getattr(state, method_name, None)
        if not callable(method):
            continue
        try:
            text = _coerce_text(method("answer"))
            if text:
                return text
        except Exception:
            pass

    text = _answer_from_messages(getattr(state, "messages", None))
    if text:
        return text

    for attr_name in ("answer", "variables", "vars"):
        text = _coerce_text(getattr(state, attr_name, None))
        if text:
            return text

    return ""


def _prompt_with_vision_token(prompt: str, media_type: str, token_format: str) -> str:
    if token_format == "none":
        return prompt
    if token_format != "qwen_vl":
        raise ValueError(f"Unsupported vision token format: {token_format}")
    if media_type == "video":
        # SGLang's native sgl.video(...) transport binds video data using its
        # own placeholder path. Adding a literal Qwen <|video_pad|> token here
        # makes the server look for an extra unbound video iterator.
        return prompt

    known_tokens = (
        "<|image_pad|>",
        "<|video_pad|>",
        "<image>",
        "<video>",
    )
    if any(existing_token in prompt for existing_token in known_tokens):
        return prompt
    return f"{QWEN_IMAGE_TOKEN}\n{prompt}"


def _openai_generate_one(
    request: dict[str, Any],
    url: str,
    model_name: str,
    temperature: float,
    max_tokens: int,
    timeout: int,
    media_url_base: str | None,
    media_root: str | None,
    input_root: str | None,
) -> tuple[str, str | None]:
    try:
        import requests
    except ImportError as e:
        raise ImportError("OpenAI API mode requires the requests package.") from e

    content = [{"type": "text", "text": request["prompt"]}]
    if request.get("image_path"):
        content.append(
            {
                "type": "image_url",
                "image_url": {
                    "url": _as_openai_media_value(
                        request["image_path"], media_url_base, media_root, input_root
                    )
                },
            }
        )
    else:
        content.append(
            {
                "type": "video_url",
                "video_url": {
                    "url": _as_openai_media_value(
                        request["video_path"], media_url_base, media_root, input_root
                    )
                },
            }
        )

    payload = {
        "model": model_name,
        "messages": [{"role": "user", "content": content}],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    try:
        response = requests.post(url, json=payload, timeout=timeout)
        if response.status_code != 200:
            return "", f"HTTP {response.status_code}: {response.text[:1000]}"
        data = response.json()
        choices = data.get("choices") or []
        if not choices:
            return "", f"No choices in OpenAI response: {data}"
        message = choices[0].get("message") or {}
        answer = _coerce_text(message.get("content"))
        if not answer:
            return "", f"Could not extract assistant content from OpenAI response: {data}"
        return answer, None
    except Exception:
        return "", traceback.format_exc()


def _write_generation(
    outfile,
    idx: int,
    sample: dict[str, Any],
    answer: str,
    generation_error: str | None,
) -> None:
    out = {
        "conversation_id": idx,
        "messages": _messages_for_output(sample, answer),
    }
    if generation_error:
        out["generation_error"] = generation_error
    for key in (
        "id",
        "category",
        "subcategory",
        "dataset",
        "answer",
        "reference_answer",
        "index2ans",
        "video_path",
        "image_path",
        "source_image",
        "source_video_path",
        "source_mcap_path",
        "source_subset",
        "camera",
        "camera_topic",
        "task_path",
        "skill",
        "stage",
        "major_task",
        "task_id",
        "episode_id",
        "task_name",
        "source_split",
        "data_subtype",
        "question_id",
        "image_id",
        "question_type",
        "answer_type",
        "multiple_choice_answer",
        "answer_counts",
        "answers",
        "window_idx",
        "windows_per_mcap",
        "window_message_start",
        "window_message_end",
    ):
        if key in sample:
            out[key] = sample[key]
    outfile.write(json.dumps(out, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--input_root", default=None, help="Mounted shard root, e.g. /input_data")
    parser.add_argument("--media_root", default=None, help="Mounted dataset root, e.g. /media_data")
    parser.add_argument("--num_threads", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max_tokens", type=int, default=2048)
    parser.add_argument("--num_frames", type=int, default=32)
    parser.add_argument("--url", type=str, default="http://localhost:8000")
    parser.add_argument(
        "--media_url_base",
        default=None,
        help="HTTP base URL serving media_root/input_root for OpenAI multimodal requests.",
    )
    parser.add_argument(
        "--api_mode",
        choices=("openai", "native"),
        default="native",
        help="Use SGLang's OpenAI-compatible API or SGLang native DSL client.",
    )
    parser.add_argument("--model_name", default="model")
    parser.add_argument("--request_timeout", type=int, default=1800)
    parser.add_argument(
        "--vision_token_format",
        choices=("qwen_vl", "none"),
        default="qwen_vl",
        help="Insert explicit Qwen/Cosmos-VL image/video placeholder tokens into prompts.",
    )
    parser.add_argument("--log_empty_conversations", action="store_true")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Remove output_path before generating. Useful after a bad/partial run.",
    )
    args = parser.parse_args()

    url = args.url.removesuffix("/v1")

    data = _load_json_or_jsonl(args.data_path)

    if args.overwrite and os.path.exists(args.output_path):
        os.remove(args.output_path)

    finished_ids: set[int] = set()
    done = False
    if os.path.exists(args.output_path):
        with open(args.output_path, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                outdata = json.loads(line)
                finished_ids.add(outdata.get("conversation_id", -1))
                if outdata.get("finished", False):
                    done = True
                    break
    if done:
        print("All conversations already generated")
        sys.exit()

    batch_args = []
    batch_meta = []
    for idx, sample in enumerate(data):
        if idx in finished_ids:
            continue
        prompt, video_path, image_path = _extract_text_and_media(sample)
        resolved_image = _resolve_media_path(image_path, args.media_root, args.input_root)
        resolved_video = None
        if not resolved_image:
            resolved_video = _resolve_media_path(video_path, args.media_root, args.input_root)
        if not resolved_video and not resolved_image:
            print(f"Skipping sample {idx}: no video_path or image_path")
            continue
        request = {
            "prompt": prompt,
            "video_path": resolved_video,
            "image_path": resolved_image,
            "num_frames": args.num_frames,
            "vision_token_format": args.vision_token_format,
        }
        batch_args.append([request])
        batch_meta.append((idx, sample))

    if not batch_args:
        print("No pending multimodal conversations to generate")
        if args.log_empty_conversations:
            os.makedirs(os.path.dirname(args.output_path) or ".", exist_ok=True)
            with open(args.output_path, "a", encoding="utf-8") as f:
                f.write(json.dumps({"finished": True}) + "\n")
        return

    if args.api_mode == "openai":
        chat_url = _openai_chat_url(args.url)
        os.makedirs(os.path.dirname(args.output_path) or ".", exist_ok=True)
        with open(args.output_path, "a", encoding="utf-8") as f:
            with concurrent.futures.ThreadPoolExecutor(max_workers=args.num_threads) as executor:
                future_to_meta = {
                    executor.submit(
                        _openai_generate_one,
                        request_args[0],
                        chat_url,
                        args.model_name,
                        args.temperature,
                        args.max_tokens,
                        args.request_timeout,
                        args.media_url_base,
                        args.media_root,
                        args.input_root,
                    ): meta
                    for request_args, meta in zip(batch_args, batch_meta)
                }
                for future in tqdm.tqdm(
                    concurrent.futures.as_completed(future_to_meta),
                    total=len(future_to_meta),
                    desc="Generating and writing outputs",
                ):
                    idx, sample = future_to_meta[future]
                    answer, error = future.result()
                    if error:
                        print(f"WARNING: generation failed for sample {idx}: {error[:500]}")
                    if not answer and not args.log_empty_conversations:
                        continue
                    _write_generation(f, idx, sample, answer, error)

            if args.log_empty_conversations:
                f.write(json.dumps({"finished": True}) + "\n")
        return

    try:
        import sglang as sgl
    except ImportError as e:
        raise ImportError("server_generate_vlm_sglang.py requires sglang in native mode.") from e

    sgl.set_default_backend(sgl.RuntimeEndpoint(url))

    @sgl.function
    def generate_answer(s, request):
        if request.get("image_path"):
            prompt = _prompt_with_vision_token(
                request["prompt"], "image", request["vision_token_format"]
            )
            s += sgl.user(sgl.image(request["image_path"]) + prompt)
        else:
            prompt = _prompt_with_vision_token(
                request["prompt"], "video", request["vision_token_format"]
            )
            s += sgl.user(
                sgl.video(request["video_path"], num_frames=request["num_frames"]) + prompt
            )
        s += sgl.assistant(sgl.gen("answer"))

    states = generate_answer.run_batch(
        batch_args,
        temperature=args.temperature,
        max_new_tokens=args.max_tokens,
        num_threads=args.num_threads,
        progress_bar=True,
    )

    os.makedirs(os.path.dirname(args.output_path) or ".", exist_ok=True)
    with open(args.output_path, "a", encoding="utf-8") as f:
        for state, (idx, sample) in tqdm.tqdm(
            zip(states, batch_meta), total=len(batch_meta), desc="Writing outputs"
        ):
            answer = _state_answer(state)
            if not answer:
                print(
                    "WARNING: Could not extract sgl.gen('answer') from state "
                    f"type={type(state).__name__}, repr={repr(state)[:500]}"
                )
            if not answer and not args.log_empty_conversations:
                continue
            error = None
            if not answer:
                error = "Could not extract sgl.gen('answer') from SGLang ProgramState."
            _write_generation(f, idx, sample, answer, error)

        if args.log_empty_conversations:
            f.write(json.dumps({"finished": True}) + "\n")


if __name__ == "__main__":
    main()
