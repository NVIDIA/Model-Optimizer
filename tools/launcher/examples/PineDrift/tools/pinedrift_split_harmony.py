#!/usr/bin/env python3
"""Split the raw Harmony stream in `content` into `reasoning` + `content`.

WHY. The xhigh snapshot stores the whole reply as one raw Harmony stream:

    " to=self<|message|>{CoT}<|eom|><|start|>assistant to=user<|message|>{answer}"

Training re-renders rows through PineDrift's own chat_template, and that template
(a) wraps the assistant turn in its own `<|start|>assistant to=user<|message|>`
header and (b) runs the content through `esc()`, which deliberately inserts a
space into every control token as injection defence. Fed the raw stream, it
produces

    <|start|>assistant to=user<|message|> to=self< |message|>{CoT}< |eom|>...

i.e. a doubled header and control tokens shattered into ordinary BPE pieces --
a target sequence that cannot occur at inference, and nothing errors.

The template already knows how to render a Harmony reply correctly: its assistant
branch reads `reasoning`/`reasoning_content` and `content` separately and emits
`to=self ... <|eom|>` then `to=user ... <|eot|>`. So split the stream into those
two fields and the stock template reconstructs the real stream exactly. Verified
by `--verify`, which renders a row back and compares to the original.

Drop rules, both on by default:
  * no `to=user` separator  -- the reply hit the 8192 cap mid-CoT and never
    answered (15.6%). Kept as reasoning-only these would make the template append
    an `<|eom|>` at an arbitrary cut point, teaching a false end-of-CoT. The
    snapshot README argues plain truncation is still valid next-token data, and
    that is true of the verbatim stream, but not once a terminator is synthesised.
  * degenerate tail loop (6.0%) -- detector lifted from
    ~haoguo/pinedrift_synth/pinedrift_filter.py, see the README's "Known issues".
"""
import glob
import json
import os
import sys

SELF_PREFIX = " to=self<|message|>"
USER_SEP = "<|eom|><|start|>assistant to=user<|message|>"


def loop_period(s, maxp=400, minrep=4, tail=4000):
    """Shortest tail period p <= maxp repeated >= minrep times, else 0.

    Verbatim from pinedrift_filter.py; period-agnostic because the observed loops
    run from ~38 to ~200 chars.
    """
    t = s[-tail:]
    for p in range(10, maxp + 1):
        if len(t) < p * minrep:
            break
        unit = t[-p:]
        if all(t[-p * (k + 1):-p * k or None] == unit for k in range(1, minrep)):
            return p
    return 0


def split_reply(content):
    """-> (reasoning, answer) or None if the row is not a usable complete reply."""
    if not content or not content.startswith(SELF_PREFIX):
        return None
    body = content[len(SELF_PREFIX):]
    if body.count(USER_SEP) != 1:
        return None                      # no answer, or an unexpected shape
    cot, answer = body.split(USER_SEP)
    if not answer.strip():
        return None
    return cot, answer


def convert_row(r):
    conv = r.get("conversations") or []
    asst = [m for m in conv if m.get("role") == "assistant"]
    if len(asst) != 1:
        return None, "not_single_assistant"
    c = asst[0].get("content") or ""
    if loop_period(c):
        return None, "degenerate"
    parts = split_reply(c)
    if parts is None:
        return None, "no_answer"
    cot, answer = parts
    out = []
    for m in conv:
        if m.get("role") == "assistant":
            out.append({"role": "assistant", "reasoning": cot, "content": answer})
        else:
            out.append({"role": m["role"], "content": m["content"]})
    # `messages` here is a user-only stub; hf_streaming_dataset prefers
    # `conversations` so it is harmless, but drop it rather than ship a decoy.
    return {k: v for k, v in r.items() if k != "messages"} | {"conversations": out}, "ok"


def verify(src, model, n=5):
    """Render converted rows through the stock template; require an exact rebuild."""
    sys.path.insert(0, "/lustre/modelopt-pinedrift")
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model)
    tok.chat_template = open(f"{model}/chat_template.jinja").read()
    checked = 0
    for f in sorted(glob.glob(src + "/train-*.jsonl"))[:2]:
        for line in open(f):
            r = json.loads(line)
            orig = [m for m in (r.get("conversations") or []) if m["role"] == "assistant"]
            new, why = convert_row(r)
            if why != "ok":
                continue
            txt = tok.apply_chat_template(new["conversations"], tokenize=False,
                                          reasoning_effort="xhigh")
            want = "<|start|>assistant" + orig[0]["content"] + "<|eot|>"
            assert want in txt, (
                "rebuild mismatch\n  want tail: %r\n  got  tail: %r"
                % (want[:160], txt[txt.find("<|start|>assistant to=self"):][:160]))
            ids = tok(txt, add_special_tokens=False)["input_ids"]
            for t, i in [("<|start|>", 200022), ("<|message|>", 200023),
                         ("<|eom|>", 200007), ("<|eot|>", 200008)]:
                assert ids.count(i) >= 1, f"control token {t} absent from the rendered ids"
            assert "Reasoning strength: 512." in txt, "reasoning_effort did not reach the system turn"
            checked += 1
            if checked >= n:
                print(f"verify: {checked} rows rebuild EXACTLY, control tokens intact, "
                      f"system turn carries xhigh")
                return
    raise SystemExit("verify: no convertible row found")


if __name__ == "__main__":
    src = sys.argv[1]
    if sys.argv[2] == "--verify":
        verify(src, sys.argv[3])
        raise SystemExit(0)
    dst = sys.argv[2]
    os.makedirs(dst, exist_ok=True)
    import collections
    c = collections.Counter()
    for f in sorted(glob.glob(src + "/train-*.jsonl")):
        out = []
        for line in open(f):
            row, why = convert_row(json.loads(line))
            c[why] += 1
            if row is not None:
                out.append(json.dumps(row, ensure_ascii=False) + "\n")
        with open(os.path.join(dst, os.path.basename(f)), "w") as fh:
            fh.writelines(out)
    tot = sum(c.values())
    print(f"rows={tot} kept={c['ok']} ({c['ok'] / max(tot, 1):.2%})")
    for k, v in sorted(c.items()):
        if k != "ok":
            print(f"  dropped, {k:22} {v} ({v / max(tot, 1):.2%})")
