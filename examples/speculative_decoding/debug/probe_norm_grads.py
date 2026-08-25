"""Separate the two candidate causes of the frozen draft norms.

All 32 draft RMSNorms sat at exactly 1.0 for 2000 steps. Two candidates:
  (A) bf16 resolution -- grads flow but each Adam step (~1e-4) is far below the
      bf16 ULP at 1.0 (7.81e-3), so every update rounds back to 1.0.
  (B) broken graph -- the norms receive literally zero/None gradient.

Run ONE real backward and print per-tensor grad norms. Then simulate what an
Adam step would actually do to a bf16 weight at 1.0 to confirm (A) quantitatively.
"""

import json
import os
import sys

import torch

CKPT, BASE, CORPUS, EVAL, TPL = sys.argv[1:6]


def main():
    import modelopt.torch.opt as mto
    import modelopt.torch.speculative as mtsp  # noqa: F401
    from transformers import AutoModelForCausalLM, AutoTokenizer

    mto.enable_huggingface_checkpointing()
    dev = "cuda"

    tok = AutoTokenizer.from_pretrained(BASE)
    if os.path.exists(TPL):
        tok.chat_template = open(TPL).read()

    with open(CORPUS) as f:
        row = json.loads(f.readline())
    convs = row.get("conversations") or row.get("messages")
    msgs = []
    for m in convs:
        r = m.get("role") or m.get("from")
        c = m.get("content") or m.get("value")
        msgs.append({"role": {"human": "user", "gpt": "assistant"}.get(r, r), "content": c})
    ids = tok(tok.apply_chat_template(msgs, tokenize=False),
              return_tensors="pt").input_ids[:, :256].to(dev)

    model = AutoModelForCausalLM.from_pretrained(
        CKPT, dtype=torch.bfloat16, trust_remote_code=True).to(dev)
    model.train()

    ev = torch.load(EVAL, map_location="cpu")
    bmo = {
        "aux_hidden_states": ev["aux_hidden"].to(dev, torch.bfloat16).unsqueeze(0),
        "base_model_hidden_states": ev["base_final_hidden"].to(dev, torch.bfloat16).unsqueeze(0),
    }

    out = model(input_ids=ids, attention_mask=torch.ones_like(ids),
                labels=ids, base_model_outputs=bmo)
    loss = out.loss
    print("loss:", float(loss))
    model.zero_grad(set_to_none=True)
    loss.backward()

    dm = model.dflash_module
    norms, linears = [], []
    for name, p in dm.named_parameters():
        if p.grad is None:
            g = None
        else:
            g = p.grad.float().abs().mean().item()
        (norms if ("norm" in name.lower()) else linears).append((name, g, p.requires_grad))

    print()
    print("=" * 84)
    print("  NORM parameters -- mean |grad|")
    print("=" * 84)
    print(f"  {'param':<52}{'requires_grad':>14}{'mean|grad|':>16}")
    zero = none = alive = 0
    for n, g, rg in norms[:16]:
        s = "None" if g is None else f"{g:.3e}"
        print(f"  {n:<52}{str(rg):>14}{s:>16}")
        if g is None:
            none += 1
        elif g == 0.0:
            zero += 1
        else:
            alive += 1
    for n, g, rg in norms[16:]:
        if g is None:
            none += 1
        elif g == 0.0:
            zero += 1
        else:
            alive += 1
    print(f"  ... total norms={len(norms)}  none={none}  exactly_zero={zero}  nonzero={alive}")

    print()
    print("=" * 84)
    print("  LINEAR parameters -- mean |grad| (control group, these DID train)")
    print("=" * 84)
    for n, g, rg in linears[:6]:
        s = "None" if g is None else f"{g:.3e}"
        print(f"  {n:<52}{str(rg):>14}{s:>16}")

    # ---- quantify candidate (A) ----
    print()
    print("=" * 84)
    print("  Would an Adam step actually move a bf16 weight at 1.0?")
    print("=" * 84)
    lr = 1e-4
    w = torch.ones(1, dtype=torch.bfloat16)
    print(f"  bf16(1.0)  - lr({lr:.0e})      = {float(w - lr):.8f}   "
          f"{'NO CHANGE' if float(w - lr) == 1.0 else 'moved'}")
    w2 = torch.full((1,), 0.02, dtype=torch.bfloat16)
    print(f"  bf16(0.02) - lr({lr:.0e})      = {float(w2 - lr):.8f}   "
          f"{'NO CHANGE' if float(w2 - lr) == 0.02 else 'moved'}")
    nxt = torch.tensor(1.0, dtype=torch.bfloat16)
    import struct
    i = struct.unpack("<I", struct.pack("<f", 1.0))[0] >> 16
    ulp = struct.unpack("<f", struct.pack("<I", (i + 1) << 16))[0] - 1.0
    print(f"  bf16 ULP at 1.0                = {ulp:.3e}   ({ulp/lr:.0f}x the step)")
    print()
    print("  VERDICT:")
    if none == len(norms):
        print("    (B) BROKEN GRAPH -- norms get no gradient at all.")
    elif zero == len(norms):
        print("    (B) BROKEN GRAPH -- norm gradients are exactly zero.")
    elif alive > 0:
        print(f"    (A) bf16 RESOLUTION -- {alive}/{len(norms)} norms DO receive gradient;")
        print("        the updates are simply too small to cross the bf16 ULP at 1.0.")
        print("        Fix = fp32 master weights (or init norms at 0 with a (1+w) form).")


if __name__ == "__main__":
    main()
