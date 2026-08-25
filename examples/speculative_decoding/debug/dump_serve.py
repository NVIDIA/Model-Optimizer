"""SERVING-side activation dump for the same fixed sample and anchor.

Recomputes the drafter forward with the exported (vLLM-format) checkpoint,
following vLLM's Gemma4DSpark math: `embed_tokens(ids) * sqrt(hidden_size)`,
`fc` -> `hidden_norm` on the aux states, k_eq_v (V from the K projection, no
v_proj), and the block attention mask measured from the training side.

Scope, stated plainly: this is a faithful REIMPLEMENTATION of the serving math,
not vLLM's own kernels. It agrees with the ModelOpt training path to <2%
(bf16 accumulation) with identical top-1, which is enough to localize logic
bugs. It is NOT a substitute for validating vLLM's paged-attention kernel,
which needs a populated KV cache and attention metadata.

Dumps the same tensor names as dump_train.py so the two are directly diffable.
"""

import argparse
import json
import os

import torch
import torch.nn.functional as F

ap = argparse.ArgumentParser()
ap.add_argument("--base", required=True)
ap.add_argument("--draft", required=True, help="vLLM-format drafter dir")
ap.add_argument("--corpus", required=True)
ap.add_argument("--row", type=int, default=0)
ap.add_argument("--anchor", type=int, default=64)
ap.add_argument("--seqlen", type=int, default=256)
ap.add_argument("--capture-ids", default="6,12,18,24,36,42")
ap.add_argument("--chat-template", default=None)
ap.add_argument("--out", required=True)
args = ap.parse_args()

os.makedirs(args.out, exist_ok=True)
DUMP = {}


def save(name, t):
    if isinstance(t, torch.Tensor):
        DUMP[name] = t.detach().float().cpu()
        print(f"  [dump] {name:<24} {tuple(t.shape)}")


def rms(x, w, eps):
    v = x.float()
    v = v * torch.rsqrt(v.pow(2).mean(-1, keepdim=True) + eps)
    return (v * w.float()).to(x.dtype)


def main():
    from safetensors.torch import load_file
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dev, DT = "cuda", torch.bfloat16
    cfg = json.load(open(os.path.join(args.draft, "config.json")))
    H = cfg["hidden_size"]
    NL = cfg["num_hidden_layers"]
    NH = cfg["num_attention_heads"]
    HD = cfg.get("global_head_dim") or cfg["head_dim"]
    NKV = cfg.get("num_global_key_value_heads") or cfg["num_key_value_heads"]
    EPS = cfg["rms_norm_eps"]
    BS = cfg["block_size"]
    MASK = cfg["dflash_config"]["mask_token_id"]
    THETA = cfg.get("rope_theta", 10000.0)

    tok = AutoTokenizer.from_pretrained(args.base)
    if args.chat_template and os.path.exists(args.chat_template):
        tok.chat_template = open(args.chat_template).read()

    row = None
    with open(args.corpus) as f:
        for i, line in enumerate(f):
            if i == args.row:
                row = json.loads(line)
                break
    convs = row.get("conversations") or row.get("messages")
    msgs = []
    for m in convs:
        r = m.get("role") or m.get("from")
        c = m.get("content") or m.get("value")
        r = {"human": "user", "gpt": "assistant"}.get(r, r)
        msgs.append({"role": r, "content": c})
    text = tok.apply_chat_template(msgs, tokenize=False)
    ids = tok(text, return_tensors="pt").input_ids[:, : args.seqlen].to(dev)
    save("input_ids", ids[0])

    # ---- aux hidden states from the REAL base, at the TRAINING capture ids ----
    base = AutoModelForCausalLM.from_pretrained(
        args.base, dtype=DT, device_map=dev
    ).eval()
    with torch.no_grad():
        o = base(ids, output_hidden_states=True)
    hs = o.hidden_states
    cap = [int(x) for x in args.capture_ids.split(",")]
    print("capture ids (post-layer, vLLM convention):", cap)
    # first N-1 are aux, last is the final/base hidden (see hf_streaming_dataset)
    aux = torch.cat([hs[i][0] for i in cap[:-1]], dim=-1).to(DT)
    save("aux_hidden", aux)
    save("base_final_hidden", hs[cap[-1]][0].to(DT))
    del base
    torch.cuda.empty_cache()

    # ---- exported drafter weights ----
    sd = {k: v.to(dev, DT) for k, v in load_file(os.path.join(args.draft, "model.safetensors")).items()}
    save("hidden_norm.weight", sd["hidden_norm.weight"])
    save("fc.weight", sd["fc.weight"])

    ctx_fc = F.linear(aux, sd["fc.weight"])
    save("ctx_fc", ctx_fc)
    ctx = rms(ctx_fc, sd["hidden_norm.weight"], EPS)
    save("ctx_hidden_norm", ctx)

    # ---- draft block input at the fixed anchor (vLLM: embed * sqrt(H)) ----
    A = args.anchor
    blk = torch.full((BS,), MASK, dtype=torch.long, device=dev)
    blk[0] = ids[0, A]
    save("noise_ids", blk)
    x = sd["embed_tokens.weight"][blk] * (H ** 0.5)
    save("noise_embedding", x)

    # Use the FULL sequence as context, exactly as training does, so position_ids
    # and the attended key set match on both sides. Causality is enforced by the
    # mask below, not by truncating the context.
    nctx = ctx.shape[0]
    c = ctx
    cpos = torch.arange(nctx, device=dev)
    qpos = torch.arange(A, A + BS, device=dev)
    save("position_ids", torch.cat([cpos, qpos]))

    def rope(hd, pos):
        inv = 1.0 / (THETA ** (torch.arange(0, hd, 2, device=dev).float() / hd))
        f = pos.float()[:, None] * inv[None, :]
        return torch.cos(f), torch.sin(f)

    def apply(t, cos, sin):
        d = t.shape[-1]
        t1, t2 = t[..., : d // 2], t[..., d // 2 :]
        cc = cos[:, None, :].to(t.dtype)
        ss = sin[:, None, :].to(t.dtype)
        return torch.cat([t1 * cc - t2 * ss, t2 * cc + t1 * ss], dim=-1)

    cos_c, sin_c = rope(HD, cpos)
    cos_q, sin_q = rope(HD, qpos)
    ones = torch.ones(HD, device=dev, dtype=DT)

    for li in range(NL):
        p = f"layers.{li}."
        res = x
        xn = rms(x, sd[p + "input_layernorm.weight"], EPS)
        q = rms(F.linear(xn, sd[p + "self_attn.q_proj.weight"]).view(BS, NH, HD),
                sd[p + "self_attn.q_norm.weight"], EPS)
        kc = rms(F.linear(c, sd[p + "self_attn.k_proj.weight"]).view(nctx, NKV, HD),
                 sd[p + "self_attn.k_norm.weight"], EPS)
        kn = rms(F.linear(xn, sd[p + "self_attn.k_proj.weight"]).view(BS, NKV, HD),
                 sd[p + "self_attn.k_norm.weight"], EPS)
        vc = rms(F.linear(c, sd[p + "self_attn.k_proj.weight"]).view(nctx, NKV, HD), ones, EPS)
        vn = rms(F.linear(xn, sd[p + "self_attn.k_proj.weight"]).view(BS, NKV, HD), ones, EPS)
        q = apply(q, cos_q, sin_q)
        kc = apply(kc, cos_c, sin_c)
        kn = apply(kn, cos_q, sin_q)
        k = torch.cat([kc, kn], 0).repeat_interleave(NH // NKV, dim=1)
        v = torch.cat([vc, vn], 0).repeat_interleave(NH // NKV, dim=1)
        _q = q.permute(1, 0, 2)[None].float()          # [1,H,q,D]
        _k = k.permute(1, 0, 2)[None].float()          # [1,H,S,D]
        _v = v.permute(1, 0, 2)[None].float()
        # Measured from the training mask tensor (not guessed): all 8 draft
        # queries share ONE window over context [0, A) plus the full 8-token
        # block. Position A itself is NOT visible, and there is no per-query
        # causal ramp -- a DSpark block is predicted in one shot.
        if li == 0:
            ctx_allow = (cpos < A)[None, :].expand(BS, nctx)
            blk_allow = torch.ones(BS, BS, dtype=torch.bool, device=dev)
            allow = torch.cat([ctx_allow, blk_allow], dim=1)[None]  # [1, B, nctx+B]
        _m = torch.zeros(1, 1, BS, k.shape[0], device=dev).masked_fill(
            ~allow, float("-inf"))
        att = F.scaled_dot_product_attention(_q, _k, _v, attn_mask=_m,
                                             scale=HD ** -0.5)
        # HF sdpa_attention_forward returns .transpose(1,2) -> [B,q,H,D]
        att = att.transpose(1, 2).contiguous().reshape(BS, NH * HD).to(DT)
        att = F.linear(att, sd[p + "self_attn.o_proj.weight"])
        att = rms(att, sd[p + "post_attention_layernorm.weight"], EPS)
        x = att + res
        res = x
        xn = rms(x, sd[p + "pre_feedforward_layernorm.weight"], EPS)
        g = F.linear(xn, sd[p + "mlp.gate_proj.weight"])
        u = F.linear(xn, sd[p + "mlp.up_proj.weight"])
        m = F.linear(F.silu(g.float()).to(DT) * u, sd[p + "mlp.down_proj.weight"])
        m = rms(m, sd[p + "post_feedforward_layernorm.weight"], EPS)
        x = m + res
        x = x * sd[p + "layer_scalar"]
        save(f"layer_{li}_out", x)

    x = rms(x, sd["norm.weight"], EPS)
    save("draft_final", x)
    logits = F.linear(x.float(), sd["lm_head.weight"].float())
    save("backbone_logits", logits)
    save("top1_backbone", logits.argmax(-1))
    print("\ntop1 tokens:", logits.argmax(-1).tolist())
    print("true next  :", ids[0, A + 1 : A + 1 + BS].tolist())

    torch.save(DUMP, os.path.join(args.out, "eval_dump.pt"))
    print("wrote", os.path.join(args.out, "eval_dump.pt"), f"({len(DUMP)} tensors)")


if __name__ == "__main__":
    main()
