"""TRAIN-side activation dump for one fixed sample.

Runs the real ModelOpt training forward (DSparkFakeBase + DFlashGemma4 draft) on
ONE corpus row, at ONE fixed anchor, and dumps every intermediate tensor.

Determinism: _sample_anchor_positions() uses torch.rand, so it is monkeypatched
to return a single fixed anchor. Both pipelines must diff the SAME position or
the comparison is meaningless.

Dumps (all bf16->fp32, CPU):
  input_ids, anchor, noise_ids
  aux_hidden          [S, 5*H]   the target's captured hidden states (fc input)
  ctx_fc              [S, H]     fc(aux)
  ctx_hidden_norm     [S, H]     hidden_norm(fc(aux))   <- what draft attends to
  noise_embedding     [B, H]     draft block input
  layer_{i}_out       [B, H]     each draft layer output
  draft_final         [B, H]     after final norm
  backbone_logits     [B, V]
  final_logits        [B, V]     after Markov head
  top1                [B]        argmax token ids
"""

import argparse
import json
import os

import torch

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", required=True, help="training checkpoint dir")
ap.add_argument("--base", required=True, help="base Gemma-4-E4B-it dir")
ap.add_argument("--corpus", required=True, help="jsonl corpus file")
ap.add_argument("--row", type=int, default=0)
ap.add_argument("--anchor", type=int, default=64, help="FIXED anchor position")
ap.add_argument("--seqlen", type=int, default=256)
ap.add_argument("--chat-template", default=None)
ap.add_argument("--out", required=True)
ap.add_argument("--force-embed-scale", type=float, default=None,
                help="override FakeBaseConfig.embed_scale. Checkpoints saved BEFORE the "
                     "embed-scale fix have no such field, so it defaults to 1.0 and the "
                     "fix is a no-op; pass sqrt(hidden_size) to exercise the fixed math.")
ap.add_argument("--aux-from", default=None,
                help="eval_dump.pt to source aux_hidden/base_final_hidden from, so BOTH "
                     "pipelines consume byte-identical target activations")
args = ap.parse_args()

os.makedirs(args.out, exist_ok=True)
DUMP = {}


def save(name, t):
    if t is None:
        return
    if isinstance(t, torch.Tensor):
        DUMP[name] = t.detach().float().cpu()
        print(f"  [dump] {name:<24} {tuple(t.shape)}")


def main():
    import modelopt.torch.opt as mto
    import modelopt.torch.speculative as mtsp  # noqa: F401
    from transformers import AutoTokenizer

    # Same hook main.py installs at import time: makes from_pretrained() replay
    # the saved modelopt_state (i.e. re-run mtsp.convert) BEFORE loading weights.
    # Without it AutoModelForCausalLM returns a bare FakeBaseModel and every
    # dflash_module.* tensor is reported UNEXPECTED and silently dropped.
    mto.enable_huggingface_checkpointing()

    torch.manual_seed(0)
    dev = "cuda"

    tok = AutoTokenizer.from_pretrained(args.base)
    if args.chat_template and os.path.exists(args.chat_template):
        tok.chat_template = open(args.chat_template).read()

    # ---- pick the sample ----
    row = None
    with open(args.corpus) as f:
        for i, line in enumerate(f):
            if i == args.row:
                row = json.loads(line)
                break
    assert row is not None, f"row {args.row} not found"
    convs = row.get("conversations") or row.get("messages")
    print("sample keys:", list(row.keys()))

    msgs = []
    for m in convs:
        r = m.get("role") or m.get("from")
        c = m.get("content") or m.get("value")
        r = {"human": "user", "gpt": "assistant"}.get(r, r)
        msgs.append({"role": r, "content": c})
    text = tok.apply_chat_template(msgs, tokenize=False)
    ids = tok(text, return_tensors="pt").input_ids[:, : args.seqlen].to(dev)
    print("input_ids:", tuple(ids.shape))
    save("input_ids", ids[0])

    # ---- build the model exactly as training does ----
    from transformers import AutoModelForCausalLM

    print("loading fake-base + draft from", args.ckpt)
    model = AutoModelForCausalLM.from_pretrained(
        args.ckpt, dtype=torch.bfloat16, trust_remote_code=True
    ).to(dev).eval()

    if args.force_embed_scale is not None:
        emb = model._base_model_embeddings
        old = getattr(emb, "embed_scale", 1.0)
        emb.embed_scale = float(args.force_embed_scale)
        model.config.embed_scale = float(args.force_embed_scale)
        print(f"embed_scale {old} -> {emb.embed_scale}  (class {type(emb).__name__})")
    print("model class:", type(model).__name__)

    # ---- force a FIXED anchor ----
    ANCHOR = args.anchor

    def fixed_anchors(self, seq_len, loss_mask, device):
        a = torch.tensor([[ANCHOR]], dtype=torch.long, device=device)
        k = torch.ones(1, 1, dtype=torch.bool, device=device)
        return a, k

    type(model)._sample_anchor_positions = fixed_anchors
    print(f"anchor FORCED to {ANCHOR}")

    # ---- hooks on the draft module ----
    dm = model.dflash_module
    save("hidden_norm.weight", dm.hidden_norm.weight)
    save("fc.weight", dm.fc.weight)

    dm.fc.register_forward_hook(lambda m, i, o: save("ctx_fc", o[0]))
    dm.hidden_norm.register_forward_hook(lambda m, i, o: save("ctx_hidden_norm", o[0]))
    for li, layer in enumerate(dm.layers):
        layer.register_forward_hook(
            lambda m, i, o, li=li: save(f"layer_{li}_out", (o[0] if isinstance(o, tuple) else o)[0])
        )
        # Sub-layer taps so a divergence localizes to a STEP, not just a layer.
        # Names mirror dump_vllm_real.py's so compare.py can pair them directly.
        a = layer.self_attn

        def tap(name, mod):
            mod.register_forward_hook(
                lambda m, i, o, n=name: save(
                    n, (o[0] if isinstance(o, tuple) else o).reshape(-1, (o[0] if isinstance(o, tuple) else o).shape[-1])
                )
            )

        # Wrap the Gemma4 draft attention so its INLINE steps (rope, attn) are
        # visible: ModelOpt computes them inside forward(), not as submodules,
        # so plain hooks cannot see them.
        def wrap_attn(a=a, li=li):
            import types

            # ModelOpt has its OWN apply_rotary_pos_emb (Q takes the LAST q_len
            # positions, K takes all) -- the Qwen3 one assumes Q and K are the
            # same length and blows up here.
            from modelopt.torch.speculative.plugins.modeling_dflash import (
                apply_rotary_pos_emb as _arpe,
            )

            def fwd(self, hidden_states, target_hidden, position_embeddings, attention_mask=None):
                bsz, q_len, _ = hidden_states.shape
                ctx_len = target_hidden.shape[1]
                q = self.q_proj(hidden_states).view(bsz, q_len, -1, self.head_dim)
                q = self.q_norm(q).transpose(1, 2)
                k_ctx = self.k_proj(target_hidden)
                k_noise = self.k_proj(hidden_states)
                save(f"L{li}_k_projraw_ctx", k_ctx[0])
                save(f"L{li}_k_projraw_block", k_noise[0])
                k = torch.cat([k_ctx, k_noise], dim=1).view(
                    bsz, ctx_len + q_len, -1, self.head_dim)
                k = self.k_norm(k).transpose(1, 2)
                # vLLM's _kv_proj returns K *after* k_norm, so compare that form.
                _kn = k.transpose(1, 2).reshape(bsz, ctx_len + q_len, -1)
                save(f"L{li}_k_proj_ctx", _kn[0, :ctx_len])
                save(f"L{li}_k_proj_block", _kn[0, ctx_len:])
                v_ctx, v_noise = self._project_v(target_hidden, hidden_states, k_ctx, k_noise)
                v = torch.cat([v_ctx, v_noise], dim=1).view(
                    bsz, ctx_len + q_len, -1, self.head_dim)
                v = self.v_norm(v).transpose(1, 2)
                save(f"L{li}_v_proj_block", v[0, :, ctx_len:].transpose(0, 1).reshape(q_len, -1))
                cos, sin = position_embeddings
                q, k = _arpe(q, k, cos, sin)
                save(f"L{li}_q_rope", q[0].transpose(0, 1).reshape(q_len, -1))
                save(f"L{li}_k_rope_ctx", k[0, :, :ctx_len].transpose(0, 1).reshape(ctx_len, -1))
                save(f"L{li}_k_rope_block", k[0, :, ctx_len:].transpose(0, 1).reshape(q_len, -1))
                save(f"L{li}_attn_in_q", q[0].transpose(0, 1).reshape(q_len, -1))
                save(f"L{li}_attn_in_k", k[0].transpose(0, 1).reshape(ctx_len + q_len, -1))
                save(f"L{li}_attn_in_v", v[0].transpose(0, 1).reshape(ctx_len + q_len, -1))
                if attention_mask is not None and li == 0:
                    am = attention_mask
                    save("attn_mask_shape", torch.tensor(list(am.shape), dtype=torch.float))
                    m2 = am[0, 0] if am.dim() == 4 else am
                    save("attn_mask_visible_per_q",
                         (m2[-q_len:] > -1e3).float().sum(-1))
                    # Dump the ACTUAL mask rows fed to SDPA plus the exact K/V
                    # tensors in their pre-expand layout, so an offline rebuild
                    # has nothing left to guess.
                    save("attn_mask_rows", m2[-q_len:])
                    save("L0_k_bhsd", k[0].reshape(k.shape[1], -1))
                    save("L0_v_bhsd", v[0].reshape(v.shape[1], -1))
                    save("L0_q_bhsd", q[0].reshape(q.shape[1], -1))
                    save("L0_kvheads", torch.tensor(
                        [k.shape[1], v.shape[1], q.shape[1], float(self.scaling)]))
                attn_fn = self._get_attn_fn()
                ao, _ = attn_fn(self, q, k, v, attention_mask, dropout=0.0,
                                scaling=self.scaling, sliding_window=self.sliding_window)
                ao = ao.reshape(bsz, q_len, -1)
                save(f"L{li}_attn_out", ao[0])
                if li == 0:
                    import torch.nn.functional as _F

                    kk = k if k.shape[1] == q.shape[1] else k.repeat_interleave(
                        q.shape[1] // k.shape[1], dim=1)
                    vv2 = v if v.shape[1] == q.shape[1] else v.repeat_interleave(
                        q.shape[1] // v.shape[1], dim=1)
                    ref = _F.scaled_dot_product_attention(
                        q, kk, vv2, attn_mask=attention_mask, scale=self.scaling)
                    ref = ref.transpose(1, 2).contiguous().reshape(bsz, q_len, -1)
                    d = (ref - ao).abs().max().item()
                    print(f"    [selfcheck] L0 inline-SDPA vs attn_fn max|d| = {d:.4e}")
                    save("selfcheck_inline_sdpa", ref[0])
                return self.o_proj(ao)

            a.forward = types.MethodType(fwd, a)

        wrap_attn()

        tap(f"L{li}_input_layernorm", layer.input_layernorm)
        tap(f"L{li}_q_proj", a.q_proj)
        tap(f"L{li}_q_norm", a.q_norm)
        tap(f"L{li}_o_proj", a.o_proj)
        tap(f"L{li}_post_attn_norm", layer.post_attention_layernorm)
        tap(f"L{li}_pre_ffn_norm", layer.pre_feedforward_layernorm)
        tap(f"L{li}_mlp", layer.mlp)
        tap(f"L{li}_post_ffn_norm", layer.post_feedforward_layernorm)
        # eval dumps L{i}_out (after layer_scalar); layer_{i}_out already covers it
        layer.register_forward_hook(
            lambda m, i, o, li=li: save(
                f"L{li}_out", (o[0] if isinstance(o, tuple) else o)[0]))
    dm.norm.register_forward_hook(lambda m, i, o: save("draft_final", o[0]))

    # capture the draft-module inputs
    def pre(m, a, kw):
        if "noise_embedding" in kw:
            save("noise_embedding", kw["noise_embedding"][0])
        if "target_hidden" in kw:
            save("aux_hidden", kw["target_hidden"][0])
        if "position_ids" in kw:
            save("position_ids", kw["position_ids"][0])
        return None

    dm.register_forward_pre_hook(pre, with_kwargs=True)

    # ---- forward ----
    lm = model._base_model_lm_head
    orig_lm = lm.forward
    seen = []

    def lm_hook(x):
        out = orig_lm(x)
        seen.append(out)
        return out

    lm.forward = lm_hook

    # Streaming/offline mode: the base layers were deleted, so aux hidden states
    # must be supplied exactly as the vLLM producer would. Reuse the EVAL dump so
    # both sides consume byte-identical target activations -- any divergence is
    # then provably inside the draft, not in the target capture.
    assert args.aux_from, "--aux-from is required for an offline/streaming checkpoint"
    ev = torch.load(args.aux_from, map_location="cpu")
    aux = ev["aux_hidden"].to(dev, torch.bfloat16).unsqueeze(0)
    fin = ev["base_final_hidden"].to(dev, torch.bfloat16).unsqueeze(0)
    save("aux_hidden", aux[0])
    bmo = {"aux_hidden_states": aux, "base_model_hidden_states": fin}

    # The offline guard triggers on `not self.training`; the streaming forward is
    # the training forward, so run in train() mode under no_grad.
    model.train()
    with torch.no_grad():
        out = model(input_ids=ids, attention_mask=torch.ones_like(ids),
                    labels=ids, base_model_outputs=bmo)

    print("loss:", float(out.loss) if getattr(out, "loss", None) is not None else None)
    if getattr(out, "train_acc", None) is not None:
        print("train_acc:", out.train_acc)

    if seen:
        bl = seen[-1]
        save("backbone_logits", bl.reshape(-1, bl.shape[-1]))
        save("top1_backbone", bl.reshape(-1, bl.shape[-1]).argmax(-1))

    torch.save(DUMP, os.path.join(args.out, "train_dump.pt"))
    print("\nwrote", os.path.join(args.out, "train_dump.pt"), f"({len(DUMP)} tensors)")


if __name__ == "__main__":
    main()
