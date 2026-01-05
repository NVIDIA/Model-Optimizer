import logging

import torch
import torch.nn.functional as F

try:
    from flash_attn import flash_attn_func

    FLASH_ATTENTION_AVAILABLE = True
except ImportError:
    flash_attn_func = None
    FLASH_ATTENTION_AVAILABLE = False


try:
    import xformers

    XFORMERS_AVAILABLE = True
except ImportError:
    xformers = None
    XFORMERS_AVAILABLE = False


SDP_BATCH_LIMIT = 2**15


def attention_pytorch(
    q,
    k,
    v,
    heads,
    mask=None,
    attn_precision=None,
    skip_reshape=False,
    skip_output_reshape=False,
    **kwargs,
):
    if skip_reshape:
        b, _, _, dim_head = q.shape
    else:
        b, _, dim_head = q.shape
        dim_head //= heads
        q, k, v = map(
            lambda t: t.view(b, -1, heads, dim_head).transpose(1, 2),
            (q, k, v),
        )

    if mask is not None:
        # add a batch dimension if there isn't already one
        if mask.ndim == 2:
            mask = mask.unsqueeze(0)
        # add a heads dimension if there isn't already one
        if mask.ndim == 3:
            mask = mask.unsqueeze(1)
        # ensure mask is bool dtype for scaled_dot_product_attention
        if mask.dtype != torch.bool:
            mask = mask.to(torch.bool)

    if b <= SDP_BATCH_LIMIT:
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=0.0, is_causal=False)
        if not skip_output_reshape:
            out = out.transpose(1, 2).reshape(b, -1, heads * dim_head)
    else:
        out = torch.empty(
            (b, q.shape[2], heads * dim_head),
            dtype=q.dtype,
            layout=q.layout,
            device=q.device,
        )
        for i in range(0, b, SDP_BATCH_LIMIT):
            m = mask
            if mask is not None:
                if mask.shape[0] > 1:
                    m = mask[i : i + SDP_BATCH_LIMIT]

            out[i : i + SDP_BATCH_LIMIT] = (
                F.scaled_dot_product_attention(
                    q[i : i + SDP_BATCH_LIMIT],
                    k[i : i + SDP_BATCH_LIMIT],
                    v[i : i + SDP_BATCH_LIMIT],
                    attn_mask=m,
                    dropout_p=0.0,
                    is_causal=False,
                )
                .transpose(1, 2)
                .reshape(-1, q.shape[2], heads * dim_head)
            )
    return out


def attention_xformers(
    q,
    k,
    v,
    heads,
    mask=None,
    attn_precision=None,
    skip_reshape=False,
    skip_output_reshape=False,
    **kwargs,
):
    b = q.shape[0]
    dim_head = q.shape[-1]

    if skip_reshape:
        # b h k d -> b k h d
        q, k, v = map(
            lambda t: t.permute(0, 2, 1, 3),
            (q, k, v),
        )
    # actually do the reshaping
    else:
        dim_head //= heads
        q, k, v = map(
            lambda t: t.reshape(b, -1, heads, dim_head),
            (q, k, v),
        )

    if torch.compiler.is_compiling():
        q, k, v = map(lambda t: t.contiguous(), (q, k, v))

    if mask is not None:
        # add a singleton batch dimension
        if mask.ndim == 2:
            mask = mask.unsqueeze(0)
        # add a singleton heads dimension
        if mask.ndim == 3:
            mask = mask.unsqueeze(1)
        # ensure mask is bool dtype
        if mask.dtype != torch.bool:
            mask = mask.to(torch.bool)
        # pad to a multiple of 8
        pad = 8 - mask.shape[-1] % 8
        # the xformers docs says that it's allowed to have a mask of shape (1, Nq, Nk)
        # but when using separated heads, the shape has to be (B, H, Nq, Nk)
        # in flux, this matrix ends up being over 1GB
        # here, we create a mask with the same batch/head size as the input mask (potentially singleton or full)
        mask_out = torch.empty(
            [mask.shape[0], mask.shape[1], q.shape[1], mask.shape[-1] + pad],
            dtype=q.dtype,
            device=q.device,
        )

        mask_out[..., : mask.shape[-1]] = mask
        # doesn't this remove the padding again??
        mask = mask_out[..., : mask.shape[-1]]
        mask = mask.expand(b, heads, -1, -1)

    out = xformers.ops.memory_efficient_attention(q, k, v, attn_bias=mask)

    out = out.permute(0, 2, 1, 3) if skip_output_reshape else out.reshape(b, -1, heads * dim_head)

    return out


try:

    @torch.library.custom_op("flash_attention::flash_attn", mutates_args=())
    def flash_attn_wrapper(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        dropout_p: float = 0.0,
        causal: bool = False,
    ) -> torch.Tensor:
        return flash_attn_func(q, k, v, dropout_p=dropout_p, causal=causal)

    @flash_attn_wrapper.register_fake
    def flash_attn_fake(q, k, v, dropout_p=0.0, causal=False):
        # Output shape is the same as q
        return q.new_empty(q.shape)

except AttributeError as error:
    ERROR_MSG = str(error)

    def flash_attn_wrapper(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        dropout_p: float = 0.0,
        causal: bool = False,
    ) -> torch.Tensor:
        assert False, f"Could not define flash_attn_wrapper: {ERROR_MSG}"


def attention_flash(
    q,
    k,
    v,
    heads,
    mask=None,
    attn_precision=None,
    skip_reshape=False,
    skip_output_reshape=False,
    **kwargs,
):
    if skip_reshape:
        b, _, _, dim_head = q.shape
    else:
        b, _, dim_head = q.shape
        dim_head //= heads
        q, k, v = map(
            lambda t: t.view(b, -1, heads, dim_head).transpose(1, 2),
            (q, k, v),
        )

    if mask is not None:
        # add a batch dimension if there isn't already one
        if mask.ndim == 2:
            mask = mask.unsqueeze(0)
        # add a heads dimension if there isn't already one
        if mask.ndim == 3:
            mask = mask.unsqueeze(1)
        # ensure mask is bool dtype for scaled_dot_product_attention fallback
        if mask.dtype != torch.bool:
            mask = mask.to(torch.bool)

    try:
        if mask is not None:
            raise RuntimeError("Mask must not be set for Flash attention")
        out = flash_attn_wrapper(
            q.transpose(1, 2),
            k.transpose(1, 2),
            v.transpose(1, 2),
            dropout_p=0.0,
            causal=False,
        ).transpose(1, 2)
    except Exception as e:
        logging.warning(f"Flash Attention failed, using default SDPA: {e}")
        out = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=0.0, is_causal=False)
    if not skip_output_reshape:
        out = out.transpose(1, 2).reshape(b, -1, heads * dim_head)
    return out


if FLASH_ATTENTION_AVAILABLE:
    logging.info("Using Flash Attention")
    optimized_attention = attention_flash
elif XFORMERS_AVAILABLE:
    logging.info("Using xformers attention")
    optimized_attention = attention_xformers
else:
    logging.info("Using PyTorch attention")
    optimized_attention = attention_pytorch


optimized_attention_masked = optimized_attention
