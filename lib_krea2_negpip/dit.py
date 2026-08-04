"""
Diffusion-model side of Krea 2 NegPiP.

Two levers, both applied inside attention, after every non-linear conditioning stage has
already run:

* the **value factor** scales the flagged tokens' value vectors — attention still scores
  them normally, then subtracts (or damps) what they contribute instead of adding it.
  This is NegPiP proper.
* the **logit bias** is added to the flagged tokens' attention scores, multiplying their
  softmax weight by `exp(bias)`.  Scaling a value vector cannot make the rest of the
  sequence attend to a token *more* — past a point it just saturates — so amplification
  needs the other side of the softmax.  A logit bias is also the only lever that survives
  the `RMSNorm`s intact, being additive in a space nothing normalises.

Krea 2 never passes an attention mask — `SingleStreamDiT.forward` hands `None` to every
block and to `txtfusion` — so the bias rides the `mask` argument that is already threaded
through `Attention.forward` to `attention_function`, and no attention math needs copying.
It does have to reach a backend that accepts an additive mask, hence the
`attention_function` rebind below, which is installed only when a prompt actually asks
for amplification.

Krea 2 is single-stream — `SingleStreamDiT.forward` concatenates the fused text context
and the image patches into one sequence and runs plain self-attention over the pair
(`backend/nn/krea.py`).  There is no cross-attention module to hook, so the flip goes on
the value projection of the blocks themselves, over the leading `txtlen` positions.  The
text-fusion refiner blocks are the same `Attention` class one stage earlier, over the
text alone; flipping there as well is optional and stronger.

`txtfusion.layerwise_blocks` are deliberately *not* touched: they run at
`(batch * seq, taps, dim)`, so their sequence axis is the 12-layer tap stack, and the
mask does not index it.

Two levels of hook, so none of Forge's own attention math is duplicated here:

* the selected `Attention.forward` parks the value factors on the module for the duration
  of one call, and substitutes the logit bias for the unused `mask` argument — it is the
  only place that sees `transformer_options`;
* that module's `wv` (`nn.Linear`) gets a wrapper that scales the rows it names.

`wv` returns `(B, L, kvheads * headdim)` before the rearrange into heads, so scaling a
whole row scales that token in every kv head at once.
"""

from functools import wraps
from typing import Any

import torch

from backend.attention import attention_pytorch
from backend.sampling import condition, sampling_function
from modules.processing import logger

from lib_krea2_negpip import (
    KREA2_TAP_DIM,
    KREA2_TAP_LAYERS,
    NEGPIP_BIAS_KEY,
    NEGPIP_BIAS_OPTION_KEY,
    NEGPIP_MASK_KEY,
    NEGPIP_OPTION_KEY,
    NEGPIP_ROLE_ATTR,
    ROLE_BLOCK,
    ROLE_REFINER,
)

ACTIVE_ATTR = "_krea2_negpip_active_mask"
ORIGINAL_FORWARD = "_krea2_negpip_original_forward"
ORIGINAL_WV = "_krea2_negpip_original_wv_forward"
ORIGINAL_ATTENTION = "_krea2_negpip_original_attention_function"

_warned_mask = False


def is_krea2_dit(dit: Any) -> bool:
    """Whether this diffusion model really is the Krea 2 `SingleStreamDiT`."""
    try:
        txtlayers = int(getattr(dit, "txtlayers", 0))
        txtdim = int(getattr(dit, "txtdim", 0))
    except (TypeError, ValueError):
        return False

    return hasattr(dit, "blocks") and hasattr(dit, "txtfusion") and hasattr(dit, "txtmlp") and hasattr(dit, "_unpack_context") and txtlayers == KREA2_TAP_LAYERS and txtdim == KREA2_TAP_DIM


def selected_blocks(count: int, start: int, end: int, stride: int) -> list[int]:
    if count <= 0:
        return []

    if end < start:
        start, end = end, start

    start = max(0, min(int(start), count - 1))
    end = max(0, min(int(end), count - 1))
    stride = max(1, int(stride))

    return [i for i in range(start, end + 1) if (i - start) % stride == 0]


def _attention_modules(dit: Any):
    """Every `Attention` the flip may legitimately reach, block-major then refiners."""
    for block in getattr(dit, "blocks", []) or []:
        yield getattr(block, "attn", None)

    txtfusion = getattr(dit, "txtfusion", None)
    for block in getattr(txtfusion, "refiner_blocks", []) or []:
        yield getattr(block, "attn", None)


# ================================================================================ #


def _align_batch(row: torch.Tensor, batch: int) -> torch.Tensor | None:
    """Broadcast a per-cond row up to the batch actually being sampled, or `None`.

    Cond and uncond are concatenated along the batch axis whenever their token counts
    match, and Forge may run several latents per cond.  A batch that is not a whole
    multiple of the row's own is not something to guess at — it means the row does not
    describe this call, and the caller leaves the tensor alone.
    """
    if row.shape[0] == batch:
        return row
    if row.shape[0] == 0 or batch % row.shape[0] != 0:
        return None

    return row.repeat(batch // row.shape[0], *([1] * (row.ndim - 1)))


def _apply_value_flip(v: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if not torch.is_tensor(v) or v.ndim != 3:
        return v

    length = min(int(mask.shape[1]), int(v.shape[1]))
    if length <= 0:
        return v

    m = _align_batch(mask[:, :length, :].to(device=v.device, dtype=v.dtype), v.shape[0])
    if m is None:
        return v

    v[:, :length, :] = v[:, :length, :] * m
    return v


def _attention_bias(x: torch.Tensor, bias: torch.Tensor) -> torch.Tensor | None:
    """`(B, 1, keys)` additive logit bias over this module's whole key axis, or `None`.

    The row only covers the text tokens; the main blocks attend over `cat(text, image)`,
    so it is zero-padded out to the full sequence.  `(B, 1, keys)` is what
    `attention_pytorch` unsqueezes to `(B, 1, 1, keys)` — one bias per batch row, which
    is what keeps cond and uncond from reading each other's row.
    """
    if not torch.is_tensor(bias) or bias.ndim != 3:
        return None

    batch, keys = int(x.shape[0]), int(x.shape[1])
    length = min(int(bias.shape[1]), keys)
    if length <= 0:
        return None

    row = _align_batch(bias[:, :length, :].to(device=x.device, dtype=x.dtype), batch)
    if row is None:
        return None

    out = x.new_zeros((batch, 1, keys))
    out[:, 0, :length] = row[:, :, 0]
    return out


def _restore_forward(obj: Any, original: Any):
    """Put `forward` back the way it was, instance dict included.

    Assigning the captured bound method back would leave an entry in `vars(obj)` that was
    never there, which keeps a reference cycle alive and shadows the class attribute for
    anyone patching later.
    """
    if getattr(original, "__func__", None) is getattr(type(obj), "forward", None):
        vars(obj).pop("forward", None)
    else:
        obj.forward = original


def _hook_wv(attn: Any, remove: bool):
    wv = getattr(attn, "wv", None)
    if wv is None:
        return

    if remove:
        original = getattr(wv, ORIGINAL_WV, None)
        if original is not None:
            if getattr(wv.forward, "_negpip", False):
                _restore_forward(wv, original)
            delattr(wv, ORIGINAL_WV)
        return

    if getattr(wv, ORIGINAL_WV, None) is not None:
        return

    original = wv.forward

    @wraps(original)
    def negpip_forward(*args, **kwargs):
        out = original(*args, **kwargs)
        mask = getattr(attn, ACTIVE_ATTR, None)
        if mask is None:
            return out
        return _apply_value_flip(out, mask)

    negpip_forward._negpip = True

    setattr(wv, ORIGINAL_WV, original)
    wv.forward = negpip_forward


def _hook_attention(attn: Any, role: str = "", remove: bool = False):
    if remove:
        original = getattr(attn, ORIGINAL_FORWARD, None)
        if original is not None:
            if getattr(attn.forward, "_negpip", False):
                _restore_forward(attn, original)
            delattr(attn, ORIGINAL_FORWARD)

        _hook_wv(attn, True)

        for name in (NEGPIP_ROLE_ATTR, ACTIVE_ATTR):
            if hasattr(attn, name):
                delattr(attn, name)
        return

    if getattr(attn, ORIGINAL_FORWARD, None) is not None:
        return

    _hook_wv(attn, False)
    original = attn.forward

    @wraps(original)
    def negpip_forward(x, freqs=None, mask=None, transformer_options={}):
        global _warned_mask

        options = transformer_options if isinstance(transformer_options, dict) else {}
        bias = options.get(NEGPIP_BIAS_OPTION_KEY, None)

        if bias is not None and torch.is_tensor(x):
            if mask is None:
                mask = _attention_bias(x, bias)
            elif not _warned_mask:
                #   Krea 2 has never passed one; if that changes, adding the bias to
                #   somebody else's mask is not obviously the right merge, so say so
                _warned_mask = True
                logger.warning("NegPiP: attention already carries a mask, skipping the emphasis bias")

        setattr(attn, ACTIVE_ATTR, options.get(NEGPIP_OPTION_KEY, None))
        try:
            return original(x, freqs, mask, transformer_options)
        finally:
            setattr(attn, ACTIVE_ATTR, None)

    negpip_forward._negpip = True

    setattr(attn, NEGPIP_ROLE_ATTR, role)
    setattr(attn, ORIGINAL_FORWARD, original)
    attn.forward = negpip_forward


def _hook_dit(dit: Any, remove: bool):
    if remove:
        original = getattr(dit, ORIGINAL_FORWARD, None)
        if original is not None:
            if getattr(dit.forward, "_negpip", False):
                _restore_forward(dit, original)
            delattr(dit, ORIGINAL_FORWARD)
        return

    if getattr(dit, ORIGINAL_FORWARD, None) is not None:
        return

    original = dit.forward

    @wraps(original)
    def negpip_forward(x, timesteps, context, attention_mask=None, transformer_options=None, **kwargs):
        mask = kwargs.pop(NEGPIP_MASK_KEY, None)
        bias = kwargs.pop(NEGPIP_BIAS_KEY, None)
        options = dict(transformer_options or {})

        #   conditioning arrives as (batch, 1, seq, features) — Krea 2 squeezes the
        #   singleton itself — so both rows arrive as (batch, 1, seq, 1)
        if torch.is_tensor(mask):
            options[NEGPIP_OPTION_KEY] = mask.reshape(mask.shape[0], -1, 1)
        if torch.is_tensor(bias):
            options[NEGPIP_BIAS_OPTION_KEY] = bias.reshape(bias.shape[0], -1, 1)

        return original(x, timesteps, context, attention_mask, options, **kwargs)

    negpip_forward._negpip = True

    setattr(dit, ORIGINAL_FORWARD, original)
    dit.forward = negpip_forward


def _hook_compile_conditions():
    """Teach `compile_conditions` the conditioning shape the text hook produces.

    It knows two: a bare tensor, or a dict carrying both `crossattn` and a pooled
    `vector`.  Krea 2 NegPiP makes a third — `crossattn` plus the per-token rows, no
    pooled vector — and the stock function would raise `KeyError: 'vector'` on it.

    A plain `Condition`, not `ConditionCrossAttn`: the rows are 1:1 with the conditioning
    they describe, and `ConditionCrossAttn.concat` would repeat them to a common length
    instead of refusing, which is exactly the misalignment worth failing loudly on.

    Installed on demand and never removed.  Forge caches compiled conditioning on the
    `StableDiffusionProcessing` *class*, so a dict cond can outlive the run that made
    it; the wrapper is a straight pass-through for every other shape, and re-installs
    itself if another extension's own hook has since replaced it.
    """
    if getattr(condition.compile_conditions, "_negpip", False):
        return

    original = condition.compile_conditions

    @wraps(original)
    def compile_conditions(cond):
        if isinstance(cond, dict) and "crossattn" in cond and "vector" not in cond:
            cross_attn = cond["crossattn"]
            model_conds = {"c_crossattn": condition.ConditionCrossAttn(cross_attn)}

            for key in (NEGPIP_MASK_KEY, NEGPIP_BIAS_KEY):
                if key in cond:
                    model_conds[key] = condition.Condition(cond[key])

            return [dict(cross_attn=cross_attn, model_conds=model_conds)]

        return original(cond)

    compile_conditions._negpip = True

    condition.compile_conditions = compile_conditions
    sampling_function.compile_conditions = compile_conditions


def _force_pytorch_attention(enable: bool):
    """Point `krea.attention_function` at the backend that takes an additive mask.

    Sage and flash attention do not accept an arbitrary float mask, so the logit bias
    needs the plain SDPA path.  `backend/nn/krea.py` binds the name at import time, so
    the module's own reference is the one to rebind — the same reason `compile_conditions`
    has to be patched in two places.

    Only installed while a prompt actually asks for amplification: taking the optimised
    backend away from everyone who just wants a value flip would be a silent slowdown.
    """
    from backend.nn import krea

    if not enable:
        original = getattr(krea, ORIGINAL_ATTENTION, None)
        if original is not None:
            krea.attention_function = original
            delattr(krea, ORIGINAL_ATTENTION)
        return

    if hasattr(krea, ORIGINAL_ATTENTION) or krea.attention_function is attention_pytorch:
        return

    setattr(krea, ORIGINAL_ATTENTION, krea.attention_function)
    krea.attention_function = attention_pytorch


# ================================================================================ #


def patch_dit(dit: Any, *, block_start: int, block_end: int, block_stride: int, patch_txtfusion_refiners: bool, force_pytorch_attention: bool = False) -> int:
    """Hook the DiT forward and the value projection of every selected attention module.

    Returns how many attention modules were hooked.
    """
    _hook_compile_conditions()
    _force_pytorch_attention(force_pytorch_attention)
    _hook_dit(dit, False)

    hooked = 0

    blocks = list(getattr(dit, "blocks", []) or [])
    for i in selected_blocks(len(blocks), block_start, block_end, block_stride):
        attn = getattr(blocks[i], "attn", None)
        if attn is not None:
            _hook_attention(attn, ROLE_BLOCK)
            hooked += 1

    if patch_txtfusion_refiners:
        txtfusion = getattr(dit, "txtfusion", None)
        for block in getattr(txtfusion, "refiner_blocks", []) or []:
            attn = getattr(block, "attn", None)
            if attn is not None:
                _hook_attention(attn, ROLE_REFINER)
                hooked += 1

    return hooked


def unpatch_dit(dit: Any):
    """Undo everything `patch_dit` did, block selection or not.

    Sweeps every candidate module rather than replaying the selection: the sliders may
    have moved since, and a hook left behind on a block nobody is tracking any more
    would keep firing for the rest of the session.
    """
    _force_pytorch_attention(False)

    if dit is None:
        return

    _hook_dit(dit, True)

    for attn in _attention_modules(dit):
        if attn is not None:
            _hook_attention(attn, remove=True)
