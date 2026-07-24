"""
Diffusion-model side of Krea 2 NegPiP.

NegPiP negates the *value* vectors of the flagged tokens: attention still scores them
normally, then subtracts what they contribute instead of adding it.  Negating the keys
instead would only change how much weight the softmax gives them, which is not the same
thing and cannot go below zero.

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

* the selected `Attention.forward` parks the mask on the module for the duration of one
  call — it is the only place that sees `transformer_options`;
* that module's `wv` (`nn.Linear`) gets a wrapper that scales the rows it names.

`wv` returns `(B, L, kvheads * headdim)` before the rearrange into heads, so scaling a
whole row scales that token in every kv head at once.
"""

from functools import wraps
from typing import Any

import torch

from backend.sampling import condition, sampling_function

from lib_krea2_negpip import (
    KREA2_TAP_DIM,
    KREA2_TAP_LAYERS,
    NEGPIP_MASK_KEY,
    NEGPIP_OPTION_KEY,
    NEGPIP_ROLE_ATTR,
    ROLE_BLOCK,
    ROLE_REFINER,
)

ACTIVE_ATTR = "_krea2_negpip_active_mask"
ORIGINAL_FORWARD = "_krea2_negpip_original_forward"
ORIGINAL_WV = "_krea2_negpip_original_wv_forward"


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


def _apply_value_flip(v: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if not torch.is_tensor(v) or v.ndim != 3:
        return v

    length = min(int(mask.shape[1]), int(v.shape[1]))
    if length <= 0:
        return v

    m = mask[:, :length, :].to(device=v.device, dtype=v.dtype)

    if m.shape[0] != v.shape[0]:
        #   cond and uncond are concatenated along the batch axis whenever their token
        #   counts match, and Forge may run several latents per cond
        if m.shape[0] == 0 or v.shape[0] % m.shape[0] != 0:
            return v
        m = m.repeat(v.shape[0] // m.shape[0], 1, 1)

    v[:, :length, :] = v[:, :length, :] * m
    return v


def _hook_wv(attn: Any, remove: bool):
    wv = getattr(attn, "wv", None)
    if wv is None:
        return

    if remove:
        original = getattr(wv, ORIGINAL_WV, None)
        if original is not None:
            if getattr(wv.forward, "_negpip", False):
                wv.forward = original
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
                attn.forward = original
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
        negpip_mask = transformer_options.get(NEGPIP_OPTION_KEY, None) if isinstance(transformer_options, dict) else None

        setattr(attn, ACTIVE_ATTR, negpip_mask)
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
                dit.forward = original
            delattr(dit, ORIGINAL_FORWARD)
        return

    if getattr(dit, ORIGINAL_FORWARD, None) is not None:
        return

    original = dit.forward

    @wraps(original)
    def negpip_forward(x, timesteps, context, attention_mask=None, transformer_options=None, **kwargs):
        mask = kwargs.pop(NEGPIP_MASK_KEY, None)
        options = dict(transformer_options or {})

        if torch.is_tensor(mask):
            #   conditioning arrives as (batch, 1, seq, features) — Krea 2 squeezes the
            #   singleton itself — so the mask arrives as (batch, 1, seq, 1)
            options[NEGPIP_OPTION_KEY] = mask.reshape(mask.shape[0], -1, 1)

        return original(x, timesteps, context, attention_mask, options, **kwargs)

    negpip_forward._negpip = True

    setattr(dit, ORIGINAL_FORWARD, original)
    dit.forward = negpip_forward


def _hook_compile_conditions():
    """Teach `compile_conditions` the conditioning shape the text hook produces.

    It knows two: a bare tensor, or a dict carrying both `crossattn` and a pooled
    `vector`.  Krea 2 NegPiP makes a third — `crossattn` plus the sign mask, no pooled
    vector — and the stock function would raise `KeyError: 'vector'` on it.

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

            if NEGPIP_MASK_KEY in cond:
                model_conds[NEGPIP_MASK_KEY] = condition.Condition(cond[NEGPIP_MASK_KEY])

            return [dict(cross_attn=cross_attn, model_conds=model_conds)]

        return original(cond)

    compile_conditions._negpip = True

    condition.compile_conditions = compile_conditions
    sampling_function.compile_conditions = compile_conditions


# ================================================================================ #


def patch_dit(dit: Any, *, block_start: int, block_end: int, block_stride: int, patch_txtfusion_refiners: bool) -> int:
    """Hook the DiT forward and the value projection of every selected attention module.

    Returns how many attention modules were hooked.
    """
    _hook_compile_conditions()
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
    if dit is None:
        return

    _hook_dit(dit, True)

    for attn in _attention_modules(dit):
        if attn is not None:
            _hook_attention(attn, remove=True)
