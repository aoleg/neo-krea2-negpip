"""
Diffusion-model side of NegPiP.

Two levers, both applied inside attention, after every non-linear conditioning stage has
already run:

* the **value factor** scales the flagged tokens' value vectors — attention still scores
  them normally, then subtracts (or damps, or amplifies) what they contribute instead of
  adding it as-is.  This is NegPiP proper.  It scales about the prompt's own *mean* value
  vector rather than about the origin, which leaves the common component of the text
  values alone; see `_apply_value_flip` for why scaling about the origin softens the
  whole image.  A token's share of the attention output is *linear* in this factor, so it
  works above `1.0` as well and never saturates, and it needs no mask, so it runs on every
  attention backend.
* the **logit bias** is added to the flagged tokens' attention scores, multiplying their
  softmax weight by `exp(bias)`.  It changes how much the sequence *attends* to the token
  rather than what the token contributes, which is a different kind of emphasis: the other
  keys lose the share the flagged one gains.  It saturates as that share approaches one,
  and it needs an attention backend that accepts an additive mask.

Both models this extension handles are single-stream: the text tokens and the image
patches share one self-attention sequence, so there is no cross-attention module to hook
and the flip goes on the value projection of the blocks themselves, over the text
positions of the joint sequence — the leading `txtlen` on Krea 2, the trailing `txtlen`
on MOSS-Image.  Each model's text refiner blocks are the same kind of attention one stage
earlier, over the text alone; flipping there as well is optional and stronger.

Two levels of hook, so none of either model's attention math is duplicated here:

* the module nearest the attention that receives `transformer_options` gets its `forward`
  wrapped: it parks the value factors on itself for the duration of one call and gets the
  logit bias into the attention the way its model allows (`ModelAdapter.wrap_attention`);
* the `nn.Linear` that produces the values gets a wrapper that scales the rows the parked
  factors name.

That linear returns `(B, L, heads * headdim)` before the rearrange into heads, so scaling a
whole row scales that token in every head at once; where q, k and v come out of one fused
projection, the adapter says which columns are the values.

Where the text sits, which modules those are, and how a mask reaches the attention are
the adapter's business (`adapters.py`); this module owns the levers and the hooking.
"""

import inspect
from contextlib import contextmanager
from functools import wraps
from typing import Any

import torch

from backend.sampling import condition, sampling_function

from lib_krea2_negpip import (
    NEGPIP_BIAS_KEY,
    NEGPIP_BIAS_OPTION_KEY,
    NEGPIP_MASK_KEY,
    NEGPIP_OPTION_KEY,
    NEGPIP_ROLE_ATTR,
    ROLE_BLOCK,
    ROLE_REFINER,
)
from lib_krea2_negpip.adapters import AttentionSite, ModelAdapter

ACTIVE_ATTR = "_krea2_negpip_active_mask"
REFERENCE_ATTR = "_krea2_negpip_mean_reference"
ORIGINAL_FORWARD = "_krea2_negpip_original_forward"
ORIGINAL_VALUE = "_krea2_negpip_original_value_forward"
ORIGINAL_ATTENTION = "_krea2_negpip_original_attention_function"


class CallSignature:
    """Where each parameter of `original` sits, so a wrapper can read or replace one by name.

    A wrapper that spells the wrapped method's parameters out breaks the day the framework
    adds one: Forge's `c6c2b43d` gave the MOSS-Image block three more positional
    parameters for reference images, and a wrapper written as `(x, mod, pe,
    transformer_options={})` raised on the fourth positional.  This reads the signature
    once and lets the wrapper take `(*args, **kwargs)`, pick the parameter it needs out of
    wherever the caller put it, substitute its replacement in that same place, and pass
    everything else through untouched.  Looking a parameter up by name also means a
    parameter interposed before it cannot silently take its value, which the earlier
    keyword-past-the-positionals rule guarded against.
    """

    def __init__(self, original: Any):
        try:
            params = [p for p in inspect.signature(original).parameters.values() if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)]
        except (TypeError, ValueError):
            params = []

        self.index = {p.name: i for i, p in enumerate(params)}
        self.defaults = {p.name: p.default for p in params if p.default is not p.empty}

    def get(self, args: tuple, kwargs: dict, name: str, default: Any = None) -> Any:
        """The value the call passes for `name`, or its default."""
        i = self.index.get(name)
        if i is not None and i < len(args):
            return args[i]
        if name in kwargs:
            return kwargs[name]
        return self.defaults.get(name, default)

    def replace(self, args: tuple, kwargs: dict, name: str, value: Any) -> tuple[tuple, dict]:
        """`(args, kwargs)` with `name` set to `value`, in the place the caller passed it."""
        i = self.index.get(name)
        if i is not None and i < len(args):
            return (*args[:i], value, *args[i + 1 :]), kwargs
        return args, {**kwargs, name: value}


def selected_blocks(count: int, start: int, end: int, stride: int) -> list[int]:
    if count <= 0:
        return []

    if end < start:
        start, end = end, start

    start = max(0, min(int(start), count - 1))
    end = max(0, min(int(end), count - 1))
    stride = max(1, int(stride))

    return [i for i in range(start, end + 1) if (i - start) % stride == 0]


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


def _text_span(length: int, total: int, text_first: bool) -> slice:
    """The `length` text positions of a joint sequence `total` long."""
    return slice(0, length) if text_first else slice(total - length, total)


def _mean_value(text: torch.Tensor, m: torch.Tensor, amount: float) -> torch.Tensor:
    """`amount` x the mean value vector of the text tokens this row leaves alone.

    Averaged over the *unflagged* tokens, so a flagged one cannot drag its own reference
    towards itself.  A prompt where every token is flagged has no such mean, and falls
    back to the whole row rather than dividing by zero.
    """
    keep = (m == 1.0).to(torch.float32)
    denom = keep.sum(dim=1, keepdim=True)

    #   decided on the device: reading the flag back to the host (`bool(empty.any())`)
    #   would synchronise the stream in every hooked block on every step, and the GPU then
    #   idles behind the Python dispatch loop, which showed up as sampling 1.4x slower
    empty = denom == 0
    keep = torch.where(empty, torch.ones_like(keep), keep)
    denom = torch.where(empty, torch.full_like(denom, float(keep.shape[1])), denom)

    mean = (text.to(torch.float32) * keep).sum(dim=1, keepdim=True) / denom
    return (mean * amount).to(text.dtype)


def _apply_value_flip(v: torch.Tensor, mask: torch.Tensor, reference: float = 0.0, text_first: bool = True) -> torch.Tensor:
    """Scale the flagged tokens' value vectors, about `reference` x the prompt's mean.

    `reference = 0.0` is NegPiP as written — `v -> m*v`, a reflection about the origin.
    That reflection is not sign-symmetric in a way the model is: attention output is a
    *convex* combination of value vectors, and the text tokens' values share a large
    common component, so `m = -1` subtracts `2*a_t` of that common component from every
    query in the sequence — the image patches and, this being a single-stream DiT, the
    text tokens as well, at every block.  A shared offset on every token is exactly what
    the next block's `RMSNorm` divides out at the expense of the token-to-token variation
    around it, which is spatial detail.

    `reference = 1.0` reflects about the prompt's own mean value vector instead:
    `v -> vbar + m*(v - vbar)`.  Writing `v = vbar + d`, the perturbation is then
    `(m-1)*d` rather than `(m-1)*(vbar + d)` — the shared component of the change is
    exactly zero, and only the part that says *which word this is* gets inverted.  The
    suppression is unchanged; what stops happening is the broadband softening.

    Written in place, so a view into a fused q/k/v output is scaled where it lies.
    """
    if not torch.is_tensor(v) or v.ndim != 3:
        return v

    length = min(int(mask.shape[1]), int(v.shape[1]))
    if length <= 0:
        return v

    m = _align_batch(mask[:, :length, :].to(device=v.device, dtype=v.dtype), v.shape[0])
    if m is None:
        return v

    span = _text_span(length, int(v.shape[1]), text_first)

    if reference > 0.0:
        #   v' = r + m*(v - r) = m*v + r*(1 - m); `r` has to be read before the write
        r = _mean_value(v[:, span, :], m, reference)
        v[:, span, :] = v[:, span, :] * m + r * (1.0 - m)
    else:
        v[:, span, :] = v[:, span, :] * m

    return v


def attention_bias(batch: int, keys: int, bias: torch.Tensor, text_first: bool, like: torch.Tensor, within: int | None = None) -> torch.Tensor | None:
    """`(B, 1, keys)` additive logit bias over a module's whole key axis, or `None`.

    The row only covers the text tokens; the main blocks attend over the joint sequence,
    so it is zero-padded out to the full key axis, at whichever end the text sits.
    `(B, 1, keys)` is what `attention_pytorch` unsqueezes to `(B, 1, 1, keys)` — one bias
    per batch row, which is what keeps cond and uncond from reading each other's row.
    `like` only lends its device and dtype.

    `within` is the length of the joint sequence the text sits in when the key axis is
    longer than it: MOSS-Image Edit appends the reference images' keys and values after
    the sequence's own inside attention, and the text is the tail of the *queries*, not of
    the keys.  Left out, the sequence is the whole key axis.
    """
    if not torch.is_tensor(bias) or bias.ndim != 3:
        return None

    total = keys if within is None else min(int(within), keys)
    length = min(int(bias.shape[1]), total)
    if length <= 0:
        return None

    row = _align_batch(bias[:, :length, :].to(device=like.device, dtype=like.dtype), batch)
    if row is None:
        return None

    out = like.new_zeros((batch, 1, keys))
    out[:, 0, _text_span(length, total, text_first)] = row[:, :, 0]
    return out


@contextmanager
def active(site: AttentionSite, transformer_options: Any):
    """Park this call's value factors on the site's module; yield its logit bias, if any.

    The value wrapper below reads the parked factors; the adapter's `forward` wrapper
    takes the yielded bias and gets it into the attention.  Cleared on the way out, so a
    module never carries a row into a call that did not bring one.
    """
    options = transformer_options if isinstance(transformer_options, dict) else {}

    setattr(site.module, ACTIVE_ATTR, options.get(NEGPIP_OPTION_KEY, None))
    try:
        yield options.get(NEGPIP_BIAS_OPTION_KEY, None)
    finally:
        setattr(site.module, ACTIVE_ATTR, None)


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


def _hook_value(adapter: ModelAdapter, site: AttentionSite, remove: bool):
    value = site.value
    if value is None:
        return

    if remove:
        original = getattr(value, ORIGINAL_VALUE, None)
        if original is not None:
            if getattr(value.forward, "_negpip", False):
                _restore_forward(value, original)
            delattr(value, ORIGINAL_VALUE)
        return

    if getattr(value, ORIGINAL_VALUE, None) is not None:
        return

    original = value.forward
    module, columns, text_first = site.module, site.value_slice, adapter.text_first

    @wraps(original)
    def negpip_forward(*args, **kwargs):
        out = original(*args, **kwargs)
        mask = getattr(module, ACTIVE_ATTR, None)
        if mask is None or not torch.is_tensor(out):
            return out

        reference = getattr(module, REFERENCE_ATTR, 0.0)
        if columns is None:
            return _apply_value_flip(out, mask, reference, text_first)

        #   a view: the values are scaled where they lie, and q and k are not touched
        _apply_value_flip(out[..., columns], mask, reference, text_first)
        return out

    negpip_forward._negpip = True

    setattr(value, ORIGINAL_VALUE, original)
    value.forward = negpip_forward


def _hook_attention(adapter: ModelAdapter, site: AttentionSite, mean_reference: float = 0.0, remove: bool = False):
    module = site.module

    if remove:
        original = getattr(module, ORIGINAL_FORWARD, None)
        if original is not None:
            if getattr(module.forward, "_negpip", False):
                _restore_forward(module, original)
            delattr(module, ORIGINAL_FORWARD)

        _hook_value(adapter, site, True)

        for name in (NEGPIP_ROLE_ATTR, ACTIVE_ATTR, REFERENCE_ATTR):
            if hasattr(module, name):
                delattr(module, name)
        return

    #   before the already-hooked check: the slider may have moved without anything else
    #   about the patch changing, and a stale reference is a silently wrong image
    setattr(module, REFERENCE_ATTR, float(mean_reference))

    if getattr(module, ORIGINAL_FORWARD, None) is not None:
        return

    _hook_value(adapter, site, False)
    original = module.forward

    negpip_forward = adapter.wrap_attention(site, original)
    negpip_forward._negpip = True

    setattr(module, NEGPIP_ROLE_ATTR, site.role)
    setattr(module, ORIGINAL_FORWARD, original)
    module.forward = negpip_forward


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
    def negpip_forward(x, timesteps, context, *args, transformer_options=None, **kwargs):
        mask = kwargs.pop(NEGPIP_MASK_KEY, None)
        bias = kwargs.pop(NEGPIP_BIAS_KEY, None)
        options = dict(transformer_options or {})

        #   a row indexed 1:1 with the conditioning's token axis arrives batched like it:
        #   (batch, seq, 1, 1) behind Krea 2's (batch, seq, taps, dim), (batch, seq, 1)
        #   behind MOSS-Image's (batch, seq, dim)
        if torch.is_tensor(mask):
            options[NEGPIP_OPTION_KEY] = mask.reshape(mask.shape[0], -1, 1)
        if torch.is_tensor(bias):
            options[NEGPIP_BIAS_OPTION_KEY] = bias.reshape(bias.shape[0], -1, 1)

        #   by keyword past the required positionals: a parameter interposed before
        #   `transformer_options` would otherwise silently take our options dict and
        #   leave the real one at its default, which is a working forward pass that
        #   quietly does nothing.  The ComfyUI node hit exactly this when Krea 2 Edit
        #   added `ref_latents` to the signature there.  Forge itself calls the model with
        #   `context`, `control` and `transformer_options` by keyword (`k_model.py`).
        return original(x, timesteps, context, *args, transformer_options=options, **kwargs)

    negpip_forward._negpip = True

    setattr(dit, ORIGINAL_FORWARD, original)
    dit.forward = negpip_forward


def _hook_compile_conditions():
    """Teach `compile_conditions` the conditioning shape the text hook produces.

    It knows two: a bare tensor, or a dict carrying both `crossattn` and a pooled
    `vector`.  NegPiP makes a third — `crossattn` plus the per-token rows, no pooled
    vector — and the stock function would raise `KeyError: 'vector'` on it.

    The rows ride as `ConditionCrossAttn`, the class the conditioning itself rides as, and
    for the same reason: when cond and uncond have different token counts Forge batches
    them into one forward pass by repeating the shorter along the token axis up to the
    lcm (`ConditionCrossAttn.concat`, when the multiple is at most 4), and a plain
    `Condition` refuses to concat on any shape difference, which made every such batch
    run cond and uncond as two passes while NegPiP was on.  Tiling the rows with the same
    rule keeps them 1:1 with the tiled conditioning: position `i` of the repeated row is
    the position `i` of the repeated tokens, by construction.

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
                    model_conds[key] = condition.ConditionCrossAttn(cond[key])

            return [dict(cross_attn=cross_attn, model_conds=model_conds)]

        return original(cond)

    compile_conditions._negpip = True

    condition.compile_conditions = compile_conditions
    sampling_function.compile_conditions = compile_conditions


# ================================================================================ #


def patch_dit(adapter: ModelAdapter, dit: Any, *, block_start: int, block_end: int, block_stride: int, patch_refiners: bool, mean_reference: float = 1.0, force_pytorch_attention: bool = False) -> int:
    """Hook the DiT forward and the value projection of every selected attention site.

    Returns how many sites were hooked.
    """
    _hook_compile_conditions()
    adapter.force_pytorch_attention(dit, force_pytorch_attention)
    _hook_dit(dit, False)

    hooked = 0

    blocks = adapter.block_sites(dit)
    for i in selected_blocks(len(blocks), block_start, block_end, block_stride):
        _hook_attention(adapter, blocks[i], mean_reference)
        hooked += 1

    if patch_refiners:
        for site in adapter.refiner_sites(dit):
            _hook_attention(adapter, site, mean_reference)
            hooked += 1

    return hooked


def unpatch_dit(adapter: ModelAdapter, dit: Any):
    """Undo everything `patch_dit` did, block selection or not.

    Sweeps every candidate site rather than replaying the selection: the sliders may
    have moved since, and a hook left behind on a block nobody is tracking any more
    would keep firing for the rest of the session.
    """
    if dit is None:
        return

    adapter.force_pytorch_attention(dit, False)
    _hook_dit(dit, True)

    for site in [*adapter.block_sites(dit), *adapter.refiner_sites(dit)]:
        _hook_attention(adapter, site, remove=True)


__all__ = ["ROLE_BLOCK", "ROLE_REFINER", "CallSignature", "active", "attention_bias", "patch_dit", "selected_blocks", "unpatch_dit"]
