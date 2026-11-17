"""
The MiniMax H3 adapter: the third-party `pruna-P-H3-Image-forge-neo` extension's
`MiniMaxH3Engine` and `MiniMaxH3Model`, as MiniMax-H3-Image, P-H3-Image or MiniMax H3, behind
either text encoder it accepts — the 32B Qwen3-VL truncated to 50 layers, or P-H3-Image's
Qwen3-VL 4B with Pruna's text connector, which stands in for it at a sixth of the memory.

Nothing is imported from the extension by path; everything is resolved from the loaded model
and checked before it is touched, and a mismatch stands the extension down with a logged
reason naming what is missing.

Text: the H3 presentation has no chat template.  The engine tokenises the raw prompt with no
special tokens through `SDTokenizer.tokenize_with_weights(..., disable_weights=True)` and
hard-wires `EmphasisNone`, so a weight is read as literal text unless this extension encodes
the line.  The conditioning is every token of the prompt — nothing to strip — `(T, 5120)` per
line.  The 32B encoder hands over its last hidden state; the 4B one hands five hidden states
to the connector, which standardises, projects and rescales each token on its own (so the rows
stay 1:1 with the tokens) and then **replaces position 0 with a learned sink**.  Position 0
is the prompt's first token, since no special token precedes it: its own state never reaches
the DiT, so `pinned_positions` keeps that row neutral and a weight on it is dropped with a
warning.  An unclaimed weight goes to `encode_token_weights` as a token weight, the ComfyUI
output lerp, as on MOSS-Image.

Pictures: an edit, and the extension's Blank reference, put a `<Picture i>` label and a vision
block in front of the text of *both* prompts, and Blank reference also appends a second,
picture-free encode of the same prompt that the DiT picks by sigma.  None of that is handled:
`hand_back` gives such a call to the engine whole, with one warning.

DiT: single-stream over `[text | reference pictures | (audio) | picture]` — text **first** —
with 50 blocks of `Attention`, whose `qkv_proj` fuses q, k and v (`split` in that order, each
`heads * head_dim` wide, which is not the model width), behind a `token_refiner` of two
text-only blocks.  The DiT runs one sample at a time without a batch axis: `forward` loops
over the batch, hands each item to `_forward` with `minimax_h3_item` in `transformer_options`,
and the blocks see `(S, C)` tensors.  So the attention wrapper cuts the item's own row out of
the batched rows, and `dit.py`'s value hook lifts a 2-D output to a batch of one.  The refiner
blocks call their attention without `transformer_options`; `hook_model` wraps `_forward` to
park the item's options on the DiT for them.  Attention ends in `kernels.attention(q, k, v,
heads, transformer_options)` with the heads split, `(1, H, S, D)`, looked up on the module at
call time — the swap point for the logit bias, as on the MOSS-Image extension.

Copyright (C) 2025 hako-mikan (NegPiP)
Copyright (C) 2026 blue-pen5805 (ComfyUI-krea2-negpip)
Copyright (C) 2026 aoleg (Forge Neo port)

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as published
by the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License
along with this program. If not, see <https://www.gnu.org/licenses/>.
"""

from functools import wraps
from typing import Any, Callable

import torch

from backend import memory_management
from backend.attention import attention_pytorch
from backend.text_processing import emphasis
from modules.processing import logger

from lib_krea2_negpip import NEGPIP_BIAS_OPTION_KEY, NEGPIP_OPTION_KEY, ROLE_BLOCK, ROLE_REFINER
from lib_krea2_negpip.adapters import AttentionSite, ModelAdapter
from lib_krea2_negpip.dit import ORIGINAL_ATTENTION, CallSignature, active, attention_bias
from lib_krea2_negpip.moss import LIVE_BIAS_KEY, _module_of, _options_with

#   `type(shared.sd_model).__name__` for the extension's `forge_h3/native/engine.py`
H3_ENGINE = "MiniMaxH3Engine"

#   what `MiniMaxH3Model` puts into `transformer_options` on the way to the blocks
ITEM_KEY = "minimax_h3_item"
LAYOUT_KEY = "minimax_h3_layout"

#   the options of the `_forward` call in flight, parked on the DiT for the refiner blocks
LIVE_OPTIONS_ATTR = "_krea2_negpip_live_options"
ORIGINAL_STEP = "_krea2_negpip_original_step"

#   Pruna's connector on the 4B text encoder (`text_encoder.py`, `TextConnector`)
TEXT_CONNECTOR = "text_connector"

PINNED_REASON = "NegPiP: the P-H3-Image text encoder replaces the first token of a MiniMax H3 prompt with its sink, so a weight on the prompt's first word cannot act; put the weighted word later in the prompt"

HAND_BACK_REASON = "NegPiP does not apply to MiniMax H3 edits or Blank reference, where both prompts are encoded after their pictures; the weights in them are read as literal text there"


def _inner(attn: Any) -> int | None:
    """The width of each of q, k and v: `heads * head_dim`, which on H3 is not the model width."""
    try:
        return int(attn.heads) * int(attn.head_dim)
    except (AttributeError, TypeError, ValueError):
        return None


def _values(attn: Any) -> slice:
    """The value third of `qkv_proj`'s output: `split(heads * head_dim)` hands `v` the last columns."""
    inner = _inner(attn)
    return slice(2 * inner, 3 * inner)


def _attention_problem(attn: Any, where: str) -> str | None:
    if attn is None:
        return f"{where} has no attn"

    qkv = getattr(attn, "qkv_proj", None)
    if qkv is None:
        return f"{where}.attn has no qkv_proj"

    inner = _inner(attn)
    if inner is None:
        return f"{where}.attn has no heads and head_dim"

    width = getattr(qkv, "out_features", None)
    if width is not None and int(width) != 3 * inner:
        return f"{where}.attn.qkv_proj is {width} wide, not 3 x {inner}"

    if "transformer_options" not in CallSignature(attn.forward).index:
        return f"{where}.attn.forward takes no transformer_options"

    return None


def _text_length(layout: Any) -> int | None:
    """The text segment's length in a `PackedLayout`, or `None` without one."""
    for start, stop, kind in getattr(layout, "segments", None) or ():
        if kind == "text":
            return int(stop) - int(start)
    return None


def _restore_method(obj: Any, name: str, original: Any):
    """Put a wrapped method back, instance dict included (as `dit._restore_forward` does for `forward`)."""
    if getattr(original, "__func__", None) is getattr(type(obj), name, None):
        vars(obj).pop(name, None)
    else:
        setattr(obj, name, original)


class H3Adapter(ModelAdapter):
    """The `pruna-P-H3-Image-forge-neo` extension's `MiniMaxH3Engine` and `MiniMaxH3Model`."""

    name = "MiniMax H3"
    text_first = True
    split_encode = False  # the engine never had Forge's one-template-per-segment encode
    template = "{}"  # no chat template: the conditioning is the prompt's own tokens
    tokenizer_kwargs = {"add_special_tokens": False}  # as `MiniMaxH3TextEngine.tokenize`
    pinned_reason = PINNED_REASON

    _warned_length = False

    @classmethod
    def matches(cls, model: Any) -> bool:
        return hasattr(model, "text_processing_engine_h3") and (type(model).__name__ == H3_ENGINE or bool(getattr(model, "is_h3", False)))

    def __init__(self, model: Any):
        super().__init__(model)
        self._dit = None

    # ---------------------------------------------------------------- text side -- #

    @property
    def engine(self) -> Any:
        return self.model.text_processing_engine_h3

    @property
    def has_connector(self) -> bool:
        """Whether the text encoder is P-H3-Image's 4B one, with Pruna's connector and its sink."""
        encoder = getattr(getattr(self.engine, "text_encoder", None), "transformer", None)
        return hasattr(encoder, TEXT_CONNECTOR)

    def engine_problem(self) -> str | None:
        engine = getattr(self.model, "text_processing_engine_h3", None)
        if engine is None:
            return "the engine has no text_processing_engine_h3"
        if not callable(getattr(engine, "tokens", None)):
            return "the text engine has no tokens"
        if not hasattr(engine, "vision_spans"):
            return "the text engine has no vision_spans"

        encoder = getattr(engine, "text_encoder", None)
        if not callable(getattr(encoder, "encode_token_weights", None)):
            return "the text encoder has no encode_token_weights"
        if getattr(encoder, "transformer", None) is None:
            return "the text encoder wrapper has no transformer"

        tokenizer = getattr(engine, "tokenizer", None)
        if not callable(getattr(tokenizer, "tokenize_with_weights", None)):
            return "the tokenizer wrapper has no tokenize_with_weights"
        if getattr(tokenizer, "tokenizer", None) is None:
            return "the tokenizer wrapper has no tokenizer"

        return None

    def _pictures(self) -> list:
        images = getattr(self.model, "condition_images", None)
        if callable(images):
            return list(images())
        return list(getattr(self.model, "references", None) or ())

    def hand_back(self, prompt) -> str | None:
        """Whether this call's prompts are encoded after pictures: an edit, or Blank reference.

        Both prompts carry the pictures, so the negative prompt is handed back too.  The
        engine's own encode also records where the vision blocks landed (`vision_spans`),
        which `prepare()` hands the DiT for the modality tags; handing the whole call back
        keeps that bookkeeping the engine's.
        """
        return HAND_BACK_REASON if self._pictures() else None

    def before_encode(self, prompt):
        """What `get_learned_conditioning` and `__call__` do before a plain encode.

        A pending error from the extension's own settings check is raised as the engine
        would raise it, and the vision spans of the last encode are dropped: `prepare()`
        reads them for the next generation, and a stale set from an earlier edit would give
        prompt tokens the picture's modality tag.
        """
        pending = getattr(_module_of(self.model), "raise_pending_error", None)
        if callable(pending):
            pending()

        self.engine.vision_spans = []

    def load_text_encoder(self):
        """As the engine loads it: with the working memory of one prompt pass kept free."""
        prompt_memory = getattr(_module_of(self.model), "prompt_memory", None)
        if not callable(prompt_memory):
            super().load_text_encoder()
            return

        memory_management.load_models_gpu([self.model.forge_objects.clip.patcher], memory_required=prompt_memory([]))

    def plain_ids(self, line: str) -> list[int] | None:
        """The engine's own token ids for a line with no pictures: `tokens` does not strip, and
        an empty line is one pad token."""
        return [int(token) for token, _ in self.engine.tokens(line)[0]]

    def plain_encode(self, line: str) -> torch.Tensor:
        """The engine's own conditioning for one line, `(T, 5120)`, as `__call__` computes it."""
        return self.engine.text_encoder.encode_token_weights(self.engine.tokens(line))[0][0]

    def pinned_positions(self) -> int:
        return 1 if self.has_connector else 0

    def encode(self, line: str, tokens: list, multipliers: list[float], weighting: emphasis.Emphasis) -> torch.Tensor:
        """`(T, 5120)`, the engine's own encode for the engine's own stream.

        As `MossAdapter.encode`: an unclaimed weight is ComfyUI's output lerp through
        `encode_token_weights`, *Original* and *No norm* both meaning it and *None* and
        *Ignore* meaning no weight.  On the 4B encoder the lerp acts on the connector's
        output, whose empty encoding went through the connector too.
        """
        if weighting.name in ("None", "Ignore") or all(m == 1.0 for m in multipliers):
            if list(tokens) == list(self.plain_ids(line) or []):
                return self.plain_encode(line)
            pairs = [(token, 1.0) for token in tokens]
        else:
            pairs = list(zip(tokens, multipliers))

        return self.engine.text_encoder.encode_token_weights([pairs])[0][0]

    # ----------------------------------------------------------------- DiT side -- #

    def dit_problem(self, dit: Any) -> str | None:
        blocks = getattr(dit, "blocks", None)
        if not blocks:
            return "the DiT has no blocks"
        for i, block in enumerate(blocks):
            problem = _attention_problem(getattr(block, "attn", None), f"blocks[{i}]")
            if problem is not None:
                return problem

        refiners = getattr(getattr(dit, "token_refiner", None), "blocks", None)
        if refiners is None:
            return "the DiT has no token_refiner.blocks"
        for i, block in enumerate(refiners):
            problem = _attention_problem(getattr(block, "attn", None), f"token_refiner.blocks[{i}]")
            if problem is not None:
                return problem

        step = getattr(dit, "_forward", None)
        if not callable(step):
            return "the DiT has no _forward"
        if "transformer_options" not in CallSignature(step).index:
            return "the DiT's _forward takes no transformer_options"

        kernels = getattr(_module_of(dit), "kernels", None)
        if not callable(getattr(kernels, "attention", None)):
            return "the transformer module has no kernels.attention"

        return None

    def block_sites(self, dit: Any) -> list[AttentionSite]:
        return [AttentionSite(module=block.attn, value=block.attn.qkv_proj, value_slice=_values(block.attn), role=ROLE_BLOCK) for block in getattr(dit, "blocks", None) or []]

    def refiner_sites(self, dit: Any) -> list[AttentionSite]:
        self._dit = dit
        blocks = getattr(getattr(dit, "token_refiner", None), "blocks", None) or []
        return [AttentionSite(module=block.attn, value=block.attn.qkv_proj, value_slice=_values(block.attn), role=ROLE_REFINER) for block in blocks]

    def _item_rows(self, options: Any) -> dict:
        """This item's rows out of the batched ones, keyed as `active` reads them.

        The rows arrive `(B, T, 1)` for the whole batch, the DiT runs one item at a time, and
        `_align_batch`'s rule for a row batch that divides the call's is repetition, so item
        `i` reads row `i % B`.  A row whose length is not the text segment's describes some
        other sequence (a picture prefix, a Blank reference's second encode) and is not
        applied — the prompts that make one are handed back, so this is a guard, said once.
        """
        if not isinstance(options, dict):
            return {}

        item = options.get(ITEM_KEY)
        length = _text_length(options.get(LAYOUT_KEY))
        rows = {}

        for key in (NEGPIP_OPTION_KEY, NEGPIP_BIAS_OPTION_KEY):
            row = options.get(key)
            if not torch.is_tensor(row) or row.ndim != 3:
                continue

            if item is not None and row.shape[0] > 1:
                index = int(item) % int(row.shape[0])
                row = row[index : index + 1]

            if length is not None and int(row.shape[1]) != length:
                if not H3Adapter._warned_length:
                    H3Adapter._warned_length = True
                    logger.warning(f"NegPiP: the MiniMax H3 prompt is {length} tokens in the DiT but its weights cover {int(row.shape[1])}; leaving it alone")
                continue

            rows[key] = row

        return rows

    def wrap_attention(self, site: AttentionSite, original: Callable) -> Callable:
        """`Attention.forward` with this item's rows parked and the bias put where the swap reads it.

        A main block's `transformer_options` carry the rows and the item index; a refiner
        block's attention is called with none, and reads the options of the `_forward` call
        in flight off the DiT instead (`hook_model`).  The bias goes into a copy of the
        options under `LIVE_BIAS_KEY`, which `kernels.attention` passes on to the swap.
        """
        sig = CallSignature(original)
        refiner = site.role == ROLE_REFINER

        @wraps(original)
        def negpip_forward(*args, **kwargs):
            given = sig.get(args, kwargs, "transformer_options", None)
            options = getattr(self._dit, LIVE_OPTIONS_ATTR, None) if refiner else given

            with active(site, self._item_rows(options)) as bias:
                if bias is not None:
                    args, kwargs = sig.replace(args, kwargs, "transformer_options", _options_with(given if isinstance(given, dict) else {}, bias))
                return original(*args, **kwargs)

        return negpip_forward

    def hook_model(self, dit: Any, enable: bool):
        """Wrap `_forward` so the refiner blocks can find the options of the item in flight.

        `_forward` is called per item with that item's options, mutates them with the
        packed layout before the text refiner runs, and passes nothing to the refiner; the
        wrapper parks the very dict on the DiT for the length of the call.
        """
        if not enable:
            original = getattr(dit, ORIGINAL_STEP, None)
            if original is not None:
                if getattr(dit._forward, "_negpip", False):
                    _restore_method(dit, "_forward", original)
                delattr(dit, ORIGINAL_STEP)
            if hasattr(dit, LIVE_OPTIONS_ATTR):
                delattr(dit, LIVE_OPTIONS_ATTR)
            return

        self._dit = dit
        if getattr(dit, ORIGINAL_STEP, None) is not None:
            return

        original = dit._forward
        sig = CallSignature(original)

        @wraps(original)
        def negpip_step(*args, **kwargs):
            options = sig.get(args, kwargs, "transformer_options", None)
            setattr(dit, LIVE_OPTIONS_ATTR, options if isinstance(options, dict) else None)
            try:
                return original(*args, **kwargs)
            finally:
                setattr(dit, LIVE_OPTIONS_ATTR, None)

        negpip_step._negpip = True

        setattr(dit, ORIGINAL_STEP, original)
        dit._forward = negpip_step

    def force_pytorch_attention(self, dit: Any, enable: bool):
        """Swap the extension's `kernels.attention` for one that takes the bias as a mask.

        `Attention.forward` calls `kernels.attention(q, k, v, heads, transformer_options)`
        with the heads split, `(1, H, S, D)`, looked up on the module at call time, so
        rebinding the attribute reaches every block and refiner.  The swap pads the bias to
        the key axis with the text first and runs `attention_pytorch`, the backend that
        accepts an additive float mask; a call that carries no bias goes to the original
        untouched, Forge's attention overrides included.
        """
        kernels = getattr(_module_of(dit), "kernels", None)
        if kernels is None:
            return

        if not enable:
            original = getattr(kernels, ORIGINAL_ATTENTION, None)
            if original is not None:
                kernels.attention = original
                delattr(kernels, ORIGINAL_ATTENTION)
            return

        if hasattr(kernels, ORIGINAL_ATTENTION):
            return

        original = kernels.attention
        text_first = self.text_first

        @wraps(original)
        def negpip_attention(q, k, v, heads, transformer_options=None, **kwargs):
            options = transformer_options if isinstance(transformer_options, dict) else {}
            bias = options.get(LIVE_BIAS_KEY, None)

            mask = None
            if bias is not None and torch.is_tensor(q) and q.ndim == 4:
                mask = attention_bias(int(q.shape[0]), int(k.shape[2]), bias, text_first, like=q, within=int(q.shape[2]))

            if mask is None:
                return original(q, k, v, heads, transformer_options=transformer_options, **kwargs)

            return attention_pytorch(q, k, v, heads, mask=mask, skip_reshape=True, transformer_options=options)

        negpip_attention._negpip = True

        setattr(kernels, ORIGINAL_ATTENTION, original)
        kernels.attention = negpip_attention


__all__ = ["H3_ENGINE", "HAND_BACK_REASON", "ITEM_KEY", "LAYOUT_KEY", "PINNED_REASON", "H3Adapter"]
