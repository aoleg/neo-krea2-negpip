"""
The MOSS-Image adapters: Forge Neo's own `MOSSImage` engine (`be6fa5d4`, 2026-10-21) and
the third-party `moss-image-forge-neo` extension's `MossImageEngine` (0.1.x) that preceded
it.  One adapter per backend, both on a shared base, since the model underneath is the
same and so is everything the hooks need: Qwen3.5-9B through Forge's `SDClipModel` with
`layer="last"`, a chat template that is **not** stripped from the conditioning, and 42
joint `[image, text]` blocks — text **last** — behind two text-only refiners.  What differs
is plumbing: where the engine hangs off the diffusion engine, the block signatures, and
which attention function the blocks call.

Nothing is imported from the extension by path; everything is resolved from the loaded
model and checked before it is touched, and a mismatch stands the extension down with a
logged reason naming what is missing.  After flyfront/sd-forge-negpip's `neo-moss` branch
and its later native support, which mapped both backends out first; the hooks are this
extension's own.

Text: the conditioning is the last hidden state, `(T, 4096)` per line with no batch axis,
for the *whole* templated prompt — the DiT reads every token, so the rows cover the whole
sequence and the template tokens carry a neutral `1.0` / `0.0`.  Both engines hard-wire
`EmphasisNone` and pass the prompt verbatim, brackets included; an unclaimed weight gets
the ComfyUI output lerp that `encode_token_weights` applies, which is what a ComfyUI user
of the model gets.  A layout prompt — one JSON object at the front of the prompt, possibly
brace-stripped by sd-dynamic-prompts on the extension, possibly with Forge's JSON Builder
*Extra* text after it — never goes near the prompt parser, whose square brackets would
read a `bbox` as de-emphasis; whatever follows the object is a prompt like any other.

DiT: the value projection is inside `block.proj`: a fused `proj["qkv"]` on Comfy-Org's
files, where the values are the last third of the output columns, or a separate
`proj["v"]` on upstream's diffusers export; the refiners' `qkv` is always fused.  Attention
is a module-level function with no mask argument, so the logit bias is applied by
rebinding the callable it ends in while a bias is in play — it receives
`transformer_options` and lacks only the mask.  Nothing of either block is re-implemented.

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

import json
import sys
from functools import wraps
from typing import Any, Callable

import torch

from backend.args import dynamic_args
from backend.attention import attention_pytorch
from backend.text_processing import emphasis
from modules.shared import opts

from lib_krea2_negpip import ROLE_BLOCK, ROLE_REFINER
from lib_krea2_negpip.adapters import AttentionSite, ModelAdapter
from lib_krea2_negpip.dit import ORIGINAL_ATTENTION, CallSignature, active, attention_bias

#   `type(shared.sd_model).__name__` for the extension's `forge_moss/native/engine.py`
MOSS_EXTENSION_ENGINE = "MossImageEngine"

#   `type(shared.sd_model).__name__` for Forge's `backend/diffusion_engine/moss_image.py`
MOSS_NATIVE_ENGINE = "MOSSImage"

#   the identifying shape of the MOSS-Image DiT: 42 joint blocks of width 3072
MOSS_LAYERS = 42
MOSS_DIM = 3072

#   the attribute of the extension's text-engine module that rebuilds a brace-stripped layout
RESTORE_LAYOUT = "restore_layout_json"

#   a line that already is chat markup; Forge's engine templates nothing around it
IM_START = "<|im_start|>"

#   the key a hooked block puts the bias under for the attention swap to find; distinct
#   from the DiT-level key so an unselected block, which sees that one too, stays untouched
LIVE_BIAS_KEY = "krea2_negpip_live_bias"


def _module_of(obj: Any):
    return sys.modules.get(type(obj).__module__)


def _dit_dim(dit: Any) -> int | None:
    """The model width: the extension's DiT keeps it as `dim`, Forge's does not keep it at all."""
    dim = getattr(dit, "dim", None)
    if dim is None:
        dim = getattr(getattr(dit, "video_embedder", None), "out_features", None)
    try:
        return int(dim)
    except (TypeError, ValueError):
        return None


def _values(dim: int) -> slice:
    """The value third of a fused q/k/v output: `chunk(3, -1)` hands `v` the last columns."""
    return slice(2 * dim, 3 * dim)


def _options_with(transformer_options, bias):
    """A copy of `transformer_options` carrying the bias where the attention swap reads it."""
    if bias is None:
        return transformer_options
    options = dict(transformer_options or {})
    options[LIVE_BIAS_KEY] = bias
    return options


class MossAdapter(ModelAdapter):
    """What both MOSS-Image backends share; the subclasses supply the names and signatures."""

    name = "MOSS-Image"
    text_first = False
    split_encode = False  # neither backend ever had Forge's one-template-per-segment encode

    @classmethod
    def matches(cls, model: Any) -> bool:
        raise NotImplementedError

    # ---------------------------------------------------------------- text side -- #

    def restore_layout(self, text: str) -> str:
        """A brace-stripped layout put back together, where the engine does that itself."""
        return text

    def unescape(self, text: str) -> str:
        """The text as the engine's tokenizer sees it, where the engine unescapes anything."""
        return text

    def split_verbatim(self, line: str) -> tuple[str | None, str]:
        """`(layout, rest)`: the JSON object a layout prompt starts with, and what follows it.

        A MOSS-Image layout is one JSON object, with `bbox` arrays whose square brackets
        `parse_prompt_attention` would read as de-emphasis, so it must never meet the
        parser.  It need not be the whole line: Forge's JSON Builder appends its *Extra*
        field after the object on a new line, and that is prompt text like any other.
        `raw_decode` finds where the object ends.  Whitespace-only text after it is kept
        with the layout, since the engine strips the whole line anyway.

        The engine's own `restore_layout` and `unescape` are applied first, so a layout
        sd-dynamic-prompts has stripped the braces from, or one the JSON Builder has
        escaped the parentheses in (`\\(` is not a JSON escape), is still recognised; the
        returned text is the unescaped one, which is what the engine tokenises.  A line
        that is already chat markup is the engine's whole, since it templates nothing
        around it.
        """
        text = self.unescape(line)
        stripped = text.strip()
        lead = len(text) - len(text.lstrip())

        if stripped.startswith(IM_START):
            return text, ""

        try:
            restored = self.restore_layout(stripped)
        except Exception:
            restored = stripped

        if not restored.startswith("{"):
            return None, text

        try:
            layout, end = json.JSONDecoder().raw_decode(restored)
        except ValueError:
            return None, text
        if not isinstance(layout, dict):
            return None, text

        rest = restored[end:]
        if not rest.strip():
            return text[:lead] + restored, ""

        return text[:lead] + restored[:end], rest

    def is_verbatim(self, line: str) -> bool:
        layout, rest = self.split_verbatim(line)
        return layout is not None and not rest.strip()

    def plain_encode(self, line: str) -> torch.Tensor:
        """The engine's own conditioning for one line, `(T, 4096)`."""
        raise NotImplementedError

    def encode(self, line: str, tokens: list, multipliers: list[float], weighting: emphasis.Emphasis) -> torch.Tensor:
        """`(T, 4096)`, as the engine returns it, the template still on.

        An unweighted stream is the engine's own encode — literally, when the stream is the
        one the engine gives the line, so a prompt with no effective weight is encoded
        byte-identically to the engine on its own.  A claimed weight has already left
        `1.0` behind, so a prompt whose weights are all claimed also takes this path, over
        the tokens of the unweighted sentence.

        An unclaimed weight goes to `encode_token_weights` as a token weight, which is
        ComfyUI's output lerp: each weighted token's hidden state is moved between the
        empty (pad) encoding and its own by the weight.  The engine has no embedding-side
        pass for Forge's emphasis classes to restore, so *Original* and *No norm* both
        mean this lerp here; *None* and *Ignore* mean no weight at all, as they do on
        Krea 2.
        """
        if weighting.name in ("None", "Ignore") or all(m == 1.0 for m in multipliers):
            if list(tokens) == list(self.plain_ids(line) or []):
                return self.plain_encode(line)
            pairs = [(token, 1.0) for token in tokens]
        else:
            pairs = list(zip(tokens, multipliers))

        states = self.engine.text_encoder.encode_token_weights([pairs])[0]
        return states[0]

    # ----------------------------------------------------------------- DiT side -- #

    def attention_problem(self, dit: Any) -> str | None:
        """Why the bias could not reach this backend's attention, or `None`."""
        raise NotImplementedError

    def dit_problem(self, dit: Any) -> str | None:
        layers = getattr(dit, "layers", None)
        if layers is None:
            return "the DiT has no layers"
        if len(layers) != MOSS_LAYERS:
            return f"the DiT has {len(layers)} layers, not {MOSS_LAYERS}"

        dim = _dit_dim(dit)
        if dim != MOSS_DIM:
            return f"the DiT's width is {dim!r}, not {MOSS_DIM}"

        for i, block in enumerate(layers):
            proj = getattr(block, "proj", None)
            if proj is None or not hasattr(proj, "keys"):
                return f"layers[{i}] has no proj"
            if "qkv" not in proj and "v" not in proj:
                return f"layers[{i}].proj has neither qkv nor v"

        refiners = getattr(getattr(dit, "text_embedder", None), "blocks", None)
        if refiners is None:
            return "the DiT has no text_embedder.blocks"
        for i, block in enumerate(refiners):
            if getattr(block, "qkv", None) is None:
                return f"text_embedder.blocks[{i}] has no qkv"

        return self.attention_problem(dit)

    def block_sites(self, dit: Any) -> list[AttentionSite]:
        dim = _dit_dim(dit) or MOSS_DIM
        sites = []

        for block in getattr(dit, "layers", []) or []:
            proj = getattr(block, "proj", None)
            if proj is None:
                continue
            if "qkv" in proj:
                sites.append(AttentionSite(module=block, value=proj["qkv"], value_slice=_values(dim), role=ROLE_BLOCK))
            elif "v" in proj:
                sites.append(AttentionSite(module=block, value=proj["v"], value_slice=None, role=ROLE_BLOCK))

        return sites

    def refiner_sites(self, dit: Any) -> list[AttentionSite]:
        dim = _dit_dim(dit) or MOSS_DIM
        blocks = getattr(getattr(dit, "text_embedder", None), "blocks", []) or []

        return [AttentionSite(module=block, value=block.qkv, value_slice=_values(dim), role=ROLE_REFINER) for block in blocks if getattr(block, "qkv", None) is not None]

    def wrap_attention(self, site: AttentionSite, original: Callable) -> Callable:
        """`Block.forward` / `TextBlock.forward` with the bias put where the attention swap reads it.

        Neither takes a mask, so the bias travels in a copy of `transformer_options` under
        `LIVE_BIAS_KEY`; `force_pytorch_attention`'s wrapper on the attention callable picks
        it up there, inside this block's own attention call and nowhere else.
        `transformer_options` is found by name in whatever shape the call arrives
        (`CallSignature`) and everything else passes through untouched: Forge's `c6c2b43d`
        gave the native block `ref`, `ref_kv` and `kv_out` after it, positionally, and the
        extension's block and refiner have different parameter lists again.
        """
        sig = CallSignature(original)

        @wraps(original)
        def negpip_forward(*args, **kwargs):
            options = sig.get(args, kwargs, "transformer_options", {})

            with active(site, options) as bias:
                if bias is not None:
                    args, kwargs = sig.replace(args, kwargs, "transformer_options", _options_with(options, bias))
                return original(*args, **kwargs)

        return negpip_forward


# ================================================================================ #


class MossExtensionAdapter(MossAdapter):
    """The `moss-image-forge-neo` extension's `MossImageEngine` and `MossImageDiT`.

    Text: `MossTextEngine`, with `token_ids()` and `encode()`, `TEMPLATE` and `MAX_LENGTH`
    as module constants, `restore_layout_json` beside them; the tokenizer is a
    `Qwen2TokenizerFast` (the extension ships `tokenizer.json`), so `return_offsets_mapping`
    works and the exact one-pass path applies.  The prompt reaches the tokenizer as typed,
    escapes included.

    DiT: `Block.forward(x, rope, mod, dtype, transformer_options=None)` and
    `TextBlock.forward(x, rope, dtype, transformer_options=None)`; attention is
    `transformer.attention(q, k, v, rope, heads, transformer_options)`, which rotates q
    and k, splits the heads and ends in `kernels.attention(q, k, v, transformer_options=...)`,
    looked up on the `kernels` module at call time — the swap point for the bias.
    """

    tokenizer_kwargs = {"add_special_tokens": False}  # as `MossTextEngine.token_ids` encodes

    @classmethod
    def matches(cls, model: Any) -> bool:
        return type(model).__name__ == MOSS_EXTENSION_ENGINE or bool(getattr(model, "is_moss_image", False))

    def __init__(self, model: Any):
        super().__init__(model)
        module = _module_of(getattr(model, "text_engine_moss", None))
        self.template = str(getattr(module, "TEMPLATE", ""))
        self.max_length = getattr(module, "MAX_LENGTH", None)
        self._restore_layout = getattr(module, RESTORE_LAYOUT, None)

    def engine_problem(self) -> str | None:
        engine = getattr(self.model, "text_engine_moss", None)
        if engine is None:
            return "the engine has no text_engine_moss"

        module = _module_of(engine)
        for name in ("TEMPLATE", "MAX_LENGTH"):
            if not hasattr(module, name):
                return f"the text engine module has no {name}"
        if "{}" not in str(module.TEMPLATE):
            return "the text engine's TEMPLATE has no placeholder"
        if not isinstance(module.MAX_LENGTH, int):
            return "the text engine's MAX_LENGTH is not an integer"

        for name in ("token_ids", "encode"):
            if not callable(getattr(engine, name, None)):
                return f"the text engine has no {name}"
        if not callable(getattr(getattr(engine, "text_encoder", None), "encode_token_weights", None)):
            return "the text encoder has no encode_token_weights"
        if getattr(getattr(engine, "tokenizer", None), "tokenizer", None) is None:
            return "the tokenizer wrapper has no tokenizer"

        return None

    @property
    def engine(self) -> Any:
        return self.model.text_engine_moss

    def restore_layout(self, text: str) -> str:
        return self._restore_layout(text) if self._restore_layout is not None else text

    def plain_ids(self, line: str) -> list[int] | None:
        """The engine's own token ids: `token_ids` strips, rebuilds a layout, templates and truncates."""
        return list(self.engine.token_ids(line))

    def plain_encode(self, line: str) -> torch.Tensor:
        return self.engine.encode(line)

    def attention_problem(self, dit: Any) -> str | None:
        kernels = getattr(_module_of(dit), "kernels", None)
        if not callable(getattr(kernels, "attention", None)):
            return "the transformer module has no kernels.attention"
        return None

    def force_pytorch_attention(self, dit: Any, enable: bool):
        """Swap the extension's `kernels.attention` for one that takes the bias as a mask.

        `transformer.attention` rotates q and k, splits the heads and calls
        `kernels.attention(q, k, v, transformer_options=...)` — looked up on the module at
        call time, so rebinding the attribute reaches every block and refiner.  The swap
        pads the bias to the `[image, text]` key order and runs `attention_pytorch`, the
        backend that accepts an additive float mask; a call that carries no bias goes to
        the original untouched.
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
        def negpip_attention(q, k, v, transformer_options=None, **kwargs):
            options = transformer_options if isinstance(transformer_options, dict) else {}
            bias = options.get(LIVE_BIAS_KEY, None)

            mask = None
            if bias is not None and torch.is_tensor(q) and q.ndim == 4:
                #   q, k, v are (B, H, S, D) here; the keys are the whole joint sequence
                mask = attention_bias(int(q.shape[0]), int(k.shape[2]), bias, text_first, like=q, within=int(q.shape[2]))

            if mask is None:
                return original(q, k, v, transformer_options=transformer_options, **kwargs)

            return attention_pytorch(q, k, v, int(q.shape[1]), mask=mask, skip_reshape=True, transformer_options=options)

        negpip_attention._negpip = True

        setattr(kernels, ORIGINAL_ATTENTION, original)
        kernels.attention = negpip_attention


# ================================================================================ #


class MossNativeAdapter(MossAdapter):
    """Forge Neo's own `MOSSImage` engine and `backend.nn.moss_image.MOSSImage`.

    Text: `Qwen359BEngine` at `text_processing_engine_qwen`, the same attribute Krea 2's
    engine uses, with `llama_template` and no truncation.  `__call__` strips the line (an
    empty one becomes a single space), templates it unless it already starts with
    `<|im_start|>`, and tokenises through `SDTokenizer.tokenize_with_weights`, which
    unescapes `\\(` and `\\)` first; `plain_ids` goes through the same call, so the ids are
    the engine's own.  The tokenizer is instantiated by `model_index.json`'s class name,
    `Qwen2Tokenizer` — the slow one on the `transformers` Forge pins, as for Krea 2 — so
    `return_offsets_mapping` raises and `_tokenize_single` takes its decoded-offsets branch.

    DiT: `Block.forward(x, mod, pe, transformer_options={})` and
    `TextBlock.forward(x, pe, transformer_options={})`, called positionally by the DiT and
    by Skip Layer Guidance's `patches_replace` wrapper; attention is the module-level
    `attention(q, k, v, q_norm, k_norm, pe, heads, transformer_options)`, which norms and
    rotates q and k and ends in `attention_function(q, k, v, heads, transformer_options=...)`
    with the heads still flat, `(B, S, H*D)` — the name is bound in the module at import,
    so rebinding `moss_image.attention_function` is the swap point, as `krea.attention_function`
    is on Krea 2.  The model has no `dim` attribute; the width is read off `video_embedder`.
    """

    @classmethod
    def matches(cls, model: Any) -> bool:
        return type(model).__name__ == MOSS_NATIVE_ENGINE and hasattr(model, "text_processing_engine_qwen")

    def __init__(self, model: Any):
        super().__init__(model)
        self.template = str(getattr(getattr(model, "text_processing_engine_qwen", None), "llama_template", ""))

    def engine_problem(self) -> str | None:
        engine = getattr(self.model, "text_processing_engine_qwen", None)
        if engine is None:
            return "the engine has no text_processing_engine_qwen"
        if "{}" not in str(getattr(engine, "llama_template", "")):
            return "the text engine's llama_template has no placeholder"
        if not callable(getattr(getattr(engine, "text_encoder", None), "encode_token_weights", None)):
            return "the text encoder has no encode_token_weights"
        tokenizer = getattr(engine, "tokenizer", None)
        if not callable(getattr(tokenizer, "tokenize_with_weights", None)):
            return "the tokenizer wrapper has no tokenize_with_weights"
        if getattr(tokenizer, "tokenizer", None) is None:
            return "the tokenizer wrapper has no tokenizer"
        if not callable(engine):
            return "the text engine is not callable"

        return None

    @property
    def engine(self) -> Any:
        return self.model.text_processing_engine_qwen

    def unescape(self, text: str) -> str:
        """`escape_important` then `unescape_important`, as `tokenize_with_weights` applies them."""
        return text.replace("\\)", ")").replace("\\(", "(")

    def plain_ids(self, line: str) -> list[int] | None:
        """The engine's own token ids, through the very call `__call__` makes.

        Since `c6c2b43d` the engine's `_tokenize_with_weights` templates every line (and
        splices image placeholders in, none here); before it, `__call__` templated inline
        and let a line that already started with `<|im_start|>` through as it was.
        """
        text = line.strip() or " "
        engine = self.engine

        tokenize = getattr(engine, "_tokenize_with_weights", None)
        if callable(tokenize):
            chunks = tokenize(text, [])
        else:
            if not text.startswith(IM_START):
                text = self.template.format(text)
            chunks = engine.tokenizer.tokenize_with_weights(text, disable_weights=True)

        return [token for token, _ in chunks[0]] if chunks else []

    def plain_encode(self, line: str) -> torch.Tensor:
        return self.engine([line])[0]

    def _referencing(self) -> bool:
        """Whether the engine would take its reference-image branch for a positive prompt."""
        methods = getattr(_module_of(self.model), "REFERENCE_METHODS", None)
        mode = getattr(opts, "moss_do_reference", None)
        if methods is not None:
            return mode in methods
        return mode not in (None, "Disable")

    def hand_back(self, prompt) -> str | None:
        """Whether `MOSSImage.get_learned_conditioning` would take its reference-image branch.

        MOSS-Image Edit (`c6c2b43d`) encodes the positive prompt alongside reference
        images — with vision placeholders in the stream for one method — and sets up the
        reference latents and their cached K/V as a side effect of that encode.  The whole
        call is handed back rather than half-applied, as on Krea 2; the negative prompt
        keeps its rows.  Read-only: the original consumes `ini_latent` and clears the
        reference state itself.
        """
        if getattr(prompt, "is_negative_prompt", False):
            return None
        if not self._referencing():
            return None

        if bool(getattr(self.model, "ref_latents", None)) or getattr(self.model, "ini_latent", None) is not None:
            return "NegPiP does not apply to the MOSS-Image reference/Edit path; the positive prompt's weights are ignored there"

        return None

    def before_encode(self, prompt):
        if getattr(prompt, "is_negative_prompt", False):
            return

        #   the bookkeeping `MOSSImage.get_learned_conditioning` does before a plain encode,
        #   and which the text hook replaces: the img2img latent is consumed and the
        #   reference state dropped, so a stale one cannot put the DiT into edit mode
        if hasattr(self.model, "ini_latent"):
            self.model.ini_latent = None
        dynamic_args.ref_latents.clear()
        for name in ("moss_reference_method", "moss_ref_kv"):
            if hasattr(dynamic_args, name):
                setattr(dynamic_args, name, None)

    def attention_problem(self, dit: Any) -> str | None:
        module = _module_of(dit)
        if not callable(getattr(module, "attention", None)):
            return "the DiT's module has no attention function"
        if not callable(getattr(module, "attention_function", None)):
            return "the DiT's module has no attention_function to swap"
        return None

    def force_pytorch_attention(self, dit: Any, enable: bool):
        """Point the DiT module's `attention_function` at the backend that takes an additive mask.

        `attention()` calls it with the heads still flat, so the swap hands `attention_pytorch`
        the same `(B, S, H*D)` tensors and lets it split them; the keys are the whole joint
        sequence in the blocks and the text alone in the refiners.  On the Edit path
        (`c6c2b43d`) the reference images' keys and values are appended after the
        sequence's own, so the bias is confined to the first `q.shape[1]` keys, where the
        text still is.  A call that carries no bias goes to the original untouched.
        """
        module = _module_of(dit)
        if module is None or not hasattr(module, "attention_function"):
            return

        if not enable:
            original = getattr(module, ORIGINAL_ATTENTION, None)
            if original is not None:
                module.attention_function = original
                delattr(module, ORIGINAL_ATTENTION)
            return

        if hasattr(module, ORIGINAL_ATTENTION):
            return

        original = module.attention_function
        text_first = self.text_first

        @wraps(original)
        def negpip_attention(q, k, v, heads, *args, transformer_options=None, **kwargs):
            options = transformer_options if isinstance(transformer_options, dict) else {}
            bias = options.get(LIVE_BIAS_KEY, None)

            mask = None
            if bias is not None and torch.is_tensor(k) and k.ndim == 3 and not args and not kwargs.get("skip_reshape", False):
                mask = attention_bias(int(q.shape[0]), int(k.shape[1]), bias, text_first, like=q, within=int(q.shape[1]))

            if mask is None:
                return original(q, k, v, heads, *args, transformer_options=transformer_options, **kwargs)

            return attention_pytorch(q, k, v, heads, mask=mask, transformer_options=options)

        negpip_attention._negpip = True

        setattr(module, ORIGINAL_ATTENTION, original)
        module.attention_function = negpip_attention


__all__ = ["IM_START", "LIVE_BIAS_KEY", "MOSS_DIM", "MOSS_EXTENSION_ENGINE", "MOSS_LAYERS", "MOSS_NATIVE_ENGINE", "MossAdapter", "MossExtensionAdapter", "MossNativeAdapter"]
