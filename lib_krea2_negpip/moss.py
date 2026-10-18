"""
The MOSS-Image adapter: the third-party `moss-image-forge-neo` extension's `MossImageEngine`
and `MossImageDiT` (https://github.com/eduardoabreu81/moss-image-forge-neo, 0.1.x).

Nothing here is part of Forge Neo, so everything is resolved from the loaded model and
checked before it is touched — the module constants through `sys.modules`, never by
importing the extension by path — and a mismatch stands the extension down with a logged
reason naming what is missing.  After flyfront/sd-forge-negpip's `neo-moss` branch, which
mapped the extension out first; the hooks are this extension's own.

Text: `MossTextEngine` is Qwen3.5-9B through Forge's `SDClipModel` with `layer="last"`, so
the conditioning is the last hidden state, `(T, 4096)` per line with no batch axis.  The
prompt is wrapped once in `TEMPLATE` and the template is **not** stripped: the DiT reads
every token, so the rows cover the whole sequence and the template tokens carry a neutral
`1.0` / `0.0`.  The tokenizer is `Qwen2TokenizerFast` (the checkpoint ships `tokenizer.json`),
so `return_offsets_mapping` works and the exact one-pass path applies.  The engine
hard-wires `EmphasisNone` and passes the prompt verbatim, brackets included; an unclaimed
weight gets the ComfyUI output lerp that `encode_token_weights` applies, which is what a
ComfyUI user of the model gets.  A layout prompt — the whole prompt as one JSON object,
possibly brace-stripped by sd-dynamic-prompts — never goes near the prompt parser, whose
square brackets would read a `bbox` as de-emphasis.

DiT: `MossImageDiT` runs 42 `Block`s over the joint sequence `[image, text]`, text **last**,
after a projection and two text-only `TextBlock` refiners (`text_embedder.blocks`).  The
value projection is inside `block.proj`: a fused `proj["qkv"]` on Comfy-Org's files, where
the values are the last third of the output columns, or a separate `proj["v"]` on upstream's
diffusers export; the refiners' `qkv` is always fused.  Attention is a module-level
function with no mask argument that ends in `kernels.attention`, so the logit bias is
applied by rebinding *that* while a bias is in play — it receives `transformer_options`
and the already-rotated heads, and the only thing it lacks is the mask.  Nothing of the
block is re-implemented.

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

from backend.attention import attention_pytorch
from backend.text_processing import emphasis

from lib_krea2_negpip import ROLE_BLOCK, ROLE_REFINER
from lib_krea2_negpip.adapters import AttentionSite, ModelAdapter
from lib_krea2_negpip.dit import ORIGINAL_ATTENTION, active, attention_bias

#   `type(shared.sd_model).__name__` for the extension's `forge_moss/native/engine.py`
MOSS_ENGINE = "MossImageEngine"

#   the identifying shape of `MossImageDiT`: 42 joint blocks of width 3072
MOSS_LAYERS = 42
MOSS_DIM = 3072

#   the attribute of the extension's text-engine module that rebuilds a brace-stripped layout
RESTORE_LAYOUT = "restore_layout_json"

#   the key a hooked block puts the bias under for the attention swap to find; distinct
#   from the DiT-level key so an unselected block, which sees that one too, stays untouched
LIVE_BIAS_KEY = "krea2_negpip_live_bias"


def _module_of(obj: Any):
    return sys.modules.get(type(obj).__module__)


class MossAdapter(ModelAdapter):
    name = "MOSS-Image"
    text_first = False
    split_encode = False  # the extension never had Forge's one-template-per-segment encode
    tokenizer_kwargs = {"add_special_tokens": False}  # as `MossTextEngine.token_ids` encodes

    @classmethod
    def matches(cls, model: Any) -> bool:
        return type(model).__name__ == MOSS_ENGINE or bool(getattr(model, "is_moss_image", False))

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

    # ---------------------------------------------------------------- text side -- #

    @property
    def engine(self) -> Any:
        return self.model.text_engine_moss

    def is_verbatim(self, line: str) -> bool:
        """Whether the line is a layout prompt: one JSON object, its brackets data.

        sd-dynamic-prompts reads `{x}` as a one-option variant and strips every brace;
        the extension rebuilds the layout before tokenising, and so does this check.
        """
        text = line.strip()
        if self._restore_layout is not None:
            try:
                text = self._restore_layout(text)
            except Exception:
                return False

        if not text.startswith("{"):
            return False

        try:
            return isinstance(json.loads(text), dict)
        except ValueError:
            return False

    def plain_ids(self, line: str) -> list[int] | None:
        """The engine's own token ids: `token_ids` strips, rebuilds a layout, templates and truncates."""
        return list(self.engine.token_ids(line))

    def encode(self, line: str, tokens: list, multipliers: list[float], weighting: emphasis.Emphasis) -> torch.Tensor:
        """`(T, 4096)`, as `MossTextEngine.encode` returns it, the template still on.

        An unweighted stream is the engine's own encode — literally, when the stream is the
        one `token_ids` gives the line, so a prompt with no effective weight is encoded
        byte-identically to the extension on its own.  A claimed weight has already left
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
            if list(tokens) == list(self.engine.token_ids(line)):
                return self.engine.encode(line)
            pairs = [(token, 1.0) for token in tokens]
        else:
            pairs = list(zip(tokens, multipliers))

        states = self.engine.text_encoder.encode_token_weights([pairs])[0]
        return states[0]

    # ----------------------------------------------------------------- DiT side -- #

    def dit_problem(self, dit: Any) -> str | None:
        layers = getattr(dit, "layers", None)
        if layers is None:
            return "the DiT has no layers"
        if len(layers) != MOSS_LAYERS:
            return f"the DiT has {len(layers)} layers, not {MOSS_LAYERS}"

        dim = getattr(dit, "dim", None)
        if dim != MOSS_DIM:
            return f"the DiT's dim is {dim!r}, not {MOSS_DIM}"

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

        kernels = getattr(_module_of(dit), "kernels", None)
        if not callable(getattr(kernels, "attention", None)):
            return "the transformer module has no kernels.attention"

        return None

    @staticmethod
    def _values(dim: int) -> slice:
        """The value third of a fused q/k/v output: `chunk(3, -1)` hands `v` the last columns."""
        return slice(2 * dim, 3 * dim)

    def block_sites(self, dit: Any) -> list[AttentionSite]:
        dim = int(getattr(dit, "dim", MOSS_DIM))
        sites = []

        for block in getattr(dit, "layers", []) or []:
            proj = getattr(block, "proj", None)
            if proj is None:
                continue
            if "qkv" in proj:
                sites.append(AttentionSite(module=block, value=proj["qkv"], value_slice=self._values(dim), role=ROLE_BLOCK))
            elif "v" in proj:
                sites.append(AttentionSite(module=block, value=proj["v"], value_slice=None, role=ROLE_BLOCK))

        return sites

    def refiner_sites(self, dit: Any) -> list[AttentionSite]:
        dim = int(getattr(dit, "dim", MOSS_DIM))
        blocks = getattr(getattr(dit, "text_embedder", None), "blocks", []) or []

        return [AttentionSite(module=block, value=block.qkv, value_slice=self._values(dim), role=ROLE_REFINER) for block in blocks if getattr(block, "qkv", None) is not None]

    def wrap_attention(self, site: AttentionSite, original: Callable) -> Callable:
        """`Block.forward` / `TextBlock.forward` with the bias put where the attention swap reads it.

        Neither takes a mask, so the bias travels in a copy of `transformer_options` under
        `LIVE_BIAS_KEY`; `force_pytorch_attention`'s wrapper on `kernels.attention` picks it
        up there, inside this block's own attention call and nowhere else.
        """

        def options_with(transformer_options, bias):
            if bias is None:
                return transformer_options
            options = dict(transformer_options or {})
            options[LIVE_BIAS_KEY] = bias
            return options

        if site.role == ROLE_REFINER:

            @wraps(original)
            def negpip_forward(x, rope, dtype, transformer_options=None):
                with active(site, transformer_options) as bias:
                    #   keyword past the required positionals, as everywhere in this extension
                    return original(x, rope, dtype, transformer_options=options_with(transformer_options, bias))

        else:

            @wraps(original)
            def negpip_forward(x, rope, mod, dtype, transformer_options=None):
                with active(site, transformer_options) as bias:
                    return original(x, rope, mod, dtype, transformer_options=options_with(transformer_options, bias))

        return negpip_forward

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
                mask = attention_bias(int(q.shape[0]), int(k.shape[2]), bias, text_first, like=q)

            if mask is None:
                return original(q, k, v, transformer_options=transformer_options, **kwargs)

            return attention_pytorch(q, k, v, int(q.shape[1]), mask=mask, skip_reshape=True, transformer_options=options)

        negpip_attention._negpip = True

        setattr(kernels, ORIGINAL_ATTENTION, original)
        kernels.attention = negpip_attention


__all__ = ["LIVE_BIAS_KEY", "MOSS_DIM", "MOSS_ENGINE", "MOSS_LAYERS", "MossAdapter"]
