"""
The Krea 2 adapter: Forge Neo's own `Krea2` engine and `SingleStreamDiT`.

Text: `Qwen3VL4BEngine` wraps the prompt in a chat template, encodes it with Forge's
`SDClipModel`, takes a 12-layer tap stack of the Qwen3-VL hidden states and strips the
template off the front of the conditioning.  Since Forge Neo `21886f41` (2026-09-30) the
engine has a read-only `EmphasisNone` and hands the prompt to the encoder as literal text,
so the emphasis pass for unclaimed weights is put back here, on the input embeddings,
where the engine applied it before that commit.

DiT: `SingleStreamDiT` concatenates the fused text and the image patches, text first, and
runs plain self-attention over the pair (`backend/nn/krea.py`).  Every `Attention` has a
separate `wv`; the DiT hands `None` to every block's `mask`, so the logit bias rides that
argument, and the module-level `attention_function` is pointed at the plain SDPA path
while a bias is in play.  `txtfusion.layerwise_blocks` are deliberately not touched: they
run at `(batch * seq, taps, dim)`, so their sequence axis is the tap stack, and the rows
do not index it.

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
from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    from backend.diffusion_engine.krea import Krea2
    from backend.text_processing.krea2_engine import Qwen3VL4BEngine

import torch

from backend import memory_management
from backend.args import dynamic_args
from backend.attention import attention_pytorch
from backend.text_processing import emphasis
from modules.processing import logger
from modules.shared import opts

from lib_krea2_negpip import ROLE_BLOCK, ROLE_REFINER
from lib_krea2_negpip.adapters import AttentionSite, ModelAdapter
from lib_krea2_negpip.dit import ORIGINAL_ATTENTION, CallSignature, active, attention_bias

#   `type(shared.sd_model).__name__` for `backend/diffusion_engine/krea.py`
KREA2_ENGINE = "Krea2"

#   the Qwen3-VL tap stack Krea 2 conditioning carries: 12 layers x 2560 features,
#   flattened into the 30720-wide `crossattn` the DiT unpacks in `_unpack_context`
KREA2_TAP_LAYERS = 12
KREA2_TAP_DIM = 2560

#   `<|im_start|>`; the engine no longer exposes it as `id_template`
ID_TEMPLATE = 151644

#   `<|im_start|>user\n` — the three tokens the template strip steps over
ID_USER = 872
ID_NEWLINE = 198

_warned_mask = False


def _as_token_id(token) -> int | None:
    try:
        return int(token)
    except TypeError:
        return None


def is_krea2_dit(dit: Any) -> bool:
    """Whether this diffusion model really is the Krea 2 `SingleStreamDiT`.

    Duck-typed on what this extension actually touches, and nothing else: Forge moved the
    tap-stack unpacking out of the DiT and into the text engine, and detecting on the
    `_unpack_context` that used to do it made the extension stand down silently — the
    worst possible failure, since everything else still looked healthy.
    """
    try:
        txtlayers = int(getattr(dit, "txtlayers", 0))
        txtdim = int(getattr(dit, "txtdim", 0))
    except (TypeError, ValueError):
        return False

    if txtlayers != KREA2_TAP_LAYERS or txtdim != KREA2_TAP_DIM:
        return False

    return hasattr(dit, "blocks") and hasattr(dit, "txtmlp") and hasattr(getattr(dit, "txtfusion", None), "refiner_blocks")


class Krea2Adapter(ModelAdapter):
    name = "Krea 2"
    text_first = True
    split_encode = True

    @classmethod
    def matches(cls, model: Any) -> bool:
        return type(model).__name__ == KREA2_ENGINE

    def __init__(self, model: "Krea2"):
        super().__init__(model)
        self.template = self.engine.llama_template

    # ---------------------------------------------------------------- text side -- #

    @property
    def engine(self) -> "Qwen3VL4BEngine":
        return self.model.text_processing_engine_qwen

    def hand_back(self, prompt) -> str | None:
        """Whether `Krea2.get_learned_conditioning` would take its reference-image branch.

        Krea 2 Edit encodes the prompt alongside reference images, on a path that carries
        no per-token rows.  The whole call is handed back rather than half-applied.
        Read-only — the original consumes `ini_latent` and clears `ref_latents` itself,
        and doing that here as well would eat the state before Forge ever sees it.
        """
        if getattr(prompt, "is_negative_prompt", False):
            return None
        if not getattr(opts, "krea2_do_reference", False):
            return None

        if bool(getattr(self.model, "ref_latents", None)) or getattr(self.model, "ini_latent", None) is not None:
            return "NegPiP does not apply to the Krea 2 reference/Edit path; the positive prompt's weights are ignored there"

        return None

    def before_encode(self, prompt):
        if getattr(prompt, "is_negative_prompt", False):
            return

        #   the bookkeeping `Krea2.get_learned_conditioning` does before encoding, and
        #   which the text hook replaces: the img2img latent is consumed and the reference
        #   list dropped, so a stale one cannot put the DiT into edit mode
        if hasattr(self.model, "ini_latent"):
            self.model.ini_latent = None
        dynamic_args.ref_latents.clear()

    def template_end(self, tokens: list, seq_len: int) -> int:
        """The offset `Qwen3VL4BEngine.__call__` slices the template off at, computed alone.

        The engine does it inline over its own token stream, so there is nothing to call.
        """
        template_end = 0
        count_im_start = 0

        for i, token in enumerate(tokens):
            try:
                elem = int(token)
            except TypeError:
                continue

            if elem == ID_TEMPLATE and count_im_start < 2:
                template_end = i
                count_im_start += 1

        if seq_len > (template_end + 3):
            if int(tokens[template_end + 1]) == ID_USER and int(tokens[template_end + 2]) == ID_NEWLINE:
                template_end += 3

        return template_end

    def segment_text_span(self, segment: list) -> tuple[int, int]:
        """`[start, end)` of the prompt fragment inside one templated segment.

        The split encode wraps *every* weighted segment in the whole chat template, so a
        prompt carrying weights tokenises to several copies of the template with the
        fragments spliced between them.  Emphasis scales all of it — that was Forge's own
        behaviour, and a weight this extension does not claim keeps it — but scaling the
        system instruction is not what `(word:-1.0)` asks for.  Both rows are confined to
        the fragment: everything between `<|im_start|>user\\n` and the `<|im_end|>` that
        ends the user turn.

        Falls back to the whole segment if the landmarks are not where the template puts
        them, which is also what happens for a prompt already written as raw chat markup.
        """
        starts = [i for i, token in enumerate(segment) if _as_token_id(token) == ID_TEMPLATE]
        if len(starts) < 3:
            return 0, len(segment)

        user = starts[1]
        if _as_token_id(segment[user + 1]) != ID_USER or _as_token_id(segment[user + 2]) != ID_NEWLINE:
            return 0, len(segment)

        start = user + 3
        end = starts[2] - 2  # the `<|im_end|>\n` that closes the user turn

        if not 0 <= start <= end <= len(segment):
            return 0, len(segment)

        return start, end

    def encode(self, line: str, tokens: list, multipliers: list[float], weighting: emphasis.Emphasis) -> torch.Tensor:
        """`SDClipModel.forward` for one line, with the emphasis pass put back in.

        Before `21886f41` the engine applied the selected emphasis to the input
        embeddings, between building them and running the transformer; `SDClipModel.forward`
        has no step there, so this is that forward with the step restored.  All-`1.0`
        multipliers skip it, which `EmphasisOriginal` makes an exact identity anyway, so an
        unweighted line is encoded exactly as the engine encodes it.

        Returns `(seq, taps, dim)`, the token axis first as `Qwen3VL4BEngine.__call__`
        hands it to `SingleStreamDiT.txtfusion`; the template is still on the front.
        """
        encoder = self.engine.text_encoder
        device = encoder.transformer.get_input_embeddings().weight.device

        embeds, mask, num_tokens, embeds_info = encoder.process_tokens([tokens], device)

        #   images would make the lengths differ; Forge skipped emphasis then, and so does this
        if embeds.shape[1] == len(multipliers) and any(m != 1.0 for m in multipliers):
            weighted = weighting(embeds, torch.asarray([multipliers]).to(embeds))
            if weighted is not None:  # `EmphasisNone` and `EmphasisIgnore` return nothing
                embeds = weighted

        outputs = encoder.transformer(
            None,
            mask if encoder.enable_attention_masks else None,
            embeds=embeds,
            num_tokens=num_tokens,
            intermediate_output=encoder.layer,  # the tap list; Krea 2 never uses "last"
            final_layer_norm_intermediate=encoder.layer_norm_hidden_state,
            dtype=torch.float32,
            embeds_info=embeds_info,
        )

        z = outputs[1].float().to(device=memory_management.intermediate_device())  # (1, taps, seq, dim)
        return z[0].permute(1, 0, 2)

    # ----------------------------------------------------------------- DiT side -- #

    def dit_problem(self, dit: Any) -> str | None:
        if is_krea2_dit(dit):
            return None

        return "the diffusion model is not Krea 2's SingleStreamDiT (txtlayers 12, txtdim 2560, blocks, txtmlp, txtfusion.refiner_blocks)"

    @staticmethod
    def _site(block: Any, role: str) -> AttentionSite | None:
        attn = getattr(block, "attn", None)
        if attn is None:
            return None
        return AttentionSite(module=attn, value=getattr(attn, "wv", None), value_slice=None, role=role)

    def block_sites(self, dit: Any) -> list[AttentionSite]:
        sites = [self._site(block, ROLE_BLOCK) for block in getattr(dit, "blocks", []) or []]
        return [site for site in sites if site is not None]

    def refiner_sites(self, dit: Any) -> list[AttentionSite]:
        txtfusion = getattr(dit, "txtfusion", None)
        sites = [self._site(block, ROLE_REFINER) for block in getattr(txtfusion, "refiner_blocks", []) or []]
        return [site for site in sites if site is not None]

    def wrap_attention(self, site: AttentionSite, original: Callable) -> Callable:
        """`Attention.forward` with the logit bias substituted for the unused `mask`.

        Krea 2 never passes an attention mask — `SingleStreamDiT.forward` hands `None` to
        every block and to `txtfusion` — so the bias rides the `mask` argument that is
        already threaded through `Attention.forward` to `attention_function`, and no
        attention math needs copying.  `mask` and `transformer_options` are found by name
        in whatever shape the call arrives (`CallSignature`), so a parameter Forge adds to
        the block neither breaks the wrapper nor takes another one's value — the ComfyUI
        node hit exactly the latter when Krea 2 Edit added `ref_latents` to its signature.
        """
        text_first = self.text_first
        sig = CallSignature(original)

        @wraps(original)
        def negpip_forward(*args, **kwargs):
            global _warned_mask

            transformer_options = sig.get(args, kwargs, "transformer_options", {})

            with active(site, transformer_options) as bias:
                if bias is not None:
                    x = sig.get(args, kwargs, "x")
                    mask = sig.get(args, kwargs, "mask")

                    if mask is None and torch.is_tensor(x):
                        args, kwargs = sig.replace(args, kwargs, "mask", attention_bias(int(x.shape[0]), int(x.shape[1]), bias, text_first, like=x))
                    elif mask is not None and not _warned_mask:
                        #   Krea 2 has never passed one; if that changes, adding the bias
                        #   to somebody else's mask is not obviously the right merge, so
                        #   say so
                        _warned_mask = True
                        logger.warning("NegPiP: attention already carries a mask, skipping the emphasis bias")

                return original(*args, **kwargs)

        return negpip_forward

    def force_pytorch_attention(self, dit: Any, enable: bool):
        """Point `krea.attention_function` at the backend that takes an additive mask.

        `backend/nn/krea.py` binds the name at import time, so the module's own reference
        is the one to rebind — the same reason `compile_conditions` has to be patched in
        two places.
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


__all__ = ["KREA2_ENGINE", "KREA2_TAP_DIM", "KREA2_TAP_LAYERS", "Krea2Adapter", "is_krea2_dit"]
