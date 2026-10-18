"""
The model adapter: everything about a model that `text.py` and `dit.py` must not know.

NegPiP's two levers are the same on every single-stream DiT — scale a text token's value
vector, bias its attention logit — and so is the way the rows reach them, as per-token
conditioning alongside the prompt.  What differs between models is plumbing: where the
text engine hangs off the diffusion engine, how the prompt is templated, whether the
template is stripped from the conditioning, where the text sits in the joint sequence,
which `nn.Linear` carries the value projection, and how an additive mask gets into the
attention call.  One adapter per model answers those questions; the two modules above
ask them and nothing else.

`detect` picks the adapter for the loaded model, or `None` for a model nobody handles.

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

from dataclasses import dataclass
from typing import Any, Callable

import torch

from backend.text_processing import emphasis


@dataclass(frozen=True)
class AttentionSite:
    """One place the value lever can act.

    * `module` — the module whose `forward` is wrapped to park the rows for the duration
      of one call and to get the logit bias into the attention; it is whatever receives
      `transformer_options` closest to the attention.  Krea 2's `Attention`, MOSS-Image's
      `Block` and `TextBlock`.
    * `value` — the `nn.Linear` whose output carries the value vectors, wrapped to scale
      the rows the mask names.
    * `value_slice` — the output columns of `value` that *are* the values, for a fused
      q/k/v projection; `None` when the whole output is the values.
    """

    module: Any
    value: Any
    value_slice: slice | None
    role: str


class ModelAdapter:
    """Base class: the questions `text.py` and `dit.py` ask about a model.

    The defaults are what both known models share — a Hugging Face tokenizer under an
    `SDTokenizer` wrapper, a chat template with one `{}` placeholder, no verbatim lines —
    so an adapter only overrides what its model does differently.
    """

    name: str = ""
    """how the model is named in log lines"""

    text_first: bool = True
    """whether the text tokens lead the joint sequence (Krea 2) or trail it (MOSS-Image)"""

    split_encode: bool = True
    """whether `single_pass=False` has a meaning: Forge's old one-template-per-segment encode"""

    max_length: int | None = None
    """the engine's own truncation of the templated token stream, if it has one"""

    template: str = ""
    """the chat template the prompt is formatted into, with one `{}` placeholder"""

    tokenizer_kwargs: dict = {}
    """passed to every call of the Hugging Face tokenizer, so the ids match the engine's own"""

    def __init__(self, model: Any):
        self.model = model

    # ---------------------------------------------------------------- text side -- #

    @property
    def engine(self) -> Any:
        raise NotImplementedError

    @property
    def hf_tokenizer(self) -> Any:
        """The Hugging Face tokenizer, which `21886f41` moved under an `SDTokenizer` wrapper."""
        return self.engine.tokenizer.tokenizer

    def engine_problem(self) -> str | None:
        """Why the text side cannot be hooked, or `None`; checked before anything is patched."""
        return None

    def hand_back(self, prompt) -> str | None:
        """A reason to hand this whole call back to the engine's own encode, or `None`."""
        return None

    def before_encode(self, prompt):
        """The bookkeeping the engine's `get_learned_conditioning` does before encoding."""

    def is_verbatim(self, line: str) -> bool:
        """Whether this line must never go through the prompt parser."""
        return False

    def plain_ids(self, line: str) -> list[int] | None:
        """The engine's own token ids for a line it would encode unchanged, or `None` to
        build them from the parsed segments like any other line."""
        return None

    def templated_ids(self, texts: list[str]) -> list[list[int]]:
        """Each text stripped and wrapped in the whole chat template, one token list per text."""
        ids = self.hf_tokenizer([self.template.format(text.strip()) for text in texts], **self.tokenizer_kwargs)["input_ids"]
        if self.max_length is not None:
            ids = [stream[: self.max_length] for stream in ids]
        return ids

    def raw_ids(self, texts: list[str]) -> list[list[int]]:
        """Each text tokenised on its own, no template."""
        return self.hf_tokenizer(texts, **self.tokenizer_kwargs)["input_ids"]

    def reported_offsets(self, templated: str):
        """The tokenizer's own `(input_ids, offset_mapping)` for one templated prompt, or `None`."""
        try:
            encoded = self.hf_tokenizer([templated], return_offsets_mapping=True, **self.tokenizer_kwargs)
        except (NotImplementedError, TypeError, ValueError):
            return None

        offsets = encoded.get("offset_mapping") if hasattr(encoded, "get") else None
        if not offsets:
            return None

        return encoded["input_ids"][0], offsets[0]

    def template_end(self, tokens: list, seq_len: int) -> int:
        """How many leading positions the engine cuts off the conditioning."""
        return 0

    def segment_text_span(self, segment: list) -> tuple[int, int]:
        """`[start, end)` of the prompt fragment inside one templated segment (split encode only)."""
        return 0, len(segment)

    def encode(self, line: str, tokens: list, multipliers: list[float], weighting: emphasis.Emphasis) -> torch.Tensor:
        """One token stream -> the conditioning, token axis first, template *not* stripped.

        `multipliers` are the emphasis multipliers of the weights this extension did not
        claim; a claimed weight has already left `1.0` behind.
        """
        raise NotImplementedError

    # ----------------------------------------------------------------- DiT side -- #

    def dit_problem(self, dit: Any) -> str | None:
        """Why this diffusion model is not the one the adapter knows, or `None` if it is.

        Duck-typed on what the hooks actually touch, and nothing else.  Every attribute in
        the check is a thing the framework may move, so one that is not load-bearing is
        pure liability — and a wrong answer here is a silent stand-down, this extension's
        worst failure mode (§HOW-IT-WORKS, *Model detection*).
        """
        raise NotImplementedError

    def block_sites(self, dit: Any) -> list[AttentionSite]:
        """One site per main block, in block order."""
        raise NotImplementedError

    def refiner_sites(self, dit: Any) -> list[AttentionSite]:
        """The text-only refiner blocks one stage earlier, in order."""
        raise NotImplementedError

    def wrap_attention(self, site: AttentionSite, original: Callable) -> Callable:
        """The replacement `forward` for `site.module`.

        Must park the rows for the duration of the call through `dit.active`, and route
        the logit bias it yields into the attention the way this model allows.
        """
        raise NotImplementedError

    def force_pytorch_attention(self, dit: Any, enable: bool):
        """Make the attention accept an additive mask while a bias is in play, and undo it.

        Sage and flash attention do not accept an arbitrary float mask, so the logit bias
        needs the plain SDPA path.  Only installed while a prompt actually asks for
        amplification: taking the optimised backend away from everyone who just wants a
        value flip would be a silent slowdown.
        """
        raise NotImplementedError


def detect(model: Any) -> ModelAdapter | None:
    """The adapter for the loaded diffusion engine, or `None` for a model nobody handles."""
    from lib_krea2_negpip.krea2 import Krea2Adapter
    from lib_krea2_negpip.moss import MossAdapter

    for adapter in (Krea2Adapter, MossAdapter):
        if adapter.matches(model):
            return adapter(model)

    return None


__all__ = ["AttentionSite", "ModelAdapter", "detect"]
