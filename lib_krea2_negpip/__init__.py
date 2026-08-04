"""
NegPiP for Krea 2 — shared constants.

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

#   `type(shared.sd_model).__name__` for `backend/diffusion_engine/krea.py`
KREA2_ENGINE = "Krea2"

#   the Qwen3-VL tap stack Krea 2 conditioning carries: 12 layers x 2560 features,
#   flattened into the 30720-wide `crossattn` the DiT unpacks in `_unpack_context`
KREA2_TAP_LAYERS = 12
KREA2_TAP_DIM = 2560

#   `model_conds` keys -> extra kwargs on the diffusion-model forward
NEGPIP_MASK_KEY = "c_negpip_mask"
NEGPIP_BIAS_KEY = "c_negpip_bias"

#   where the DiT forward hook parks them for the attention hooks to find
NEGPIP_OPTION_KEY = "krea2_negpip_mask"
NEGPIP_BIAS_OPTION_KEY = "krea2_negpip_bias"

#   marks the `Attention` modules that are allowed to flip values
NEGPIP_ROLE_ATTR = "_krea2_negpip_role"

ROLE_BLOCK = "block"
ROLE_REFINER = "txtfusion_refiner"

MAX_STRENGTH = 8.0
MAX_GAIN = 8.0

#   a logit bias is exponential in the softmax; e^16 is already past anything useful and
#   the clamp is there to keep a wild prompt weight from producing an inf
MAX_BIAS = 16.0


@dataclass(frozen=True)
class WeightConfig:
    """How a prompt weight is turned into the two levers that survive text fusion.

    Forge's emphasis pass multiplies the Qwen3-VL hidden states, and `EmphasisOriginal`
    renormalises the whole chunk's mean afterwards, so it is a poor lever for Krea 2: the
    text-fusion transformer's `RMSNorm` is scale-invariant, which leaves only the residual
    branch carrying any of the magnitude, and the sign does not survive `SwiGLU` at all.

    Both replacements act inside attention instead, and each is a strict superset of doing
    nothing — a weight this config does not claim keeps going through Forge's emphasis
    exactly as it would without the extension installed.

    * `value_factor` scales the token's **value** vector.  Continuous: `1.0` is untouched,
      `0.0` removes the token's contribution, negative subtracts it.  `1 + strength*(w-1)`
      reproduces the old plain sign flip at the default `strength=1.0, w=-1.0`.
    * `logit_bias` adds a constant to the token's **attention logit**, which multiplies its
      softmax weight by `exp(bias)`.  Immune to every norm in the path, so this is the one
      that actually works for amplification.
    """

    strength: float = 1.0
    deemphasis: bool = False
    """also claim `0 <= w < 1`; off by default, since ordinary prompts use it freely"""
    emphasis: bool = False
    """claim `w > 1` and route it through `logit_bias` instead of Forge's emphasis"""
    gain: float = 2.0
    single_pass: bool = False
    """encode the prompt once instead of once per weighted segment; see `text.py`"""

    @property
    def uses_value(self) -> bool:
        return self.strength > 0.0

    @property
    def uses_bias(self) -> bool:
        return self.emphasis and self.gain > 0.0

    def value_factor(self, weight: float) -> float:
        if weight >= 1.0 or not self.uses_value:
            return 1.0
        if weight >= 0.0 and not self.deemphasis:
            return 1.0

        return min(1.0, max(-MAX_STRENGTH, 1.0 + self.strength * (weight - 1.0)))

    def logit_bias(self, weight: float) -> float:
        if weight <= 1.0 or not self.uses_bias:
            return 0.0

        return min(MAX_BIAS, self.gain * (weight - 1.0))

    def handles(self, weight: float) -> bool:
        """Whether this weight is taken over from Forge's emphasis pass."""
        return self.value_factor(weight) != 1.0 or self.logit_bias(weight) != 0.0


__all__ = [
    "KREA2_ENGINE",
    "KREA2_TAP_LAYERS",
    "KREA2_TAP_DIM",
    "MAX_BIAS",
    "MAX_GAIN",
    "MAX_STRENGTH",
    "NEGPIP_BIAS_KEY",
    "NEGPIP_BIAS_OPTION_KEY",
    "NEGPIP_MASK_KEY",
    "NEGPIP_OPTION_KEY",
    "NEGPIP_ROLE_ATTR",
    "ROLE_BLOCK",
    "ROLE_REFINER",
    "WeightConfig",
]
