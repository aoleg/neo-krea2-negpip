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

#   `type(shared.sd_model).__name__` for `backend/diffusion_engine/krea.py`
KREA2_ENGINE = "Krea2"

#   the Qwen3-VL tap stack Krea 2 conditioning carries: 12 layers x 2560 features,
#   flattened into the 30720-wide `crossattn` the DiT unpacks in `_unpack_context`
KREA2_TAP_LAYERS = 12
KREA2_TAP_DIM = 2560

#   `model_conds` key -> extra kwarg on the diffusion-model forward
NEGPIP_MASK_KEY = "c_negpip_mask"

#   where the DiT forward hook parks the mask for the attention hooks to find
NEGPIP_OPTION_KEY = "krea2_negpip_mask"

#   marks the `Attention` modules that are allowed to flip values
NEGPIP_ROLE_ATTR = "_krea2_negpip_role"

ROLE_BLOCK = "block"
ROLE_REFINER = "txtfusion_refiner"

__all__ = [
    "KREA2_ENGINE",
    "KREA2_TAP_LAYERS",
    "KREA2_TAP_DIM",
    "NEGPIP_MASK_KEY",
    "NEGPIP_OPTION_KEY",
    "NEGPIP_ROLE_ATTR",
    "ROLE_BLOCK",
    "ROLE_REFINER",
]
