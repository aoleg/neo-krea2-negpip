"""
NegPiP for Krea 2 — a Forge Neo extension.

Ported from blue-pen5805' ComfyUI node (https://github.com/blue-pen5805/ComfyUI-krea2-negpip),
which is itself a Krea 2 adaptation of hako-mikan's NegPiP.

The existing Forge extension (https://github.com/Haoming02/sd-forge-negpip) covers SD1,
SDXL and Anima.  None of those paths carry over: they either append a separately-encoded
negative fragment to the cross-attention context and negate the tail of `to_v`, or ride a
`c_negpip_mask` through a dedicated `SelfCrossAttention` module.  Krea 2 has neither — it
is single-stream, so the text tokens live in the same self-attention sequence as the image
patches, and its conditioning is a 12-layer Qwen3-VL tap stack that goes through a
non-linear text-fusion transformer before any attention sees it.

So the port keeps the ComfyUI node's two halves and re-plumbs the middle:

* `lib_krea2_negpip/text.py` — hand the emphasis pass `abs(weight)` and carry the sign
  out of band as a `+1 / -strength` mask alongside the conditioning.  The ComfyUI node
  smuggles the same information through a sidecar token appended to the conditioning
  tensor, because ComfyUI has no clean channel for it; Forge does, so this uses one.
* `lib_krea2_negpip/dit.py` — apply the mask to the value projection inside the DiT,
  after text fusion has run.

Which leaves `compile_conditions`, which only knows a bare tensor or a `crossattn` +
pooled `vector` dict; it gets taught the third shape.  See `dit.py`.
"""

import gradio as gr

from lib_krea2_negpip import KREA2_ENGINE
from lib_krea2_negpip.dit import is_krea2_dit, patch_dit, unpatch_dit
from lib_krea2_negpip.prompts import any_negative, current_emphasis_name, reset_prompt_cache
from lib_krea2_negpip.text import patch_text_encoder, unpatch_text_encoder

from modules import scripts
from modules.processing import logger
from modules.ui_components import InputAccordion

#   Krea 2 is 28 single-stream blocks; the sliders are clamped to the real count anyway
DEFAULT_LAST_BLOCK = 27

MAX_STRENGTH = 8.0


class Krea2NegPiP(scripts.Script):
    sorting_priority = 15

    #   class attributes: `postprocess` is not guaranteed to see the same instance that
    #   `process_batch` ran on, and the patches outlive both
    active: bool = False
    dirty: bool = False
    """whether the conditioning cache may still hold NegPiP-shaped conds"""
    signature: tuple | None = None
    model = None
    dit = None
    warned_emphasis: bool = False

    def title(self):
        return "NegPiP (Krea 2)"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def ui(self, is_img2img):
        with InputAccordion(False, label=self.title()) as enable:
            gr.Markdown("Give a word a negative weight to suppress it: `(blurry:-1.0)` in the positive prompt, or in the negative prompt to enforce it instead.")

            value_strength = gr.Slider(
                minimum=0.0,
                maximum=MAX_STRENGTH,
                value=1.0,
                step=0.05,
                label="Value strength",
                info="how hard the flagged tokens are subtracted; 1.0 is a plain sign flip, 0.0 is off",
            )

            with gr.Accordion("Advanced", open=False):
                patch_txtfusion_refiners = gr.Checkbox(
                    False,
                    label="Also flip inside the text-fusion refiners",
                    info="stronger, and applies before the image tokens ever see the text",
                )

                with gr.Row():
                    block_start = gr.Slider(minimum=0, maximum=DEFAULT_LAST_BLOCK, value=0, step=1, label="First block")
                    block_end = gr.Slider(minimum=0, maximum=DEFAULT_LAST_BLOCK, value=DEFAULT_LAST_BLOCK, step=1, label="Last block")

                block_stride = gr.Slider(minimum=1, maximum=16, value=1, step=1, label="Block stride", info="flip in every Nth block of the range")

        self.infotext_fields = [
            (enable, lambda d: "Krea2 NegPiP value strength" in d),
            (value_strength, "Krea2 NegPiP value strength"),
            (patch_txtfusion_refiners, "Krea2 NegPiP refiners"),
            (block_start, "Krea2 NegPiP first block"),
            (block_end, "Krea2 NegPiP last block"),
            (block_stride, "Krea2 NegPiP block stride"),
        ]

        return [enable, value_strength, patch_txtfusion_refiners, block_start, block_end, block_stride]

    # ============================================================================ #

    @classmethod
    def _teardown(cls):
        if cls.model is not None:
            unpatch_text_encoder(cls.model)
        if cls.dit is not None:
            unpatch_dit(cls.dit)

        cls.model = None
        cls.dit = None
        cls.signature = None
        cls.active = False

    @classmethod
    def _reset_cache(cls, p):
        if cls.dirty:
            reset_prompt_cache(p)
            cls.dirty = False

    @classmethod
    def _warn_emphasis(cls):
        if cls.warned_emphasis:
            return

        cls.warned_emphasis = True
        logger.warning('NegPiP needs prompt emphasis parsing; Emphasis is set to "None", so negative weights are read as literal text')

    def _resolve(self, p, enable, value_strength, patch_txtfusion_refiners, block_start, block_end, block_stride):
        """UI arguments + this batch's prompts -> what to patch, or `None` to stand down."""
        if not enable:
            return None

        model = getattr(p, "sd_model", None)
        if model is None or type(model).__name__ != KREA2_ENGINE:
            return None

        strength = min(MAX_STRENGTH, max(0.0, float(value_strength)))
        if strength == 0.0:
            return None

        if current_emphasis_name() == "None":
            self._warn_emphasis()
            return None

        if not any_negative(p):
            return None

        try:
            dit = model.forge_objects.unet.model.diffusion_model
        except AttributeError:
            return None

        if not is_krea2_dit(dit):
            return None

        options = {
            "block_start": int(block_start),
            "block_end": int(block_end),
            "block_stride": int(block_stride),
            "patch_txtfusion_refiners": bool(patch_txtfusion_refiners),
        }

        signature = (id(model), id(dit), strength, tuple(sorted(options.items())))

        return model, dit, strength, options, signature

    def process_batch(self, p, enable, value_strength, patch_txtfusion_refiners, block_start, block_end, block_stride, *args, **kwargs):
        cls = Krea2NegPiP

        resolved = self._resolve(p, enable, value_strength, patch_txtfusion_refiners, block_start, block_end, block_stride)

        if resolved is None:
            cls._teardown()
            #   a cond cached while NegPiP was on is the wrong shape for a run with it
            #   off, and the cache key knows nothing about either
            cls._reset_cache(p)
            return

        model, dit, strength, options, signature = resolved

        params = {"Krea2 NegPiP value strength": strength}
        if options["patch_txtfusion_refiners"]:
            params["Krea2 NegPiP refiners"] = True
        if options["block_start"] != 0:
            params["Krea2 NegPiP first block"] = options["block_start"]
        if options["block_end"] != DEFAULT_LAST_BLOCK:
            params["Krea2 NegPiP last block"] = options["block_end"]
        if options["block_stride"] != 1:
            params["Krea2 NegPiP block stride"] = options["block_stride"]

        if cls.active and cls.signature == signature:
            #   same model, same settings: keep the patches and the cached conditioning,
            #   so a multi-iteration run encodes the prompt once
            p.extra_generation_params.update(params)
            return

        cls._teardown()

        patch_text_encoder(model, strength)
        hooked = patch_dit(dit, **options)

        cls.model = model
        cls.dit = dit
        cls.signature = signature
        cls.active = True
        cls.dirty = True

        reset_prompt_cache(p)
        p.extra_generation_params.update(params)

        logger.debug(f"NegPiP patched {hooked} attention modules")

    def postprocess(self, p, processed, *args):
        Krea2NegPiP._teardown()
