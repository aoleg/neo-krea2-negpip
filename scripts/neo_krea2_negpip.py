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

* `lib_krea2_negpip/text.py` — take the claimed weights away from the emphasis pass and
  carry them out of band, as per-token value factors and logit biases alongside the
  conditioning.  The ComfyUI node smuggles the same information through a sidecar token
  appended to the conditioning tensor, because ComfyUI has no clean channel for it; Forge
  does, so this uses one, which is also what keeps cond and uncond from reading each
  other's rows.
* `lib_krea2_negpip/dit.py` — apply them inside the DiT's attention, after text fusion
  has run.

Which leaves `compile_conditions`, which only knows a bare tensor or a `crossattn` +
pooled `vector` dict; it gets taught the third shape.  See `dit.py`.
"""

from dataclasses import replace

import gradio as gr

from lib_krea2_negpip import KREA2_ENGINE, MAX_GAIN, MAX_STRENGTH, WeightConfig
from lib_krea2_negpip.dit import is_krea2_dit, patch_dit, unpatch_dit
from lib_krea2_negpip.prompts import current_emphasis_name, reset_prompt_cache, scan
from lib_krea2_negpip.text import patch_text_encoder, unpatch_text_encoder

from modules import scripts
from modules.processing import logger

#   Krea 2 is 28 single-stream blocks; the sliders are clamped to the real count anyway
DEFAULT_LAST_BLOCK = 27


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
        #   a plain Accordion, not an InputAccordion: the latter's open state *is* its value, so
        #   a `.../NegPiP (Krea 2)/value: true` in ui-config.json — the way to have NegPiP on by
        #   default — would also force the panel open at every launch
        with gr.Accordion(label=self.title(), open=False, elem_id=self.elem_id("accordion")):
            enable = gr.Checkbox(False, label="Enable", elem_id=self.elem_id("enable"))

            gr.Markdown("Give a word a negative weight to suppress it: `(blurry:-1.0)` in the positive prompt, or in the negative prompt to enforce it instead.")

            value_strength = gr.Slider(
                minimum=0.0,
                maximum=MAX_STRENGTH,
                value=1.0,
                step=0.05,
                label="Value strength",
                info="how far the weight is taken; 1.0 makes (word:-1.0) a plain sign flip, 0.0 is off",
            )

            mean_reference = gr.Slider(
                minimum=0.0,
                maximum=1.0,
                value=1.0,
                step=0.05,
                label="Reference the prompt mean",
                info="flip the word against the average of the other text tokens rather than against zero; 0.0 is NegPiP as originally written, which also subtracts a slice of the conditioning itself and softens the whole image",
            )

            handle_deemphasis = gr.Checkbox(
                False,
                label="Handle de-emphasis too",
                info="claim weights from 0 up to 1 as well, instead of leaving them to the ordinary emphasis pass; needed for (word:0) to delete a word outright",
            )

            handle_emphasis = gr.Checkbox(
                False,
                label="Handle emphasis in attention",
                info="make weights above 1 raise how much the image attends to the word, which survives the text-fusion norms — scaling the embedding largely does not",
            )

            emphasis_gain = gr.Slider(
                minimum=0.0,
                maximum=MAX_GAIN,
                value=2.0,
                step=0.05,
                label="Emphasis gain",
                info="attention weight is multiplied by exp(gain x (weight - 1)), so this compounds quickly over blocks",
            )

            single_pass = gr.Checkbox(
                True,
                label="Encode the prompt in one pass",
                info="a weighted prompt is otherwise encoded once per weighted segment, each in its own copy of the chat template — which buries the prompt in repeated boilerplate and is the single largest cause of quality loss from a weighted prompt; this rejoins them so the weights change nothing but the attention levers",
            )

            with gr.Accordion("Advanced", open=False):
                patch_txtfusion_refiners = gr.Checkbox(
                    False,
                    label="Also act inside the text-fusion refiners",
                    info="stronger, and applies before the image tokens ever see the text",
                )

                with gr.Row():
                    block_start = gr.Slider(minimum=0, maximum=DEFAULT_LAST_BLOCK, value=0, step=1, label="First block")
                    block_end = gr.Slider(minimum=0, maximum=DEFAULT_LAST_BLOCK, value=DEFAULT_LAST_BLOCK, step=1, label="Last block")

                block_stride = gr.Slider(minimum=1, maximum=16, value=1, step=1, label="Block stride", info="act in every Nth block of the range")

        self.infotext_fields = [
            (enable, lambda d: "Krea2 NegPiP value strength" in d),
            (value_strength, "Krea2 NegPiP value strength"),
            (mean_reference, "Krea2 NegPiP mean reference"),
            (handle_deemphasis, "Krea2 NegPiP de-emphasis"),
            (handle_emphasis, "Krea2 NegPiP emphasis"),
            (emphasis_gain, "Krea2 NegPiP emphasis gain"),
            (single_pass, "Krea2 NegPiP single pass"),
            (patch_txtfusion_refiners, "Krea2 NegPiP refiners"),
            (block_start, "Krea2 NegPiP first block"),
            (block_end, "Krea2 NegPiP last block"),
            (block_stride, "Krea2 NegPiP block stride"),
        ]

        return [enable, value_strength, mean_reference, handle_deemphasis, handle_emphasis, emphasis_gain, single_pass, patch_txtfusion_refiners, block_start, block_end, block_stride]

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

    def _resolve(self, p, enable, value_strength, mean_reference, handle_deemphasis, handle_emphasis, emphasis_gain, single_pass, patch_txtfusion_refiners, block_start, block_end, block_stride):
        """UI arguments + this batch's prompts -> what to patch, or `None` to stand down."""
        if not enable:
            return None

        model = getattr(p, "sd_model", None)
        if model is None or type(model).__name__ != KREA2_ENGINE:
            return None

        config = WeightConfig(
            strength=min(MAX_STRENGTH, max(0.0, float(value_strength))),
            deemphasis=bool(handle_deemphasis),
            emphasis=bool(handle_emphasis),
            gain=min(MAX_GAIN, max(0.0, float(emphasis_gain))),
            single_pass=bool(single_pass),
        )

        if not (config.uses_value or config.uses_bias):
            return None

        if current_emphasis_name() == "None":
            self._warn_emphasis()
            return None

        claimed, biased = scan(p, config)
        if not claimed:
            return None

        #   the emphasis lever ticked with nothing above 1.0 in the prompt has to switch
        #   itself off, not merely produce an all-zero row: emitting the row at all means
        #   handing the attention backend a float mask, and the optimised ones do not
        #   take one.  The config the model sees is the one this batch actually needs.
        active = config if biased else replace(config, emphasis=False)

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
            "mean_reference": min(1.0, max(0.0, float(mean_reference))),
            "force_pytorch_attention": active.uses_bias,
        }

        signature = (id(model), id(dit), active, tuple(sorted(options.items())))

        return model, dit, config, active, options, signature

    def process_batch(self, p, enable, value_strength, mean_reference, handle_deemphasis, handle_emphasis, emphasis_gain, single_pass, patch_txtfusion_refiners, block_start, block_end, block_stride, *args, **kwargs):
        cls = Krea2NegPiP

        resolved = self._resolve(p, enable, value_strength, mean_reference, handle_deemphasis, handle_emphasis, emphasis_gain, single_pass, patch_txtfusion_refiners, block_start, block_end, block_stride)

        if resolved is None:
            cls._teardown()
            #   a cond cached while NegPiP was on is the wrong shape for a run with it
            #   off, and the cache key knows nothing about either
            cls._reset_cache(p)
            return

        #   `config` is what the UI asked for and is what the infotext records; `active`
        #   is what this batch's prompts actually need patching for
        model, dit, config, active, options, signature = resolved

        #   the two quality-critical ones are recorded unconditionally, default or not:
        #   both changed default at the same time as the softening was tracked down, and
        #   an image whose infotext omits them is an image nobody can place afterwards
        params = {
            "Krea2 NegPiP value strength": config.strength,
            "Krea2 NegPiP mean reference": options["mean_reference"],
            "Krea2 NegPiP single pass": config.single_pass,
        }
        if config.deemphasis:
            params["Krea2 NegPiP de-emphasis"] = True
        if config.emphasis:
            params["Krea2 NegPiP emphasis"] = True
            params["Krea2 NegPiP emphasis gain"] = config.gain
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

        patch_text_encoder(model, active)
        hooked = patch_dit(dit, **options)

        cls.model = model
        cls.dit = dit
        cls.signature = signature
        cls.active = True
        cls.dirty = True

        reset_prompt_cache(p)
        p.extra_generation_params.update(params)

        logger.debug(f"NegPiP patched {hooked} attention modules{', SDPA forced' if options['force_pytorch_attention'] else ''}")

    def postprocess(self, p, processed, *args):
        Krea2NegPiP._teardown()
