"""
NegPiP for Krea 2 and MOSS-Image — a Forge Neo extension.

Ported from blue-pen5805' ComfyUI node (https://github.com/blue-pen5805/ComfyUI-krea2-negpip),
which is itself a Krea 2 adaptation of hako-mikan's NegPiP.

The existing Forge extension (https://github.com/Haoming02/sd-forge-negpip) covers SD1,
SDXL and Anima.  None of those paths carry over: they either append a separately-encoded
negative fragment to the cross-attention context and negate the tail of `to_v`, or ride a
`c_negpip_mask` through a dedicated `SelfCrossAttention` module.  Krea 2 and MOSS-Image
have neither — both are single-stream, so the text tokens live in the same self-attention
sequence as the image patches, and both put non-linear text stages (Krea 2's text-fusion
transformer, MOSS-Image's text refiners) between the conditioning and the first attention.

So the port keeps the ComfyUI node's two halves and re-plumbs the middle:

* `lib_krea2_negpip/text.py` — take the claimed weights away from the emphasis pass and
  carry them out of band, as per-token value factors and logit biases alongside the
  conditioning.  The ComfyUI node smuggles the same information through a sidecar token
  appended to the conditioning tensor, because ComfyUI has no clean channel for it; Forge
  does, so this uses one, which is also what keeps cond and uncond from reading each
  other's rows.
* `lib_krea2_negpip/dit.py` — apply them inside the DiT's attention, after those text
  stages have run.

Everything that differs between the two models — where the engine is, how the prompt is
templated, where the text sits in the joint sequence, which linear carries the values,
how a mask reaches the attention — is behind one `ModelAdapter` (`lib_krea2_negpip/
adapters.py`), with one adapter per model (`krea2.py`, `moss.py`).

Which leaves `compile_conditions`, which only knows a bare tensor or a `crossattn` +
pooled `vector` dict; it gets taught the third shape.  See `dit.py`.
"""

from dataclasses import replace

import gradio as gr

from lib_krea2_negpip import MAX_GAIN, MAX_STRENGTH, WeightConfig
from lib_krea2_negpip.adapters import detect
from lib_krea2_negpip.dit import patch_dit, unpatch_dit
from lib_krea2_negpip.prompts import current_emphasis_name, reset_prompt_cache, scan
from lib_krea2_negpip.text import patch_text_encoder, unpatch_text_encoder

from modules import scripts
from modules.processing import logger

#   Krea 2 is 28 single-stream blocks, MOSS-Image 42; the sliders run to the larger and
#   `selected_blocks` clamps to the loaded model's real count, so the default means "all"
#   on either
DEFAULT_LAST_BLOCK = 41

#   infotext keys were `Krea2 NegPiP ...` while Krea 2 was the only model; images made then
#   paste back through the old name, new images are written with the model-neutral one
INFOTEXT_PREFIX = "NegPiP"
LEGACY_INFOTEXT_PREFIX = "Krea2 NegPiP"


def infotext_key(name: str) -> str:
    return f"{INFOTEXT_PREFIX} {name}"


def recorded(name: str, absent=None):
    """An infotext reader for one setting, under the current key or the legacy one.

    Returns `None` — leave the field alone — for an image with no NegPiP entries at all,
    and `absent` for a NegPiP image that did not record this setting: the infotext only
    records a setting that is on or off its default, and Forge leaves a field alone when
    its key is missing, so a pasted image would otherwise keep whatever the UI shows,
    which for "Handle emphasis in attention" (default changed to on) is the opposite of
    what an older image was made with.  A NegPiP image with the key missing means the
    setting was at the value it implies; an image with no NegPiP entries leaves the panel
    alone.
    """
    keys = (infotext_key(name), f"{LEGACY_INFOTEXT_PREFIX} {name}")
    marker = (infotext_key("value strength"), f"{LEGACY_INFOTEXT_PREFIX} value strength")

    def read(d):
        if not any(key in d for key in marker):
            return None
        for key in keys:
            if key in d:
                return d[key]
        return absent

    return read


class NegPiP(scripts.Script):
    sorting_priority = 15

    #   class attributes: `postprocess` is not guaranteed to see the same instance that
    #   `process_batch` ran on, and the patches outlive both
    active: bool = False
    dirty: bool = False
    """whether the conditioning cache may still hold NegPiP-shaped conds"""
    signature: tuple | None = None
    adapter = None
    dit = None
    warned_emphasis: bool = False
    warned_problem: str | None = None

    def title(self):
        return "NegPiP (Krea 2 / MOSS-Image)"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def ui(self, is_img2img):
        #   a plain Accordion, not an InputAccordion: the latter's open state *is* its value, so
        #   a `.../NegPiP (Krea 2 / MOSS-Image)/value: true` in ui-config.json — the way to have
        #   NegPiP on by default — would also force the panel open at every launch
        with gr.Accordion(label=self.title(), open=False, elem_id=self.elem_id("accordion")):
            enable = gr.Checkbox(False, label="Enable", elem_id=self.elem_id("enable"))

            gr.Markdown("Give a word a negative weight to suppress it: `(blurry:-1.0)` in the positive prompt, or in the negative prompt to enforce it instead.")

            value_strength = gr.Slider(
                minimum=0.0,
                maximum=MAX_STRENGTH,
                value=1.0,
                step=0.05,
                label="Value strength",
                info="how far the weight is taken; 1.0 makes (word:-1.0) a plain sign flip, 0.0 turns the value lever off and leaves every weight to emphasis",
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
                True,
                label="Handle emphasis in attention",
                info="make weights above 1 raise how much the image attends to the word, which survives the text-stage norms — scaling the embedding does not, so without this or the value lever a weight above 1 is close to inert on these models",
            )

            emphasis_gain = gr.Slider(
                minimum=0.0,
                maximum=MAX_GAIN,
                value=1.0,
                step=0.05,
                label="Emphasis gain",
                info="attention weight is multiplied by exp(gain x (weight - 1)), so this compounds quickly over blocks",
            )

            handle_value_emphasis = gr.Checkbox(
                False,
                label="Amplify emphasis in values",
                info="make weights above 1 scale what the word contributes, the same lever a negative weight uses, continued upwards: (word:2.0) contributes twice; linear inside attention, but the image's response flattens quickly; works on every attention backend; Value strength applies",
            )

            single_pass = gr.Checkbox(
                True,
                label="Encode the prompt in one pass",
                info="on Krea 2 a weighted prompt is otherwise encoded once per weighted segment, each in its own copy of the chat template — which buries the prompt in repeated boilerplate and is the single largest cause of quality loss from a weighted prompt; this rejoins them so the weights change nothing but the attention levers; MOSS-Image is always encoded in one pass",
            )

            with gr.Accordion("Advanced", open=False):
                patch_refiners = gr.Checkbox(
                    False,
                    label="Also act inside the text refiners",
                    info="stronger, and applies before the image tokens ever see the text: Krea 2's text-fusion refiners, MOSS-Image's two text blocks",
                )

                with gr.Row():
                    block_start = gr.Slider(minimum=0, maximum=DEFAULT_LAST_BLOCK, value=0, step=1, label="First block")
                    block_end = gr.Slider(minimum=0, maximum=DEFAULT_LAST_BLOCK, value=DEFAULT_LAST_BLOCK, step=1, label="Last block")

                block_stride = gr.Slider(minimum=1, maximum=16, value=1, step=1, label="Block stride", info="act in every Nth block of the range; the range is clamped to the model's block count, 28 on Krea 2 and 42 on MOSS-Image")

        self.infotext_fields = [
            (enable, lambda d: recorded("value strength")(d) is not None),
            (value_strength, recorded("value strength")),
            (mean_reference, recorded("mean reference")),
            (handle_deemphasis, recorded("de-emphasis", False)),
            (handle_emphasis, recorded("emphasis", False)),
            (emphasis_gain, recorded("emphasis gain")),
            (handle_value_emphasis, recorded("value emphasis", False)),
            (single_pass, recorded("single pass")),
            (patch_refiners, recorded("refiners", False)),
            (block_start, recorded("first block", 0)),
            (block_end, recorded("last block", DEFAULT_LAST_BLOCK)),
            (block_stride, recorded("block stride", 1)),
        ]

        return [enable, value_strength, mean_reference, handle_deemphasis, handle_emphasis, emphasis_gain, handle_value_emphasis, single_pass, patch_refiners, block_start, block_end, block_stride]

    # ============================================================================ #

    @classmethod
    def _teardown(cls):
        if cls.adapter is not None:
            unpatch_text_encoder(cls.adapter.model)
            unpatch_dit(cls.adapter, cls.dit)

        cls.adapter = None
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

    @classmethod
    def _stand_down(cls, adapter, problem: str):
        """Say why the model is not being hooked, once per reason.

        Standing down is indistinguishable from a prompt with no weights in it, which
        makes it this extension's worst failure mode; the log line is what makes it
        distinguishable.
        """
        if cls.warned_problem == problem:
            return

        cls.warned_problem = problem
        logger.warning(f"NegPiP stands down on {adapter.name}: {problem}")

    def _resolve(self, p, enable, value_strength, mean_reference, handle_deemphasis, handle_emphasis, emphasis_gain, handle_value_emphasis, single_pass, patch_refiners, block_start, block_end, block_stride):
        """UI arguments + this batch's prompts -> what to patch, or `None` to stand down."""
        if not enable:
            return None

        model = getattr(p, "sd_model", None)
        adapter = detect(model) if model is not None else None
        if adapter is None:
            return None

        problem = adapter.engine_problem()
        if problem is not None:
            self._stand_down(adapter, problem)
            return None

        config = WeightConfig(
            strength=min(MAX_STRENGTH, max(0.0, float(value_strength))),
            deemphasis=bool(handle_deemphasis),
            emphasis=bool(handle_emphasis),
            gain=min(MAX_GAIN, max(0.0, float(emphasis_gain))),
            value_emphasis=bool(handle_value_emphasis),
            single_pass=bool(single_pass),
        )

        if current_emphasis_name() == "None":
            self._warn_emphasis()
            return None

        #   any weight, not only a claimed one: since Forge Neo `21886f41` the engine hands
        #   Krea 2 weights to the encoder as literal text, and the MOSS-Image engine always
        #   did, so a batch the extension left alone would read `(word:1.2)` brackets and all
        weighted, claimed, biased = scan(p, config, adapter)
        if not weighted:
            return None

        options = {
            "block_start": int(block_start),
            "block_end": int(block_end),
            "block_stride": int(block_stride),
            "patch_refiners": bool(patch_refiners),
            "mean_reference": min(1.0, max(0.0, float(mean_reference))),
            "force_pytorch_attention": False,
        }

        if not claimed:
            #   emphasis only: the text hook with neither lever, which returns the engine's
            #   own conditioning shape, so the DiT is left untouched and the DiT options
            #   cannot change the image
            active = replace(config, strength=0.0, emphasis=False)
            return adapter, None, config, active, options, (id(model), None, active)

        #   the emphasis lever ticked with nothing above 1.0 in the prompt has to switch
        #   itself off, not merely produce an all-zero row: emitting the row at all means
        #   handing the attention backend a float mask, and the optimised ones do not
        #   take one.  The config the model sees is the one this batch actually needs.
        active = config if biased else replace(config, emphasis=False)

        try:
            dit = model.forge_objects.unet.model.diffusion_model
        except AttributeError:
            self._stand_down(adapter, "the engine has no forge_objects.unet.model.diffusion_model")
            return None

        problem = adapter.dit_problem(dit)
        if problem is not None:
            self._stand_down(adapter, problem)
            return None

        options["force_pytorch_attention"] = active.uses_bias

        signature = (id(model), id(dit), active, tuple(sorted(options.items())))

        return adapter, dit, config, active, options, signature

    def process_batch(self, p, enable, value_strength, mean_reference, handle_deemphasis, handle_emphasis, emphasis_gain, handle_value_emphasis, single_pass, patch_refiners, block_start, block_end, block_stride, *args, **kwargs):
        cls = NegPiP

        resolved = self._resolve(p, enable, value_strength, mean_reference, handle_deemphasis, handle_emphasis, emphasis_gain, handle_value_emphasis, single_pass, patch_refiners, block_start, block_end, block_stride)

        if resolved is None:
            cls._teardown()
            #   a cond cached while NegPiP was on is the wrong shape for a run with it
            #   off, and the cache key knows nothing about either
            cls._reset_cache(p)
            return

        #   `config` is what the UI asked for and is what the infotext records; `active`
        #   is what this batch's prompts actually need patching for
        adapter, dit, config, active, options, signature = resolved

        #   the two quality-critical ones are recorded unconditionally, default or not:
        #   both changed default at the same time as the softening was tracked down, and
        #   an image whose infotext omits them is an image nobody can place afterwards
        params = {
            infotext_key("value strength"): config.strength,
            infotext_key("mean reference"): options["mean_reference"],
            infotext_key("single pass"): config.single_pass,
        }
        if config.deemphasis:
            params[infotext_key("de-emphasis")] = True
        if config.emphasis:
            params[infotext_key("emphasis")] = True
            params[infotext_key("emphasis gain")] = config.gain
        if config.value_emphasis:
            params[infotext_key("value emphasis")] = True
        if options["patch_refiners"]:
            params[infotext_key("refiners")] = True
        if options["block_start"] != 0:
            params[infotext_key("first block")] = options["block_start"]
        if options["block_end"] != DEFAULT_LAST_BLOCK:
            params[infotext_key("last block")] = options["block_end"]
        if options["block_stride"] != 1:
            params[infotext_key("block stride")] = options["block_stride"]

        if cls.active and cls.signature == signature:
            #   same model, same settings: keep the patches and the cached conditioning,
            #   so a multi-iteration run encodes the prompt once
            p.extra_generation_params.update(params)
            return

        cls._teardown()

        patch_text_encoder(adapter, active)
        hooked = patch_dit(adapter, dit, **options) if dit is not None else 0

        cls.adapter = adapter
        cls.dit = dit
        cls.signature = signature
        cls.active = True
        cls.dirty = True

        reset_prompt_cache(p)
        p.extra_generation_params.update(params)

        if options["force_pytorch_attention"]:
            #   the one setting with a speed cost, said out loud: the optimised attention
            #   backend cannot take the additive mask, so every attention in the model runs
            #   on plain PyTorch SDPA for this batch
            logger.info(f"NegPiP: a weight above 1 with 'Handle emphasis in attention' on; {adapter.name} attention runs on PyTorch SDPA for this batch, which is slower than the optimised backend")
        logger.debug(f"NegPiP patched {hooked} attention modules on {adapter.name}")

    def postprocess(self, p, processed, *args):
        NegPiP._teardown()
