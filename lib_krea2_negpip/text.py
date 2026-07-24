"""
Text-encoder side of Krea 2 NegPiP.

Forge's emphasis pass multiplies the encoder's hidden states by the prompt weight, so a
negative weight reaches the model as a sign-flipped *hidden state*.  On SD1/SDXL that is
close enough to NegPiP — the conditioning is fed straight to `to_k`/`to_v`, both linear —
but Krea 2 puts a two-stage text-fusion transformer between the conditioning and the
first attention layer (`backend/nn/krea.py`, `TextFusionTransformer`).  RMSNorm and SwiGLU
are not odd functions, so a negated Qwen3-VL tap stack is simply a different prompt, not
an inverted one.

So the sign is stripped here and carried out of band: the emphasis pass gets `abs(weight)`
— the magnitude NegPiP always intended — and the positions that were negative are handed
to the diffusion model as a `+1 / -strength` mask, to be applied to the *value* vectors
inside attention, after text fusion has run.  See `dit.py`.

The mask is built over the exact token stream the multipliers are built over, and sliced
at the exact same `strip_template` offset, so the two cannot drift apart.
"""

from functools import wraps
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from backend.diffusion_engine.krea import Krea2
    from backend.text_processing.qwen3vl_engine import Qwen3VLTextProcessingEngine

import torch

from backend import memory_management
from backend.args import dynamic_args
from backend.text_processing import emphasis
from modules.processing import logger
from modules.shared import opts

from lib_krea2_negpip import NEGPIP_MASK_KEY
from lib_krea2_negpip.prompts import weighted_segments

ORIGINAL_ATTR = "_krea2_negpip_original_conditioning"

#   `<|im_start|>user\n` — the three tokens `strip_template` steps over
ID_USER = 872
ID_NEWLINE = 198


def _as_token_id(token) -> int | None:
    try:
        return int(token)
    except TypeError:
        return None


def _segment_text_span(engine: "Qwen3VLTextProcessingEngine", segment: list) -> tuple[int, int]:
    """`[start, end)` of the prompt fragment inside one templated segment.

    `Qwen3VLTextProcessingEngine.tokenize` wraps *every* weighted segment in the whole
    chat template, so a prompt carrying weights tokenises to several copies of the
    template with the fragments spliced between them.  Emphasis scales all of it — that
    is Forge's own behaviour for positive weights and is left alone — but a sign flip on
    the system instruction is not what `(word:-1.0)` asks for.  The mask is confined to
    the fragment: everything between `<|im_start|>user\\n` and the `<|im_end|>` that ends
    the user turn.

    Falls back to the whole segment if the landmarks are not where the template puts
    them, which is also what happens for a prompt already written as raw chat markup.
    """
    starts = [i for i, token in enumerate(segment) if _as_token_id(token) == engine.id_template]
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


def _tokenize_line(engine: "Qwen3VLTextProcessingEngine", line: str) -> tuple[list, list[float], list[bool]]:
    """`Qwen3VLTextProcessingEngine.tokenize_line`, with the sign split off the weight.

    Krea 2 conditioning in Forge never carries images — `Krea2.get_learned_conditioning`
    calls the engine with the text only — so the image-placeholder branch of the original
    has no counterpart here.
    """
    parsed = weighted_segments(line, engine.emphasis.name)
    tokenized = engine.tokenize([text for text, _ in parsed])

    tokens: list = []
    multipliers: list[float] = []
    negatives: list[bool] = []

    for segment, (_, weight) in zip(tokenized, parsed):
        negative = weight < 0.0
        magnitude = abs(weight)
        start, end = _segment_text_span(engine, segment) if negative else (0, 0)

        for i, token in enumerate(segment):
            tokens.append(token)
            multipliers.append(magnitude)
            negatives.append(negative and start <= i < end)

    return tokens, multipliers, negatives


def _template_end(engine: "Qwen3VLTextProcessingEngine", tokens: list, seq_len: int) -> int:
    """The offset `Qwen3VLTextProcessingEngine.strip_template` slices at, computed alone.

    Kept as a separate copy rather than instrumenting `strip_template`, because the engine
    caches per prompt line and would not call it again for a repeat.
    """
    template_end = 0
    count_im_start = 0

    for i, token in enumerate(tokens):
        try:
            elem = int(token)
        except TypeError:
            continue

        if elem == engine.id_template and count_im_start < 2:
            template_end = i
            count_im_start += 1

    if seq_len > (template_end + 3):
        if int(tokens[template_end + 1]) == ID_USER and int(tokens[template_end + 2]) == ID_NEWLINE:
            template_end += 3

    return template_end


def _encode_line(engine: "Qwen3VLTextProcessingEngine", line: str, strength: float) -> tuple[torch.Tensor, torch.Tensor, int]:
    """One prompt line -> `(conditioning, negpip mask, negative token count)`."""
    tokens, multipliers, negatives = _tokenize_line(engine, line)

    z = engine.process_tokens([tokens], [multipliers])  # (1, taps, seq, dim)
    template_end = _template_end(engine, tokens, z.shape[2])
    z = z[:, :, template_end:]

    batch, taps, seq, dim = z.shape
    z = z.permute(0, 2, 1, 3).reshape(batch, seq, taps * dim)

    positions = [i for i, negative in enumerate(negatives[template_end : template_end + seq]) if negative]

    mask = torch.ones((batch, seq, 1), device=z.device, dtype=z.dtype)
    if positions:
        mask[:, positions, 0] = -strength

    return z, mask, len(positions)


def _report(prompt, count: int):
    if count <= 0:
        return

    key = "Negative" if getattr(prompt, "is_negative_prompt", False) else "Positive"
    logger.info(f"NegPiP Enable ({key}: {count})")


def patch_text_encoder(model: "Krea2", strength: float):
    """Replace `Krea2.get_learned_conditioning` with the mask-producing version.

    The result is a `dict` of *lists* rather than stacked tensors: prompt scheduling
    (`[a:b:0.5]`) hands the engine several lines of different token counts in one call,
    and `prompt_parser.get_learned_conditioning` indexes whatever comes back per schedule
    entry — a list indexes fine where `torch.stack` would have raised.
    """
    if getattr(model, ORIGINAL_ATTR, None) is not None:
        return

    original = model.get_learned_conditioning
    engine: "Qwen3VLTextProcessingEngine" = model.text_processing_engine_qwen

    @torch.inference_mode()
    @wraps(original)
    def negpip_get_learned_conditioning(prompt):
        memory_management.load_model_gpu(model.forge_objects.clip.patcher)

        engine.emphasis = emphasis.get_current_option(opts.emphasis)()
        if any(emphasis.uses_emphasis(line) for line in prompt):
            dynamic_args.last_extra_generation_params["Emphasis"] = engine.emphasis.name

        conds: list[torch.Tensor] = []
        masks: list[torch.Tensor] = []
        cache: dict[str, tuple[torch.Tensor, torch.Tensor, int]] = {}
        count = 0

        for line in prompt:
            encoded = cache.get(line)
            if encoded is None:
                encoded = _encode_line(engine, line, strength)
                cache[line] = encoded

            z, mask, negatives = encoded
            conds.append(z)
            masks.append(mask)
            count += negatives

        _report(prompt, count)

        return {"crossattn": conds, NEGPIP_MASK_KEY: masks}

    negpip_get_learned_conditioning._negpip = True

    setattr(model, ORIGINAL_ATTR, original)
    model.get_learned_conditioning = negpip_get_learned_conditioning


def unpatch_text_encoder(model):
    original = getattr(model, ORIGINAL_ATTR, None)
    if original is None:
        return

    delattr(model, ORIGINAL_ATTR)

    if not getattr(model.get_learned_conditioning, "_negpip", False):
        #   somebody else patched on top of us; theirs is the one in play, leave it alone
        return

    if getattr(original, "__func__", None) is getattr(type(model), "get_learned_conditioning", None):
        vars(model).pop("get_learned_conditioning", None)
    else:
        model.get_learned_conditioning = original
