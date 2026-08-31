"""
Text-encoder side of Krea 2 NegPiP.

Forge's emphasis pass multiplies the encoder's hidden states by the prompt weight, so a
negative weight reaches the model as a sign-flipped *hidden state*.  On SD1/SDXL that is
close enough to NegPiP — the conditioning is fed straight to `to_k`/`to_v`, both linear —
but Krea 2 puts a two-stage text-fusion transformer between the conditioning and the
first attention layer (`backend/nn/krea.py`, `TextFusionTransformer`).  RMSNorm and SwiGLU
are not odd functions, so a negated Qwen3-VL tap stack is simply a different prompt, not
an inverted one.

So a weight this extension claims is taken away from the emphasis pass entirely — it gets
a flat `1.0` for those segments — and is handed to the diffusion model out of band, as two
per-token rows applied inside attention after text fusion has run:

* a **value factor**, multiplied into the value vectors (`WeightConfig.value_factor`);
* a **logit bias**, added to the attention scores (`WeightConfig.logit_bias`).

A weight the config does not claim is left alone and reaches emphasis as `abs(weight)`,
which is Forge's own behaviour minus the sign that text fusion would have eaten.  See
`dit.py` for the consuming end.

Both rows are built over the exact token stream the multipliers are built over, and sliced
at the exact same `strip_template` offset, so they cannot drift apart.

There are two ways to build that stream.  Forge's own is one templated encode per weighted
segment (`_tokenize_segmented`), which means a weight changes what the model reads before
any lever is applied.  The other is to rejoin the segments and encode once
(`_tokenize_single`), locating each fragment by character offset — the conditioning is
then identical to the same prompt written with no weights at all, and the weights do
nothing but drive the levers.  The second is opt-in, and falls back to the first whenever
the tokenizer cannot report offsets.
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

from lib_krea2_negpip import NEGPIP_BIAS_KEY, NEGPIP_MASK_KEY, WeightConfig
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
    is Forge's own behaviour, and a weight this extension does not claim keeps it — but
    scaling the system instruction is not what `(word:-1.0)` asks for.  Both rows are
    confined to the fragment: everything between `<|im_start|>user\\n` and the
    `<|im_end|>` that ends the user turn.

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


def _levers(config: WeightConfig, weight: float) -> tuple[float, float, float]:
    """One parsed weight -> `(emphasis multiplier, value factor, logit bias)`.

    A segment is either the emphasis pass's or ours, never both — a claimed weight leaves
    `1.0` behind in the multipliers, so the magnitude is applied once, at the lever that
    can carry it.  `abs`, not the signed weight, for anything left to emphasis: a negated
    hidden state is a different prompt, not an inverted one, which is the whole reason
    this extension exists.
    """
    factor = config.value_factor(weight)
    bias = config.logit_bias(weight)
    claimed = factor != 1.0 or bias != 0.0

    return (1.0 if claimed else abs(weight)), factor, bias


def _tokenize_segmented(engine: "Qwen3VLTextProcessingEngine", parsed: list[tuple[str, float]], config: WeightConfig) -> tuple[list, list[float], list[float], list[float]]:
    """`Qwen3VLTextProcessingEngine.tokenize_line`, with the claimed weights split off.

    Forge's own shape: every weighted segment is tokenised in its own copy of the whole
    chat template, so the emphasis multiplier covers the boilerplate too.  Our rows are
    confined to the fragment (§`_segment_text_span`), but the extra template copies are
    still in the conditioning.  `_tokenize_single` is the way out of that; this stays as
    the fallback, and as the behaviour for anyone who leaves the option off.

    Krea 2 conditioning in Forge never carries images — `Krea2.get_learned_conditioning`
    calls the engine with the text only — so the image-placeholder branch of the original
    has no counterpart here.
    """
    tokenized = engine.tokenize([text for text, _ in parsed])

    tokens: list = []
    multipliers: list[float] = []
    factors: list[float] = []
    biases: list[float] = []

    for segment, (_, weight) in zip(tokenized, parsed):
        multiplier, factor, bias = _levers(config, weight)
        claimed = factor != 1.0 or bias != 0.0
        start, end = _segment_text_span(engine, segment) if claimed else (0, 0)

        for i, token in enumerate(segment):
            inside = start <= i < end
            tokens.append(token)
            multipliers.append(multiplier)
            factors.append(factor if inside else 1.0)
            biases.append(bias if inside else 0.0)

    return tokens, multipliers, factors, biases


def _token_owners(offsets: list[tuple[int, int]], bounds: list[tuple[int, int]]) -> list[int | None]:
    """Which parsed segment each token belongs to, by majority of its characters.

    A token can straddle a segment boundary — `(blurry:-1.0), sharp` leaves `blurry` and
    `,` adjacent with nothing between them, and a BPE merge across that seam is ordinary.
    Whoever owns most of the token's text owns the token; ties go to the earlier segment,
    so the rule is deterministic and never gives one token two weights.
    """
    owners: list[int | None] = []

    for start, end in offsets:
        best, overlap = None, 0

        for index, (low, high) in enumerate(bounds):
            shared = min(end, high) - max(start, low)
            if shared > overlap:
                best, overlap = index, shared

        owners.append(best)

    return owners


def _reported_offsets(engine: "Qwen3VLTextProcessingEngine", templated: str, tokens: list) -> list[tuple[int, int]] | None:
    """Character spans straight from the tokenizer, for the fast ones that report them."""
    try:
        encoded = engine.tokenizer([templated], return_offsets_mapping=True)
    except (NotImplementedError, TypeError, ValueError):
        return None

    offsets = encoded.get("offset_mapping") if hasattr(encoded, "get") else None
    if not offsets:
        return None

    try:
        #   the token stream still comes from the engine; this only has to agree with it
        if [int(token) for token in encoded["input_ids"][0]] != [int(token) for token in tokens]:
            return None
    except (TypeError, ValueError):
        return None

    return [tuple(span) for span in offsets[0]]


def _decoded_offsets(tokenizer, tokens: list, templated: str) -> list[tuple[int, int]] | None:
    """Character spans worked out by decoding, for a tokenizer that cannot report them.

    Krea 2 needs this, and needs it on the default install.  The checkpoint ships
    `vocab.json` and `merges.txt` with no `tokenizer.json`, its `model_index.json` names
    `Qwen2Tokenizer`, and `backend/loader.py` instantiates that class by name — so on the
    `transformers` Forge Neo pins it is the *slow*, pure-Python tokenizer, which raises
    `NotImplementedError` on `return_offsets_mapping`.  With only the reported-offset path,
    single-pass encoding stood down on every Krea 2 prompt while still reporting itself as
    enabled in the infotext.

    Not a second guess at where a fragment landed: token `i` spans exactly what decoding
    one more token adds to the decoded prefix, and the whole result is thrown away unless
    decoding the full stream reproduces the templated prompt character for character.
    """
    ids: list[int] = []
    for token in tokens:
        value = _as_token_id(token)
        if value is None:
            return None  # an embedding occupies no characters of the prompt
        ids.append(value)

    def decode(seq: list[int]) -> str:
        return tokenizer.decode(seq, skip_special_tokens=False, clean_up_tokenization_spaces=False)

    try:
        if decode(ids) != templated:
            return None

        offsets: list[tuple[int, int]] = []
        cursor = 0
        for i in range(len(ids)):
            end = max(cursor, len(decode(ids[: i + 1])))
            offsets.append((cursor, end))
            cursor = end
    except (TypeError, ValueError, UnicodeDecodeError):
        return None

    return offsets if cursor == len(templated) else None


def _tokenize_single(engine: "Qwen3VLTextProcessingEngine", parsed: list[tuple[str, float]], config: WeightConfig):
    """The weighted prompt as **one** templated encode, or `None` if that isn't possible.

    `parse_prompt_attention` splits the prompt before the encoder ever sees it, and the
    engine then wraps each piece in the full chat template.  Rejoining the pieces and
    encoding once gives conditioning identical to the same prompt written without any
    weights at all — the weights stop perturbing what the model reads and only drive the
    attention levers, which is what they were supposed to do.

    Localisation is by character offsets: reported by the tokenizer where it can
    (`_reported_offsets`), decoded back out of the token stream where it cannot
    (`_decoded_offsets`), which on Forge Neo's pinned `transformers` is the Krea 2 case
    and therefore the one that matters.  Both are checked against `engine.tokenize` before
    they are used, and if neither can produce spans the caller falls back to
    `_tokenize_segmented` — a real tested encoder rather than a guess.
    """
    clean = "".join(text for text, _ in parsed)
    stripped = clean.strip()
    lead = len(clean) - len(clean.lstrip())

    prefix, _, _ = engine.llama_template.partition("{}")
    tokens = engine.tokenize([clean])[0]
    templated = engine.llama_template.format(stripped)

    offsets = _reported_offsets(engine, templated, tokens)
    if offsets is None:
        offsets = _decoded_offsets(engine.tokenizer, tokens, templated)
    if offsets is None or len(offsets) != len(tokens):
        return None

    bounds: list[tuple[int, int]] = []
    cursor = 0
    for text, _ in parsed:
        low = min(max(cursor - lead, 0), len(stripped))
        cursor += len(text)
        high = min(max(cursor - lead, 0), len(stripped))
        bounds.append((len(prefix) + low, len(prefix) + high))

    levers = [_levers(config, weight) for _, weight in parsed]

    multipliers = [1.0] * len(tokens)
    factors = [1.0] * len(tokens)
    biases = [0.0] * len(tokens)

    for i, owner in enumerate(_token_owners(offsets, bounds)):
        if owner is None:
            continue
        multipliers[i], factors[i], biases[i] = levers[owner]

    return tokens, multipliers, factors, biases


def _tokenize_line(engine: "Qwen3VLTextProcessingEngine", line: str, config: WeightConfig) -> tuple[list, list[float], list[float], list[float]]:
    """One prompt line -> the token stream plus the three rows indexed by it."""
    global _warned_single_pass

    parsed = weighted_segments(line, engine.emphasis.name)

    if config.single_pass:
        single = _tokenize_single(engine, parsed, config)
        if single is not None:
            return single

        #   the infotext records the option, not whether it engaged, so an unannounced
        #   fallback here reads afterwards as "one pass was on and made no difference"
        if not _warned_single_pass:
            _warned_single_pass = True
            logger.warning("NegPiP: cannot locate the prompt fragments in the token stream; falling back to the split encoding for this prompt")

    return _tokenize_segmented(engine, parsed, config)


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


def _row(values: list[float], seq: int, default: float) -> list[float]:
    """One per-token row, cut or padded to the conditioning's own sequence length."""
    row = values[:seq]
    return row + [default] * (seq - len(row))


def _encode_line(engine: "Qwen3VLTextProcessingEngine", line: str, config: WeightConfig) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """One prompt line -> `(conditioning, value factors, logit biases, claimed tokens)`."""
    tokens, multipliers, factors, biases = _tokenize_line(engine, line, config)

    global _warned_expanded

    z = engine.process_tokens([tokens], [multipliers])  # (1, taps, seq, dim)

    #   a token is not always one sequence position: `process_embeds` splices an embedding
    #   in over several, so a row indexed by *token* stops describing the conditioning.
    #   There is no realignment to attempt that would not duplicate that splice, and Forge
    #   guards the identical hazard by skipping emphasis outright when the lengths
    #   disagree.  Unreachable today — only the image path makes non-integer tokens, and
    #   that path is handed back to Forge whole — but silently misplacing a sign flip is
    #   not a failure worth leaving unguarded.
    aligned = int(z.shape[2]) == len(tokens)
    if not aligned and not _warned_expanded:
        _warned_expanded = True
        logger.warning("NegPiP: the encoder expanded the token stream, so the weights cannot be placed; leaving this prompt alone")

    template_end = _template_end(engine, tokens, z.shape[2])
    z = z[:, :, template_end:]

    batch, taps, seq, dim = z.shape

    #   `strip_template` flattens the tap axis into the features and `Qwen3VLTextProcessing
    #   Engine.__call__` unpacks it straight back out with the *token* axis leading, which
    #   is the shape `SingleStreamDiT.txtfusion` reads.  One line at a time, so `batch` is
    #   1 and the leading axis is the token count.
    z = z.permute(0, 2, 1, 3).reshape(batch * seq, taps, dim)

    visible_factors = _row(factors[template_end:], seq, 1.0) if aligned else [1.0] * seq
    visible_biases = _row(biases[template_end:], seq, 0.0) if aligned else [0.0] * seq

    def column(values: list[float]) -> torch.Tensor:
        #   indexed 1:1 with the conditioning's leading axis, so Forge's own batching and
        #   scheduling code keeps the two together without any realignment of ours
        return torch.tensor(values, device=z.device, dtype=z.dtype).reshape(batch * seq, 1, 1)

    count = sum(1 for factor, bias in zip(visible_factors, visible_biases) if factor != 1.0 or bias != 0.0)

    return z, column(visible_factors), column(visible_biases), count


_warned_reference = False
_warned_expanded = False
_warned_single_pass = False


def _reference_active(model: "Krea2", prompt) -> bool:
    """Whether `Krea2.get_learned_conditioning` would take its reference-image branch.

    Read-only — the original consumes `ini_latent` and clears `ref_latents` itself, and
    doing that here as well would eat the state before Forge ever sees it.
    """
    if getattr(prompt, "is_negative_prompt", False):
        return False
    if not getattr(opts, "krea2_do_reference", False):
        return False

    return bool(getattr(model, "ref_latents", None)) or getattr(model, "ini_latent", None) is not None


def _report(prompt, count: int):
    if count <= 0:
        return

    key = "Negative" if getattr(prompt, "is_negative_prompt", False) else "Positive"
    logger.info(f"NegPiP Enable ({key}: {count})")


def patch_text_encoder(model: "Krea2", config: WeightConfig):
    """Replace `Krea2.get_learned_conditioning` with the row-producing version.

    The result is a `dict` of *lists* rather than stacked tensors: prompt scheduling
    (`[a:b:0.5]`) hands the engine several lines of different token counts in one call,
    and `prompt_parser.get_learned_conditioning` indexes whatever comes back per schedule
    entry — a list indexes fine where `torch.stack` would have raised.

    Which keys the dict carries follows the *config*, not this batch's prompts, so cond
    and uncond always agree on the shape even when only one of them uses a lever.
    """
    if getattr(model, ORIGINAL_ATTR, None) is not None:
        return

    original = model.get_learned_conditioning
    engine: "Qwen3VLTextProcessingEngine" = model.text_processing_engine_qwen

    @torch.inference_mode()
    @wraps(original)
    def negpip_get_learned_conditioning(prompt):
        global _warned_reference

        if _reference_active(model, prompt):
            #   Krea 2 Edit encodes the prompt alongside reference images, on a path that
            #   carries no per-token rows.  Hand the whole call back rather than half-apply
            #   it, and say so — silently dropping the weights is how this went unnoticed
            #   the last time the framework moved underneath the extension.
            if not _warned_reference:
                _warned_reference = True
                logger.warning("NegPiP does not apply to the Krea 2 reference/Edit path; the positive prompt's weights are ignored there")
            return original(prompt)

        memory_management.load_model_gpu(model.forge_objects.clip.patcher)

        if not getattr(prompt, "is_negative_prompt", False):
            #   the bookkeeping `Krea2.get_learned_conditioning` does before encoding, and
            #   which this replaces: the img2img latent is consumed and the reference list
            #   dropped, so a stale one cannot put the DiT into edit mode
            if hasattr(model, "ini_latent"):
                model.ini_latent = None
            dynamic_args.ref_latents.clear()

        engine.emphasis = emphasis.get_current_option(opts.emphasis)()
        if any(emphasis.uses_emphasis(line) for line in prompt):
            dynamic_args.last_extra_generation_params["Emphasis"] = engine.emphasis.name

        conds: list[torch.Tensor] = []
        masks: list[torch.Tensor] = []
        biases: list[torch.Tensor] = []
        cache: dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]] = {}
        count = 0

        for line in prompt:
            encoded = cache.get(line)
            if encoded is None:
                encoded = _encode_line(engine, line, config)
                cache[line] = encoded

            z, mask, bias, claimed = encoded
            conds.append(z)
            masks.append(mask)
            biases.append(bias)
            count += claimed

        _report(prompt, count)

        result = {"crossattn": conds}
        if config.uses_value:
            result[NEGPIP_MASK_KEY] = masks
        if config.uses_bias:
            result[NEGPIP_BIAS_KEY] = biases

        return result

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
