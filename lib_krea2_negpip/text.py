"""
Text-encoder side of NegPiP.

Forge's emphasis pass multiplies the encoder's hidden states by the prompt weight, so a
negative weight reaches the model as a sign-flipped *hidden state*.  On SD1/SDXL that is
close enough to NegPiP — the conditioning is fed straight to `to_k`/`to_v`, both linear —
but Krea 2 puts a two-stage text-fusion transformer between the conditioning and the
first attention layer (`backend/nn/krea.py`, `TextFusionTransformer`), and MOSS-Image a
projection and two text refiner blocks.  RMSNorm and SwiGLU are not odd functions, so a
negated hidden state is simply a different prompt, not an inverted one.

So a weight this extension claims is taken away from the emphasis pass entirely — it gets
a flat `1.0` for those segments — and is handed to the diffusion model out of band, as two
per-token rows applied inside attention after the text has been through those stages:

* a **value factor**, multiplied into the value vectors (`WeightConfig.value_factor`);
* a **logit bias**, added to the attention scores (`WeightConfig.logit_bias`).

A weight the config does not claim is left alone and reaches emphasis as itself, which is
Forge's own behaviour; a negative one reaches it as `1.0`, since the text stages would eat
the sign (§`_levers`).  See `dit.py` for the consuming end.

Neither engine applies that emphasis itself any more: since Forge Neo `21886f41`
(2026-09-30) Krea 2's `Qwen3VL4BEngine` has a read-only `EmphasisNone` and hands the prompt
to the encoder as literal text, parentheses included, and the MOSS-Image extension never
did otherwise.  So the encode is this module's, through the adapter: `Krea2Adapter.encode`
puts back what the engine did before that commit — the selected emphasis on the input
embeddings — and `MossAdapter.encode` applies the ComfyUI output lerp its encoder offers.
Either way an unclaimed weight still means something.

Both rows are built over the exact token stream the multipliers are built over, and sliced
at the exact same template offset, so they cannot drift apart.

There are two ways to build that stream.  Forge's own, until `21886f41`, was one templated
encode per weighted segment (`_tokenize_segmented`), which means a weight changes what the
model reads before any lever is applied.  The other is to rejoin the segments and encode once
(`_tokenize_single`), locating each fragment by character offset — the conditioning is
then identical to the same prompt written with no weights at all, and the weights do
nothing but drive the levers.  When no offsets can be had, `_tokenize_joined` tokenises
each segment on its own and splices the pieces into one copy of the template: not
byte-identical to the unweighted prompt at the segment seams, but free of the repeated
boilerplate, which is the damage that matters.  The split encode is only ever used when
one-pass encoding is switched off, and only on a model that ever had it (Krea 2).

Everything model-specific — which engine, which template, whether it is stripped from the
conditioning, how the stream is encoded — goes through a `ModelAdapter` (`adapters.py`).
"""

from functools import wraps
from typing import Any

import torch

from backend import memory_management
from backend.args import dynamic_args
from backend.text_processing import emphasis
from modules.processing import logger
from modules.shared import opts

from lib_krea2_negpip import NEGPIP_BIAS_KEY, NEGPIP_MASK_KEY, WeightConfig
from lib_krea2_negpip.adapters import ModelAdapter
from lib_krea2_negpip.prompts import weighted_segments

ORIGINAL_ATTR = "_krea2_negpip_original_conditioning"


def _as_token_id(token) -> int | None:
    try:
        return int(token)
    except TypeError:
        return None


def _levers(config: WeightConfig, weight: float) -> tuple[float, float, float]:
    """One parsed weight -> `(emphasis multiplier, value factor, logit bias)`.

    A segment is either the emphasis pass's or ours, never both — a claimed weight leaves
    `1.0` behind in the multipliers, so the magnitude is applied once, at the lever that
    can carry it.  A negative weight is never left to emphasis as a sign: a negated hidden
    state is a different prompt, not an inverted one, which is the whole reason this
    extension exists.  One is unclaimed only at `Value strength` 0, which is "negative
    weights do nothing", so it gets `1.0`.
    """
    factor = config.value_factor(weight)
    bias = config.logit_bias(weight)
    claimed = factor != 1.0 or bias != 0.0

    return (1.0 if claimed or weight < 0.0 else weight), factor, bias


def _neutral(tokens: list) -> tuple[list, list[float], list[float], list[float]]:
    """A stream no weight touches: the multipliers and factors `1.0`, the biases `0.0`."""
    return list(tokens), [1.0] * len(tokens), [1.0] * len(tokens), [0.0] * len(tokens)


def _tokenize_segmented(adapter: ModelAdapter, parsed: list[tuple[str, float]], config: WeightConfig) -> tuple[list, list[float], list[float], list[float]]:
    """The pre-`21886f41` `tokenize_line`, with the claimed weights split off.

    Forge's own shape until that commit: every weighted segment is tokenised in its own
    copy of the whole chat template, so the emphasis multiplier covers the boilerplate too.  Our rows are
    confined to the fragment (§`Krea2Adapter.segment_text_span`), but the extra template
    copies are still in the conditioning.  `_tokenize_single` is the way out of that; this
    stays as the fallback, and as the behaviour for anyone who leaves the option off.

    Krea 2 conditioning in Forge never carries images — `Krea2.get_learned_conditioning`
    calls the engine with the text only — so the image-placeholder branch of the original
    has no counterpart here.
    """
    tokenized = adapter.templated_ids([text for text, _ in parsed])

    tokens: list = []
    multipliers: list[float] = []
    factors: list[float] = []
    biases: list[float] = []

    for segment, (_, weight) in zip(tokenized, parsed):
        multiplier, factor, bias = _levers(config, weight)
        claimed = factor != 1.0 or bias != 0.0
        start, end = adapter.segment_text_span(segment) if claimed else (0, 0)

        for i, token in enumerate(segment):
            inside = start <= i < end
            tokens.append(token)
            multipliers.append(multiplier)
            factors.append(factor if inside else 1.0)
            biases.append(bias if inside else 0.0)

    return tokens, multipliers, factors, biases


def _token_owners(offsets: list[tuple[int, int]], bounds: list[tuple[int, int]], text: str) -> list[int | None]:
    """Which parsed segment each token belongs to, by majority of its characters.

    A token can straddle a segment boundary — `(blurry:-1.0), sharp` leaves `blurry` and
    `,` adjacent with nothing between them, and a BPE merge across that seam is ordinary.
    Whoever owns most of the token's text owns the token, counting the characters that
    are not whitespace first: Qwen's BPE carries the space *before* a word inside the
    word's token, so `, (a red apple:-1.0)` puts the space of ` a` in the unweighted
    segment and the `a` in the weighted one, and by plain character count the one-letter
    word would be a tie and go to the wrong side.  What is left tied goes to the earlier
    segment, so the rule is deterministic and never gives one token two weights.
    """
    owners: list[int | None] = []

    for start, end in offsets:
        best, score = None, (0, 0)

        for index, (low, high) in enumerate(bounds):
            lo, hi = max(start, low), min(end, high)
            shared = hi - lo
            if shared <= 0:
                continue

            candidate = (shared - sum(1 for c in text[lo:hi] if c.isspace()), shared)
            if candidate > score:
                best, score = index, candidate

        owners.append(best)

    return owners


def _reported_offsets(adapter: ModelAdapter, templated: str, tokens: list) -> list[tuple[int, int]] | None:
    """Character spans straight from the tokenizer, for the fast ones that report them."""
    reported = adapter.reported_offsets(templated)
    if reported is None:
        return None

    ids, offsets = reported

    try:
        #   the token stream still comes from the engine; this only has to agree with it
        if [int(token) for token in ids] != [int(token) for token in tokens]:
            return None
    except (TypeError, ValueError):
        return None

    return [tuple(span) for span in offsets]


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


def _tokenize_single(adapter: ModelAdapter, parsed: list[tuple[str, float]], config: WeightConfig):
    """The weighted prompt as **one** templated encode, or `None` if that isn't possible.

    `parse_prompt_attention` splits the prompt before the encoder ever sees it, and the
    engine then wraps each piece in the full chat template.  Rejoining the pieces and
    encoding once gives conditioning identical to the same prompt written without any
    weights at all — the weights stop perturbing what the model reads and only drive the
    attention levers, which is what they were supposed to do.

    Localisation is by character offsets: reported by the tokenizer where it can
    (`_reported_offsets`, which is MOSS-Image's fast tokenizer), decoded back out of the
    token stream where it cannot (`_decoded_offsets`, which on Forge Neo's pinned
    `transformers` is the Krea 2 case).  Both are checked against the templated token
    stream before they are used, and if neither can produce spans the caller falls back to
    `_tokenize_joined` — a real tested encoder rather than a guess.
    """
    clean = "".join(text for text, _ in parsed)
    stripped = clean.strip()
    lead = len(clean) - len(clean.lstrip())

    prefix, _, _ = adapter.template.partition("{}")
    tokens = adapter.templated_ids([clean])[0]
    templated = adapter.template.format(stripped)

    offsets = _reported_offsets(adapter, templated, tokens)
    if offsets is None:
        offsets = _decoded_offsets(adapter.hf_tokenizer, tokens, templated)
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

    for i, owner in enumerate(_token_owners(offsets, bounds, templated)):
        if owner is None:
            continue
        multipliers[i], factors[i], biases[i] = levers[owner]

    return tokens, multipliers, factors, biases


def _tokenize_joined(adapter: ModelAdapter, parsed: list[tuple[str, float]], config: WeightConfig) -> tuple[list, list[float], list[float], list[float]] | None:
    """Each segment tokenised on its own, spliced into **one** copy of the chat template.

    The fallback for `_tokenize_single` when neither offset source can place the
    fragments (after flyfront/sd-forge-negpip).  Every segment's tokens are exactly the
    segment's own, so the rows need no localisation at all — the stream is
    `prefix + seg_1 + ... + seg_n + suffix` by construction.

    What it gives up against `_tokenize_single` is byte-identity with the unweighted
    prompt: a BPE merge that would have crossed a segment seam cannot happen, so a token
    or two can differ at each seam.  The sentence is otherwise the sentence, and the
    template appears once.  The split encode would repeat the system instruction per
    segment, which is the damage that matters.

    The outer ends are stripped because the engine strips the whole prompt before
    formatting it.  Interior whitespace moves to the *start* of the segment that follows
    it (flyfront's fix): Qwen's BPE merges a leading space into the word — `red` inside a
    sentence is the single token `Ġred` — so a segment that ends in a space and one that
    starts without it would both tokenise differently from the whole prompt, and the
    weight would miss the space-carrying token.  Moved, the segment tokenises as it does
    in the sentence and the weight covers the whole word.
    """
    prefix, placeholder, suffix = adapter.template.partition("{}")
    if not placeholder:
        return None

    texts = [text for text, _ in parsed]
    if texts:
        texts[0] = texts[0].lstrip()
        texts[-1] = texts[-1].rstrip()

    for i in range(len(texts) - 1):
        kept = texts[i].rstrip()
        texts[i + 1] = texts[i][len(kept) :] + texts[i + 1]
        texts[i] = kept

    try:
        encoded = adapter.raw_ids([prefix, *texts, suffix])
    except (TypeError, ValueError, KeyError):
        return None
    if len(encoded) != len(texts) + 2:
        return None

    tokens: list = list(encoded[0])
    multipliers: list[float] = [1.0] * len(tokens)
    factors: list[float] = [1.0] * len(tokens)
    biases: list[float] = [0.0] * len(tokens)

    for segment, (_, weight) in zip(encoded[1:-1], parsed):
        multiplier, factor, bias = _levers(config, weight)
        tokens.extend(segment)
        multipliers.extend([multiplier] * len(segment))
        factors.extend([factor] * len(segment))
        biases.extend([bias] * len(segment))

    tail = list(encoded[-1])
    tokens.extend(tail)
    multipliers.extend([1.0] * len(tail))
    factors.extend([1.0] * len(tail))
    biases.extend([0.0] * len(tail))

    return tokens, multipliers, factors, biases


def _tokenize_line(adapter: ModelAdapter, line: str, config: WeightConfig, emphasis_name: str) -> tuple[list, list[float], list[float], list[float]]:
    """One prompt line -> the token stream plus the three rows indexed by it."""
    global _warned_single_pass, _warned_joined, _warned_split

    #   a line the model reads as data (a MOSS-Image layout) never meets the parser, whose
    #   square brackets would turn a bounding box into de-emphasis; it is the engine's own
    #   stream with neutral rows
    if adapter.is_verbatim(line):
        return _neutral(adapter.plain_ids(line))

    parsed = weighted_segments(line, emphasis_name)

    if all(weight == 1.0 for _, weight in parsed):
        #   an engine that can hand over its own ids for an unweighted line does, so the
        #   conditioning is its own to the byte; Krea 2 builds the stream like any other
        plain = adapter.plain_ids(line)
        if plain is not None:
            return _neutral(plain)

    single_pass = config.single_pass
    if not single_pass and not adapter.split_encode:
        #   the infotext records the option; a model that never had the split encode has
        #   nothing for "off" to mean, which is worth one line rather than a silent "on"
        if not _warned_split:
            _warned_split = True
            logger.info(f"NegPiP: {adapter.name} has no split encoding; the prompt is encoded in one pass regardless of the option")
        single_pass = True

    stream = None

    if single_pass:
        stream = _tokenize_single(adapter, parsed, config)

        if stream is None:
            stream = _tokenize_joined(adapter, parsed, config)
            if stream is not None and not _warned_joined:
                #   the infotext records the option, not which path produced the stream; a
                #   seam-level difference from the exact path is worth one line in the log
                _warned_joined = True
                logger.info("NegPiP: cannot locate the prompt fragments by character offset; tokenising the segments separately inside one chat template instead")

        if stream is None and not _warned_single_pass:
            #   the infotext records the option, not whether it engaged, so an unannounced
            #   fallback here reads afterwards as "one pass was on and made no difference"
            _warned_single_pass = True
            logger.warning("NegPiP: cannot tokenise the prompt in one pass; falling back to the split encoding for this prompt")

    if stream is None:
        stream = _tokenize_segmented(adapter, parsed, config)

    if adapter.max_length is not None:
        #   the engine truncates its own stream; the rows have to be cut where it cuts
        stream = tuple(row[: adapter.max_length] for row in stream)

    return stream


def _row(values: list[float], seq: int, default: float) -> list[float]:
    """One per-token row, cut or padded to the conditioning's own sequence length."""
    row = values[:seq]
    return row + [default] * (seq - len(row))


def _encode_line(adapter: ModelAdapter, line: str, config: WeightConfig, weighting: emphasis.Emphasis) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """One prompt line -> `(conditioning, value factors, logit biases, claimed tokens)`."""
    tokens, multipliers, factors, biases = _tokenize_line(adapter, line, config, weighting.name)

    global _warned_expanded

    z = adapter.encode(line, tokens, multipliers, weighting)  # token axis first, template on

    #   a token is not always one sequence position: `process_tokens` splices an embedding
    #   in over several, so a row indexed by *token* stops describing the conditioning.
    #   There is no realignment to attempt that would not duplicate that splice, and Forge
    #   guards the identical hazard by skipping emphasis outright when the lengths
    #   disagree.  Unreachable today — only the image path makes non-integer tokens, and
    #   that path is handed back to Forge whole — but silently misplacing a sign flip is
    #   not a failure worth leaving unguarded.
    aligned = int(z.shape[0]) == len(tokens)
    if not aligned and not _warned_expanded:
        _warned_expanded = True
        logger.warning("NegPiP: the encoder expanded the token stream, so the weights cannot be placed; leaving this prompt alone")

    template_end = adapter.template_end(tokens, int(z.shape[0]))
    z = z[template_end:]

    seq = int(z.shape[0])

    visible_factors = _row(factors[template_end:], seq, 1.0) if aligned else [1.0] * seq
    visible_biases = _row(biases[template_end:], seq, 0.0) if aligned else [0.0] * seq

    def column(values: list[float]) -> torch.Tensor:
        #   indexed 1:1 with the conditioning's leading axis, so Forge's own batching and
        #   scheduling code keeps the two together without any realignment of ours; the
        #   trailing singleton axes make it batch like the conditioning it rides with
        return torch.tensor(values, device=z.device, dtype=z.dtype).reshape(seq, *([1] * (z.ndim - 1)))

    count = sum(1 for factor, bias in zip(visible_factors, visible_biases) if factor != 1.0 or bias != 0.0)

    return z, column(visible_factors), column(visible_biases), count


_warned_hand_back = False
_warned_expanded = False
_warned_single_pass = False
_warned_joined = False
_warned_split = False


def _report(prompt, count: int):
    if count <= 0:
        return

    key = "Negative" if getattr(prompt, "is_negative_prompt", False) else "Positive"
    logger.info(f"NegPiP Enable ({key}: {count})")


def patch_text_encoder(adapter: ModelAdapter, config: WeightConfig):
    """Replace the engine's `get_learned_conditioning` with the row-producing version.

    The result is a `dict` of *lists* rather than stacked tensors: prompt scheduling
    (`[a:b:0.5]`) hands the engine several lines of different token counts in one call,
    and `prompt_parser.get_learned_conditioning` indexes whatever comes back per schedule
    entry — a list indexes fine where `torch.stack` would have raised.

    Which keys the dict carries follows the *config*, not this batch's prompts, so cond
    and uncond always agree on the shape even when only one of them uses a lever.  A
    config with neither lever returns the bare list, the shape the engine itself returns:
    that is the batch where only emphasis is needed.
    """
    model: Any = adapter.model

    if getattr(model, ORIGINAL_ATTR, None) is not None:
        return

    original = model.get_learned_conditioning

    @torch.inference_mode()
    @wraps(original)
    def negpip_get_learned_conditioning(prompt):
        global _warned_hand_back

        reason = adapter.hand_back(prompt)
        if reason is not None:
            #   hand the whole call back rather than half-apply it, and say so — silently
            #   dropping the weights is how this went unnoticed the last time the framework
            #   moved underneath the extension
            if not _warned_hand_back:
                _warned_hand_back = True
                logger.warning(reason)
            return original(prompt)

        memory_management.load_model_gpu(model.forge_objects.clip.patcher)
        adapter.before_encode(prompt)

        #   neither engine reads the setting any more; it is read here, as the engines did
        #   before, and recorded under the name Forge records it
        weighting = emphasis.get_current_option(opts.emphasis)()
        if any(emphasis.uses_emphasis(line) for line in prompt):
            dynamic_args.last_extra_generation_params["Emphasis"] = weighting.name

        conds: list[torch.Tensor] = []
        masks: list[torch.Tensor] = []
        biases: list[torch.Tensor] = []
        cache: dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]] = {}
        count = 0

        for line in prompt:
            encoded = cache.get(line)
            if encoded is None:
                encoded = _encode_line(adapter, line, config, weighting)
                cache[line] = encoded

            z, mask, bias, claimed = encoded
            conds.append(z)
            masks.append(mask)
            biases.append(bias)
            count += claimed

        _report(prompt, count)

        if not (config.uses_value or config.uses_bias):
            #   emphasis only: the engine's own shape, so the DiT and `compile_conditions`
            #   need no hooks at all
            return conds

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
