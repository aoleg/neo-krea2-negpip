"""Prompt inspection, shared by the activation check and the mask builder."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from modules.processing import StableDiffusionProcessing

from backend.text_processing import emphasis, parsing
from modules.shared import opts

from lib_krea2_negpip import WeightConfig

#   `parse_prompt_attention` marks a BREAK with the sentinel weight -1.  It is not a
#   weighted token and must never be read as one — sd-webui-negpip skips the same set.
SENTINELS = frozenset({"BREAK", "AND", "ADDCOL", "ADDROW"})

PROMPT_FIELDS = ("prompts", "negative_prompts", "hr_prompts", "hr_negative_prompts")


def current_emphasis_name() -> str:
    """The emphasis mode `text.py` will parse with on the next encode."""
    return emphasis.get_current_option(opts.emphasis).name


def weighted_segments(line: str, emphasis_name: str) -> list[tuple[str, float]]:
    """`(text, weight)` pairs, with the BREAK sentinel neutralised.

    The parser eats the whitespace around `BREAK`, so rejoining its segments for the
    one-pass encode would read `hat BREAK a dog` as `hatBREAKa dog`.  Krea 2 has no chunk
    boundary for it to mean, and Forge's own engine reads it as the literal word, so the
    sentinel goes back in as a spaced word.
    """
    segments = []
    for text, weight in parsing.parse_prompt_attention(line, emphasis_name):
        if text.strip() in SENTINELS:
            segments.append((f" {text.strip()} ", 1.0))
        else:
            segments.append((text, float(weight)))

    return segments


def scan(p: "StableDiffusionProcessing", config: WeightConfig) -> tuple[bool, bool, bool]:
    """`(any weight at all, any weight this config claims, any weight that needs a logit bias)`.

    All three come out of one parse over every prompt of the batch, the Hires. fix pass
    included.  The first decides whether the extension acts at all: since Forge Neo
    `21886f41` the engine reads Krea 2 weights as literal text, so any weight, claimed or
    not, needs this extension's encode.  The second decides whether the DiT needs its
    hooks.  The third decides whether the attention backend has to be forced to the one
    that accepts an additive mask, which is a cost worth avoiding when nothing in the
    prompt asks for amplification.
    """
    emphasis_name = current_emphasis_name()
    weighted = handled = biased = False

    for field in PROMPT_FIELDS:
        for line in getattr(p, field, None) or ():
            if not isinstance(line, str):
                continue

            for _, weight in weighted_segments(line, emphasis_name):
                weighted = weighted or weight != 1.0
                handled = handled or config.handles(weight)
                biased = biased or config.logit_bias(weight) != 0.0

                if handled and biased:
                    return True, True, True

    return weighted, handled, biased


def reset_prompt_cache(p: "StableDiffusionProcessing"):
    """
    Drop every cached conditioning.

    Forge keys its cond cache on the prompt text, the step count and the extra-network
    data — never on whether NegPiP was on.  Without this, toggling the extension while
    leaving the prompt alone hands the next run the previous run's conditioning, which
    is the wrong *shape*, not merely the wrong values.

    The caches live on the `StableDiffusionProcessing` class, so `clear_prompt_cache`
    (which rebinds both the instance and the class attribute) is the one that matters;
    the Hires. fix pair only exists on txt2img and is cleared in place, since `p` holds
    the very list object the class does.
    """
    if hasattr(p, "clear_prompt_cache"):
        p.clear_prompt_cache()

    for name in ("cached_hr_c", "cached_hr_uc"):
        cache = getattr(p, name, None)
        if isinstance(cache, list):
            for i in range(len(cache)):
                cache[i] = None
