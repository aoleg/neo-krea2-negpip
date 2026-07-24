"""Prompt inspection, shared by the activation check and the mask builder."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from modules.processing import StableDiffusionProcessing

from backend.text_processing import emphasis, parsing
from modules.shared import opts

#   `parse_prompt_attention` marks a BREAK with the sentinel weight -1.  It is not a
#   weighted token and must never be read as one — sd-webui-negpip skips the same set.
SENTINELS = frozenset({"BREAK", "AND", "ADDCOL", "ADDROW"})

PROMPT_FIELDS = ("prompts", "negative_prompts", "hr_prompts", "hr_negative_prompts")


def current_emphasis_name() -> str:
    """The emphasis mode the text engine will parse with on the next encode."""
    return emphasis.get_current_option(opts.emphasis).name


def weighted_segments(line: str, emphasis_name: str) -> list[tuple[str, float]]:
    """`(text, weight)` pairs, with the BREAK sentinel neutralised."""
    return [(text, 1.0 if text.strip() in SENTINELS else float(weight)) for text, weight in parsing.parse_prompt_attention(line, emphasis_name)]


def has_negative(line: str, emphasis_name: str) -> bool:
    return any(weight < 0.0 for _, weight in weighted_segments(line, emphasis_name))


def any_negative(p: "StableDiffusionProcessing") -> bool:
    """Whether any prompt of this batch — including the Hires. fix pass — asks for NegPiP."""
    emphasis_name = current_emphasis_name()

    for field in PROMPT_FIELDS:
        for line in getattr(p, field, None) or ():
            if isinstance(line, str) and has_negative(line, emphasis_name):
                return True

    return False


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
