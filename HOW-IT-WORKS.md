# How Neo Krea 2 NegPiP works

Implementation notes for anyone reading or modifying the code. Nothing here is needed to
use the extension — see the [README](README.md) for that.

Ported from [blue-pen5805/ComfyUI-krea2-negpip](https://github.com/blue-pen5805/ComfyUI-krea2-negpip),
itself a Krea 2 adaptation of [hako-mikan](https://github.com/hako-mikan/sd-webui-negpip)'s
original NegPiP.

## Why the existing Forge port does not cover Krea 2

[sd-forge-negpip](https://github.com/Haoming02/sd-forge-negpip) has three paths, and Krea 2
fits none of them. SD1 and SDXL encode the negative fragment separately, append it to the
cross-attention context, and negate the tail of `to_v`. Anima rides a mask through its
dedicated `SelfCrossAttention` module. Krea 2 has no cross-attention at all: it is
single-stream, so the text tokens sit in the *same* self-attention sequence as the image
patches (`backend/nn/krea.py`, `SingleStreamDiT.forward` concatenates them). And its
conditioning is not an embedding — it is a 12-layer × 2560-feature Qwen3-VL tap stack that
goes through a two-stage text-fusion transformer before any attention sees it.

## The two halves

**Weights out of band.** Forge's emphasis pass multiplies the encoder's hidden states by the
prompt weight, so a negative weight arrives as a sign-flipped *hidden state*. On SD1/SDXL
that is close enough — the conditioning feeds `to_k`/`to_v` directly, both linear. On
Krea 2 the RMSNorm and SwiGLU of the text-fusion stack are not odd functions, so a negated
tap stack is simply a *different prompt*, not an inverted one; and RMSNorm being
scale-invariant, a *scaled* one is barely scaled at all. So a claimed weight is taken away
from the emphasis pass — which sees a flat `1.0` for those segments — and travels alongside
the conditioning as two per-token rows: a **value factor** and a **logit bias**.

The ComfyUI node smuggles the same information through a sidecar token appended to the
conditioning tensor, with magic constants and a checksum, because ComfyUI's graph gives it
no other channel. Forge does have one: extra `model_conds` entries ride through
`reconstruct_cond_batch` → `compile_conditions` → `apply_model` and land as keyword
arguments on the diffusion model, batched and repeated in lockstep with the conditioning
they describe. So the sidecar is gone, and with it every way it could be mangled — and
because each row travels *with* its own conditioning, the positive and negative prompts
cannot read each other's, which is what lets weights work in both at any CFG.

**Act inside attention.** Both rows are applied in the DiT, after text fusion has run, over
the leading `txtlen` positions — exactly the text half of the concatenated sequence.

The value factor scales the output of each selected block's `wv`: attention still scores the
token normally, then subtracts or damps what it contributes instead of adding it. The logit
bias is added to the token's attention score, multiplying its softmax weight by `exp(bias)`
for every query in the sequence — scaling a value vector cannot make the rest of the image
attend to a word *more*, so amplification needs the other side of the softmax.

Two hooks do it: the attention module's `forward` parks the value factors for the duration
of one call and substitutes the bias for the `mask` argument (Krea 2 never passes one — the
DiT hands `None` to every block), and its `wv` linear scales the rows it names. Nothing of
Forge's own attention math is copied, so an upstream change to Krea 2's attention does not
silently break either lever. The bias does need a backend that accepts an additive mask, so
`krea.attention_function` is pointed at the plain SDPA path while one is in play, and put
back afterwards.

`txtfusion.layerwise_blocks` are deliberately left alone: they run at
`(batch * seq, taps, dim)`, so their sequence axis is the 12-layer tap stack, and the rows
do not index it.

## Getting the weights onto the right tokens

**`compile_conditions` learns a third shape.** It knows a bare tensor, or a dict with both
`crossattn` and a pooled `vector`. NegPiP conditioning is `crossattn` plus the rows and no
pooled vector, which would have raised `KeyError: 'vector'`. The rows are registered as
plain `Condition`s rather than `ConditionCrossAttn`, so mismatched lengths refuse to batch
instead of being repeated to a common length — a repeated row would describe the wrong
tokens. The wrapper is installed on demand, is a straight pass-through for every other
shape, and is never removed — Forge caches compiled conditioning on the
`StableDiffusionProcessing` *class*, so a cond can outlive the run that made it.

**The rows cover the words, not the boilerplate.** `Qwen3VLTextProcessingEngine.tokenize`
wraps *every* weighted segment in the full chat template, so a prompt carrying weights
tokenises to several copies of the template with the fragments spliced between them.
Emphasis scales all of it, which is Forge's own behaviour and is what an unclaimed weight
still gets — but flipping the sign of the system instruction is not what `(word:-1.0)`
asks for. Both rows are confined to the fragment itself, located structurally inside each
templated segment rather than at an assumed offset.

**Or the extra templates never happen at all.** *Encode the prompt in one pass* rejoins
the parser's segments and encodes that once, locating each fragment by the character
offsets a fast tokenizer reports and assigning every token to whichever segment covers
most of its characters — so a BPE merge across a segment seam lands in exactly one of
them. The token stream still comes from `engine.tokenize`; the offsets only have to agree
with it, and are checked against it before being used. There is deliberately no second
localisation heuristic: when offsets are unavailable or disagree, it falls back to the
per-segment path above, which is a real tested encoder rather than a guess at where a
fragment landed.

This is what makes the conditioning for `a portrait (blurry:-1.0) sharp` byte-identical to
`a portrait blurry sharp` — and it only holds because a claimed weight already leaves a
flat `1.0` behind in the emphasis multipliers.

## Model detection

`is_krea2_dit` duck-types on the identifying constants (`txtlayers`, `txtdim`) and on the
attributes the hooks actually wrap (`blocks`, `txtfusion.refiner_blocks`, `txtmlp`) — and
on nothing else. Every attribute in that check is a thing the framework may move, so one
that is not load-bearing is pure liability. This is not hypothetical: an earlier version
also tested for `SingleStreamDiT._unpack_context`, purely because it looked distinctive.
Forge moved that method into the text encoder, detection failed, and the extension stood
down **silently** — no error, no log line, the UI intact, and images pixel-identical to
having it switched off. Standing down is indistinguishable from a prompt with no weights
in it, which makes it this extension's worst failure mode by a distance.

## Testing

Everything is verified offline against Forge Neo's real sources, with no GPU, no checkpoint
and no running webui. Five tiers:

1. **Config** — the weight → lever mapping in isolation.
2. **Text** — a fake tokenizer behind the *real* `Qwen3VLTextProcessingEngine`, so
   `tokenize_line`, `strip_template` and the emphasis classes are Forge's own code.
3. **Model** — a real (not mocked) tiny `nn.Module` shaped like `SingleStreamDiT`, asserting
   byte-level that exactly the intended rows of `wv` change and by exactly the expected
   factor, and that unpatching is a true inverse down to the instance dictionary.
4. **Pipeline** — the real `prompt_parser` and `backend/sampling/condition.py` functions
   chained end to end, proving cond and uncond stay row-aligned once Forge's batching code
   has had them.
5. **Contract** — reads Forge's actual source files and asserts every precondition the port
   rests on: the attributes detection reads, the module names the hooks wrap, the forward
   signatures, the argument assumed to always be `None`, the tensor rank the conditioning
   arrives in, the template token ids. Tiers 1–4 test the extension against a *model* of the
   framework, and that model moves with our assumptions rather than Forge's — this is the
   only tier that fails when Forge changes rather than when we do.
