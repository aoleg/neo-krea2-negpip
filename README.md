# Neo Krea 2 NegPiP

A **Forge Neo** extension that brings **NegPiP** — negative prompt in prompt — to **Krea 2**.

Give a word a negative weight and it gets *subtracted* instead of emphasised:

- `(blurry:-1.0)` in the **positive** prompt removes the concept
- `(aqua hair:-1.0)` in the **negative** prompt enforces it instead
- `(blurry:0)` removes the word's contribution outright, without inverting it

This is the thing a negative prompt cannot do on its own. A negative prompt only pushes the
prediction away at CFG time; NegPiP negates the token's *value* vector inside attention, so
the concept is cancelled where it is read rather than argued with afterwards.

> [!NOTE]
> **Krea 2 only.** For SD1, SDXL and Anima use
> [sd-forge-negpip](https://github.com/Haoming02/sd-forge-negpip) — none of its three paths
> apply to Krea 2, see [How it works](#how-it-works). The two extensions can be installed
> side by side; each stands down for models it does not handle.

## Install

Extensions → Install from URL:

```text
https://github.com/aoleg/neo-krea2-negpip
```

Or clone into your Forge Neo `extensions` directory and restart:

```bash
git clone https://github.com/aoleg/neo-krea2-negpip
```

No extra dependencies.

## Usage

Load a Krea 2 checkpoint, open the **NegPiP (Krea 2)** accordion, tick it on, and write
negative weights into either prompt field:

```text
a portrait photo, cinematic lighting, (blurry:-1.0), (low quality:-1.2)
```

The console prints how many tokens were flagged:

```text
NegPiP Enable (Positive: 3)
```

Nothing happens without a negative weight somewhere in the prompt — the extension checks
first and stays out of the way, so it is safe to leave ticked.

## Controls

| Control | Default | Description |
| --- | --- | --- |
| Value strength | `1.0` | How far a negative weight is taken. At `1.0` the value factor *is* the weight, so `(word:-1.0)` is a plain sign flip — the classic NegPiP — and `(word:0)` is a clean removal. Above `1.0` overshoots, below softens, `0.0` is off. |
| Handle de-emphasis too | off | Also claim weights between `0` and `1`, scaling the value vector instead of the embedding. Off by default because ordinary prompts use `(word:0.8)` freely and Forge already has a meaning for it. |
| Handle emphasis in attention | off | Claim weights above `1` and apply them as an attention bias rather than an embedding scale — see [Notes](#notes) for why the ordinary kind barely works here. |
| Emphasis gain | `2.0` | How much bias a weight above `1` is worth. Attention weight is multiplied by `exp(gain × (weight − 1))` in every hooked block, so this compounds fast; lower it before raising prompt weights. |

### Advanced

| Control | Default | Description |
| --- | --- | --- |
| Also act inside the text-fusion refiners | off | Applies one stage earlier, while the text is still being fused and before any image token has read it. Stronger, and harder to control. |
| First block / Last block | `0` / `27` | Which of Krea 2's 28 single-stream blocks are affected. Narrowing the range weakens the effect and localises it — early blocks lean towards composition, late blocks towards detail. |
| Block stride | `1` | Act in every Nth block of the range. Another way to dial the effect down. |

Everything is written to the infotext and pastes back from it.

## Notes

- **The weight is the dial.** At the default strength, a negative weight lands on the value
  vector unchanged: `(word:-2.0)` subtracts twice as hard as `(word:-1.0)`, and `(word:0)`
  merely deletes the word's contribution. A claimed weight is taken away from Forge's
  emphasis pass entirely, so the magnitude is applied once, not twice.
- **Scaling an embedding barely works on Krea 2.** Emphasis multiplies the Qwen3-VL hidden
  states, and the text-fusion transformer's `RMSNorm` is scale-invariant — most of the
  magnitude is normalised straight back out, and `EmphasisOriginal` then renormalises the
  whole chunk's mean on top. An attention bias is additive in a space nothing normalises,
  which is why *Handle emphasis in attention* exists. It costs the optimised attention
  backend (sage/flash do not take an additive mask), so it is only switched in when a prompt
  actually contains a weight above `1`.
- **Anything not claimed behaves exactly as it does without this extension.** With both
  opt-ins off, `(word:0.8)` and `(word:1.4)` go through Forge's emphasis untouched.
- **Emphasis must be on.** If the *Emphasis* setting is `None`, Forge never parses `(x:-1)`
  as a weight at all, so there is nothing to claim. The extension logs a warning and stands
  down.
- **Hires. fix is covered** — the hires prompts are inspected too, and the patch spans both
  passes.
- **Text conditioning is re-encoded** when the extension is switched on, off, or retuned,
  because Forge's conditioning cache is keyed on the prompt alone and knows nothing about
  NegPiP. Repeat batches at unchanged settings reuse the cache as normal.
- Nothing in `sd-webui-forge-classic` is modified; every patch is undone when the extension
  stands down.

## How it works

Ported from [blue-pen5805/ComfyUI-krea2-negpip](https://github.com/blue-pen5805/ComfyUI-krea2-negpip),
itself a Krea 2 adaptation of [hako-mikan](https://github.com/hako-mikan/sd-webui-negpip)'s
original NegPiP.

### Why the existing Forge port does not cover Krea 2

[sd-forge-negpip](https://github.com/Haoming02/sd-forge-negpip) has three paths, and Krea 2
fits none of them. SD1 and SDXL encode the negative fragment separately, append it to the
cross-attention context, and negate the tail of `to_v`. Anima rides a mask through its
dedicated `SelfCrossAttention` module. Krea 2 has no cross-attention at all: it is
single-stream, so the text tokens sit in the *same* self-attention sequence as the image
patches (`backend/nn/krea.py`, `SingleStreamDiT.forward` concatenates them). And its
conditioning is not an embedding — it is a 12-layer × 2560-feature Qwen3-VL tap stack that
goes through a two-stage text-fusion transformer before any attention sees it.

### The two halves

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

### Two smaller adjustments

- **`compile_conditions` learns a third shape.** It knows a bare tensor, or a dict with both
  `crossattn` and a pooled `vector`. NegPiP conditioning is `crossattn` plus the rows and no
  pooled vector, which would have raised `KeyError: 'vector'`. The rows are registered as
  plain `Condition`s rather than `ConditionCrossAttn`, so mismatched lengths refuse to batch
  instead of being repeated to a common length — a repeated row would describe the wrong
  tokens. The wrapper is installed on demand, is a straight pass-through for every other
  shape, and is never removed — Forge caches compiled conditioning on the
  `StableDiffusionProcessing` *class*, so a cond can outlive the run that made it.

- **The rows cover the words, not the boilerplate.** `Qwen3VLTextProcessingEngine.tokenize`
  wraps *every* weighted segment in the full chat template, so a prompt carrying weights
  tokenises to several copies of the template with the fragments spliced between them.
  Emphasis scales all of it, which is Forge's own behaviour and is what an unclaimed weight
  still gets — but flipping the sign of the system instruction is not what `(word:-1.0)`
  asks for. Both rows are confined to the fragment itself.

## Credits

- [hako-mikan](https://github.com/hako-mikan/sd-webui-negpip) — NegPiP itself.
- [blue-pen5805](https://github.com/blue-pen5805/ComfyUI-krea2-negpip) — the Krea 2 ComfyUI
  node this is a port of.
- [Haoming02](https://github.com/Haoming02/sd-webui-forge-classic) — Forge Neo, and the
  [sd-forge-negpip](https://github.com/Haoming02/sd-forge-negpip) port that mapped out how
  NegPiP fits into it.

## License

AGPL-3.0, inherited from the projects above. See [LICENSE](LICENSE).
