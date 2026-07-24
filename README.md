# Neo Krea 2 NegPiP

A **Forge Neo** extension that brings **NegPiP** — negative prompt in prompt — to **Krea 2**.

Give a word a negative weight and it gets *subtracted* instead of emphasised:

- `(blurry:-1.0)` in the **positive** prompt removes the concept
- `(aqua hair:-1.0)` in the **negative** prompt enforces it instead

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
| Value strength | `1.0` | How hard the flagged tokens are subtracted. `1.0` is a plain sign flip — the classic NegPiP. Above that overshoots, below softens, `0.0` is off. |

### Advanced

| Control | Default | Description |
| --- | --- | --- |
| Also flip inside the text-fusion refiners | off | Applies the flip one stage earlier, while the text is still being fused and before any image token has read it. Stronger, and harder to control. |
| First block / Last block | `0` / `27` | Which of Krea 2's 28 single-stream blocks the flip is active in. Narrowing the range weakens the effect and localises it — early blocks lean towards composition, late blocks towards detail. |
| Block stride | `1` | Flip in every Nth block of the range. Another way to dial the effect down. |

Everything is written to the infotext and pastes back from it.

## Notes

- **Weights are still weights.** `(word:-1.2)` applies emphasis `1.2` *and* flips the sign;
  the magnitude means what it always meant. `(word:-1.0)` is a pure flip.
- **Emphasis must be on.** If the *Emphasis* setting is `None`, Forge never parses `(x:-1)`
  as a weight at all, so there is nothing to flip. The extension logs a warning and stands
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

**Sign out of band.** Forge's emphasis pass multiplies the encoder's hidden states by the
prompt weight, so a negative weight arrives as a sign-flipped *hidden state*. On SD1/SDXL
that is close enough — the conditioning feeds `to_k`/`to_v` directly, both linear. On
Krea 2 the RMSNorm and SwiGLU of the text-fusion stack are not odd functions, so a negated
tap stack is simply a *different prompt*, not an inverted one. So the emphasis pass is
handed `abs(weight)` — the magnitude NegPiP always meant — and the positions that were
negative travel alongside the conditioning as a `+1 / -strength` mask.

The ComfyUI node smuggles the same information through a sidecar token appended to the
conditioning tensor, with magic constants and a checksum, because ComfyUI's graph gives it
no other channel. Forge does have one: an extra `model_conds` entry rides through
`reconstruct_cond_batch` → `compile_conditions` → `apply_model` and lands as a keyword
argument on the diffusion model, batched and repeated in lockstep with the conditioning it
describes. So the sidecar is gone, and with it every way it could be mangled.

**Flip at the value projection.** The mask is applied inside the DiT, after text fusion has
run, to the output of each selected block's `wv` — over the leading `txtlen` rows, which is
exactly the text half of the concatenated sequence. Two hooks do it: the attention module's
`forward` parks the mask for the duration of one call (it is the only place that sees
`transformer_options`), and its `wv` linear scales the rows it names. Nothing of Forge's own
attention math is copied, so an upstream change to Krea 2's attention does not silently
break the flip.

`txtfusion.layerwise_blocks` are deliberately left alone: they run at
`(batch * seq, taps, dim)`, so their sequence axis is the 12-layer tap stack, and the mask
does not index it.

### Two smaller adjustments

- **`compile_conditions` learns a third shape.** It knows a bare tensor, or a dict with both
  `crossattn` and a pooled `vector`. NegPiP conditioning is `crossattn` plus the mask and no
  pooled vector, which would have raised `KeyError: 'vector'`. The wrapper is installed on
  demand, is a straight pass-through for every other shape, and is never removed — Forge
  caches compiled conditioning on the `StableDiffusionProcessing` *class*, so a cond can
  outlive the run that made it.

- **The mask covers the words, not the boilerplate.** `Qwen3VLTextProcessingEngine.tokenize`
  wraps *every* weighted segment in the full chat template, so a prompt carrying weights
  tokenises to several copies of the template with the fragments spliced between them.
  Emphasis scales all of it, which is Forge's own behaviour and is left alone — but flipping
  the sign of the system instruction is not what `(word:-1.0)` asks for. The mask is confined
  to the fragment itself.

The port was verified offline against Forge Neo's real `tokenize_line`, `strip_template`,
`prompt_parser` and `backend/sampling` code, with only the text encoder and the diffusion
model faked: the conditioning it emits is byte-identical to the engine's own, the mask
indexes the same positions after template stripping, and cond/uncond stay aligned once
batched together.

## Credits

- [hako-mikan](https://github.com/hako-mikan/sd-webui-negpip) — NegPiP itself.
- [blue-pen5805](https://github.com/blue-pen5805/ComfyUI-krea2-negpip) — the Krea 2 ComfyUI
  node this is a port of.
- [Haoming02](https://github.com/Haoming02/sd-webui-forge-classic) — Forge Neo, and the
  [sd-forge-negpip](https://github.com/Haoming02/sd-forge-negpip) port that mapped out how
  NegPiP fits into it.

## License

AGPL-3.0, inherited from the projects above. See [LICENSE](LICENSE).
