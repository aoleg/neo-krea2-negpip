# Neo Krea 2 NegPiP

A **Forge Neo** extension that lets you put negative prompting *inside* your prompt, for **Krea 2**.

Give a word a negative weight and the model subtracts it instead of emphasising it:

```text
a portrait photo, cinematic lighting, (blurry:-1.0), (plastic skin:-1.0)
```

The negative prompt field pushes the whole image away from a concept after the fact. A negative weight works in the opposite direction — the word is cancelled *where the model reads it*, in place, without touching anything else you asked for. In practice that means you can suppress something specific — a texture, a lighting style, a look you keep getting by accident — without the rest of the image drifting the way a heavy negative prompt makes it drift.

It works in either field:

- `(blurry:-1.0)` in the **positive** prompt removes the concept
- `(aqua hair:-1.0)` in the **negative** prompt *enforces* it instead — a double negative

> [!NOTE]
> **Krea 2 only.** For SD1, SDXL and Anima use [sd-forge-negpip](https://github.com/Haoming02/sd-forge-negpip). The two can be installed side by side; each stands down for models it does not handle.

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

## Quick start

1. Load a Krea 2 checkpoint.
2. Open the **NegPiP (Krea 2)** panel and tick it on.
3. Put a negative weight in either prompt field and generate.

The console confirms it took effect, with the number of words affected:

```text
NegPiP Enable (Positive: 3)
```

Nothing happens unless a prompt actually contains a negative weight, so it is safe to leave switched on all the time.

## What the numbers mean

With the default settings, the weight you type is the strength of the subtraction:

| Weight | What happens to that word |
| --- | --- |
| `(word:-2.0)` | Subtracted hard. Use when `-1.0` was not enough; can start pulling the image around it. |
| `(word:-1.0)` | The classic setting. The word's contribution is inverted — the model actively steers away from it. |
| `(word:-0.5)` | A gentle push away. Good when `-1.0` overcorrects into something else. |
| `(word:0)` † | The word is simply deleted from the image, without pushing away from it. Useful for a word the sentence needs in order to read naturally, but which you do not want rendered. |
| `(word:1.0)` | Normal. No effect. |
| `(word:2.0)` ‡ | The word contributes twice as much. Continues the same scale upwards: `1.5` is half again, `3.0` three times. |

† Weights from `0` up to `1` — including `(word:0)` itself — are only handled once **Handle de-emphasis too** is switched on. Left off, they fall through to Forge's ordinary de-emphasis, and a prompt containing nothing but those will not engage the extension at all.

‡ Weights above `1` are only handled once **Amplify emphasis in values** (or **Handle emphasis in attention**) is switched on. Left off, they fall through to Forge's ordinary emphasis, which Krea 2 largely normalises away.

Start at `-1.0`. If the thing is still there, go further negative; if the image starts looking distorted or fixated on the opposite, come back up.

Multi-word phrases work: `(plastic looking skin:-1.0)` affects all three words.

## Settings

### Value strength

**What it does:** scales every negative weight in the prompt at once. At `1.0` the weight you typed is used as-is, which is what the table above describes.

**What you'll see:** turning it up makes every negative weight bite harder without editing the prompt — the quickest way to test whether a prompt needs more suppression. Turning it down softens all of them together. At `0.0` negative weights do nothing.

Prefer editing individual weights once you know which word needs it; this is the global dial for when *everything* is too weak or too strong.

### Reference the prompt mean

**What it does:** decides what a negative weight subtracts the word *from*. At `0.0` — the way NegPiP has always worked — the word's contribution is flipped about zero. At `1.0` it is flipped about the average of the other words in your prompt.

**What you'll see:** at `0.0` the image keeps its fine texture but loses mid-scale contrast — the broad light-and-shade that makes a photo look modelled rather than flat. Flipping about zero subtracts a slice of "there is a prompt here" along with the word, the same slice whatever word you negated. Flipping about the prompt mean inverts only the part that identifies the word, and suppresses it exactly as hard. Measured on a fixed seed with one-pass encoding on, `0.0` holds 35-74% of the unweighted image's power across the middle of the frequency range where `1.0` holds 82-94%.

**Defaults to `1.0`.** Set it to `0.0` to reproduce images made before this setting existed, or to compare the two on a fixed seed. If you are chasing softening specifically, look at **Encode the prompt in one pass** first — it is much the bigger effect.

### Handle de-emphasis too

**What it does:** takes over ordinary de-emphasis — weights between `0` and `1`, like `(word:0.7)` — instead of leaving it to Forge's normal handling.

**What you'll see:** de-emphasis becomes considerably more effective. Forge's normal approach scales the text embedding, and Krea 2's text encoder largely normalises that back out, so `(word:0.7)` often does very little. Handled here, it genuinely fades the word out, on the same continuum as the table above — and `(word:0)` becomes a clean deletion.

**Off by default** because `(word:0.8)` is common in ordinary prompts and this noticeably changes what those prompts produce. Turn it on if de-emphasis has felt like it does nothing.

### Handle emphasis in attention

**What it does:** takes over emphasis — weights above `1`, like `(word:1.5)` — and applies it by making the image pay more attention to that word, rather than by scaling the text embedding.

**What you'll see:** emphasis that actually works. The usual kind runs into the same normalisation problem as de-emphasis above, which is why `(word:1.4)` on Krea 2 often looks much like `(word:1.0)`. Handled here, the emphasised word visibly takes over more of the image.

**Off by default,** and it does cost a little speed while a prompt is using it. Turn it on if emphasis has felt inert.

### Emphasis gain

**What it does:** how much a weight above `1` is worth, for the setting above.

**What you'll see:** the effect grows very quickly — going from `2.0` to `4.0` is far more than twice as strong, and it compounds through the model. If an emphasised word starts dominating the image, smearing, or crowding out everything else in the prompt, lower this before lowering your prompt weights. `1.0` is a good starting point if `2.0` feels wild.

### Amplify emphasis in values

**What it does:** takes over emphasis — weights above `1` — and applies it with the same lever a negative weight uses, continued upwards: `(word:2.0)` makes the word contribute twice as much to every part of the image that attends to it, `(word:-1.0)` makes it contribute the opposite, and `(word:1.0)` sits in between. **Value strength** scales this the same way it scales negative weights.

**What you'll see:** emphasis that works, in the same currency as suppression. Compared with **Handle emphasis in attention** above, this one is linear — `3.0` really is three times `1.0` — and it does not saturate, so large weights keep growing rather than levelling off, and it costs nothing in speed on any attention backend. The other one changes how much of the image's attention the word *captures* from the rest of the prompt, which is a different effect; try both on a fixed seed and keep the one that reads as emphasis to you. Switching both on at once is allowed but doubles up.

**Off by default,** for the same reason as the setting above.

### Encode the prompt in one pass

**What it does:** fixes something that happens to every weighted Krea 2 prompt, with or without this extension. Forge splits your prompt at each weight and sends the pieces to the text encoder separately — so `a portrait (blurry:-1.0) sharp` is read as three fragments rather than one sentence. This rejoins them, so the model reads exactly the prompt you would have written with no weights in it.

Worse than it sounds: each fragment gets its own copy of Krea 2's whole chat template, and only the first copy's system instruction is stripped again afterwards. A single `(watermark:-1.0)` on a short prompt therefore roughly quadruples the text the model reads, and most of the addition is the same boilerplate instruction repeated — so the words you actually care about end up with a fraction of the attention they had.

**What you'll see:** weighted prompts stop drifting away from their unweighted versions. If you have noticed that adding a weight to a prompt changes the image more than the weight itself should account for — different composition, a different mood, a general loss of contrast and fine detail, not just more or less of the weighted word — this is the largest single cause, and this is the fix. The weights then do nothing but the job you gave them.

**On by default.** Switch it off to reproduce images made before that changed.

If the tokenizer cannot say where each fragment landed, the extension tokenises the fragments separately and splices them into one copy of the template instead — the sentence is the same, only a token or two at each seam can differ from the unweighted prompt, and the console says so once. The old behaviour of falling back to the split encoding is gone; that encoding is now only used with this option off.

> [!IMPORTANT]
> If you used this option before and found it changed nothing, that was a bug, not a verdict. Krea 2's tokenizer cannot report the character offsets the option needed, so it stood down on every prompt while still recording itself as enabled — toggling it produced bit-identical images. It now works out the offsets itself, and logs a warning on the rare prompt where it still cannot. Any comparison you made before is void.

### Advanced

| Setting | Default | What it does |
| --- | --- | --- |
| Also act inside the text-fusion refiners | off | Applies the effect one stage earlier, before the image has read the text at all. Noticeably stronger and harder to steer — reach for it when a stubborn concept survives everything else. |
| First block / Last block | `0` / `27` | Which part of the model is affected, out of 28 stages. Narrowing the range weakens the effect and changes its character: as a rule of thumb the early stages shape composition and layout, the later ones texture and detail. Restricting to late blocks can remove a *look* while leaving the arrangement of the image alone. |
| Block stride | `1` | Apply in every Nth stage instead of all of them. Another way to soften the effect while keeping it spread across the whole model. |

All settings are saved into the image's generation parameters and paste back from them.

## If nothing seems to happen

- **Check the console.** No `NegPiP Enable` line means it never engaged.
- **Is the weight negative?** With the three opt-in settings off, only negative weights do anything. `(word:1.5)` alone will not trigger it.
- **Check Settings → *Emphasis*.** If it is set to `None`, Forge treats `(word:-1.0)` as literal text and there is no weight to act on. The extension logs a warning and stands down.
- **Is the checkpoint Krea 2?** It deliberately does nothing on other models.

## If a negative weight softens the whole image

Adding `(watermark:-1.0)` should change watermarks, not skin texture. If the image comes out flatter and less detailed than the same prompt without the weight, two settings above are the cause, and both now default to the corrected behaviour:

- **Encode the prompt in one pass** — on. The larger of the two by a distance. Off, a weight buries your prompt in repeated template boilerplate before any of this extension's machinery runs, and the same thing happens to a plain `(word:1.2)` with the extension disabled entirely.
- **Reference the prompt mean** — `1.0`. Stops the flip from subtracting part of the conditioning along with the word.

If you are reading this because an *old* image looked better, check its generation parameters: images made before these defaults changed carry `Krea2 NegPiP single pass: False` or no entry at all, and no `Krea2 NegPiP mean reference` line.

## Good to know

- **Hires. fix is covered.** Weights in the hires prompt count too, and the effect applies across both passes.
- **Re-encoding.** Switching the extension on, off, or changing any of its settings makes the next image re-encode its prompt. Repeat batches at unchanged settings are unaffected.
- **Nothing is modified in Forge itself.** Every change is undone when the extension stands down.

## How it works

See [HOW-IT-WORKS.md](HOW-IT-WORKS.md) for the implementation: why the existing Forge NegPiP port does not cover Krea 2, how the weights reach the model, and how it is tested.

## Credits

- [hako-mikan](https://github.com/hako-mikan/sd-webui-negpip) — NegPiP itself.
- [blue-pen5805](https://github.com/blue-pen5805/ComfyUI-krea2-negpip) — the Krea 2 ComfyUI node this is a port of.
- [Haoming02](https://github.com/Haoming02/sd-webui-forge-classic) — Forge Neo, and the [sd-forge-negpip](https://github.com/Haoming02/sd-forge-negpip) port that mapped out how NegPiP fits into it.
- [flyfront](https://github.com/flyfront/sd-forge-negpip) — the per-segment one-pass tokenizer this uses as a fallback, and the case for value-side emphasis.

## License

AGPL-3.0, inherited from the projects above. See [LICENSE](LICENSE).
