"""Offline check of the Krea 2 path against Forge Neo's real Krea 2 text engine.

Real: Qwen3VL4BEngine, SDClipModel, SDTokenizer, the Krea 2 slow tokenizer, the parser,
the emphasis classes, the script.  Fake: the Qwen3-VL transformer (a deterministic causal
stand-in) and the diffusion engine around the text engine.

The reference the MOSS-Image port must not disturb: everything here passed before the
adapter split and has to pass after it.
"""

import types

import forge_env  # noqa: F401  (bootstraps Forge; must come first)
from forge_env import FORGE, Prompt, check, finish, load_script

import torch
from transformers import Qwen2Tokenizer

from backend.text_processing import emphasis
from backend.text_processing.krea2_engine import Qwen3VL4BEngine
from modules.shared import opts

from lib_krea2_negpip import NEGPIP_MASK_KEY, WeightConfig
from lib_krea2_negpip import text as negpip_text
from lib_krea2_negpip.adapters import detect
from lib_krea2_negpip.krea2 import Krea2Adapter
from lib_krea2_negpip.prompts import scan

DIM = 2560
TAPS = [2, 5, 8, 11, 14, 17, 20, 23, 26, 29, 32, 35]


class FakeEmbedding(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(1))
        self.freq = torch.linspace(0.001, 0.05, DIM)

    def forward(self, ids):
        return torch.sin(ids.to(torch.float32)[..., None] * self.freq + 0.3)


class FakeQwen(torch.nn.Module):
    """Causal and non-linear, so a changed token or a scaled embedding shows up downstream."""

    num_layers = 36

    def __init__(self):
        super().__init__()
        self.embed = FakeEmbedding()

    def get_input_embeddings(self):
        return self.embed

    def forward(self, x, attention_mask=None, embeds=None, num_tokens=None, intermediate_output=None, final_layer_norm_intermediate=True, dtype=None, embeds_info=[]):
        assert x is None and isinstance(intermediate_output, list) and len(intermediate_output) == 12
        h = embeds.to(torch.float32)
        csum = torch.cumsum(h, dim=1) / torch.arange(1, h.shape[1] + 1).reshape(1, -1, 1)
        taps = [torch.tanh(h * (k + 1) * 0.1 + csum) for k in range(12)]
        return h, torch.stack(taps, dim=1)  # (b, 12, seq, dim)


tokenizer = Qwen2Tokenizer.from_pretrained(str(FORGE / "backend/huggingface/krea/Krea-2-Raw/tokenizer"))
engine = Qwen3VL4BEngine(text_encoder=FakeQwen(), tokenizer=tokenizer)
assert engine.text_encoder.layer == TAPS


class Krea2:
    """Named as Forge's engine is, which is what detection reads."""

    text_processing_engine_qwen = engine
    forge_objects = types.SimpleNamespace(clip=types.SimpleNamespace(patcher=None))
    ini_latent = None
    ref_latents = []

    def get_learned_conditioning(self, prompt):
        return engine(prompt)


model = Krea2()
adapter = detect(model)
check("detect: Krea2 -> Krea2Adapter", isinstance(adapter, Krea2Adapter) and adapter.engine_problem() is None)
check("detect: an unknown engine -> None", detect(types.SimpleNamespace()) is None)


def upstream(lines):
    return engine(list(lines))


def negpip(lines, config):
    negpip_text.unpatch_text_encoder(model)
    negpip_text.patch_text_encoder(adapter, config)
    try:
        return model.get_learned_conditioning(Prompt(lines))
    finally:
        negpip_text.unpatch_text_encoder(model)


one_pass = WeightConfig(single_pass=True)
split = WeightConfig(single_pass=False)

opts.emphasis = "Original"

# 1. an unweighted line is encoded exactly as the engine encodes it
plain = "a portrait of a woman posing on a beach, high quality analog photo"
for name, config in (("one pass", one_pass), ("split", split)):
    out = negpip([plain], config)
    check(f"unweighted == engine ({name})", torch.equal(out["crossattn"][0], upstream([plain])[0]))
    check(f"unweighted rows are neutral ({name})", bool((out[NEGPIP_MASK_KEY][0] == 1.0).all()))

# 2. one pass: a claimed weight leaves the conditioning of the unweighted sentence
weighted = "a photo of a cat, (watermark:-1.0), sharp focus"
bare = "a photo of a cat, watermark, sharp focus"
out = negpip([weighted], one_pass)
cond, mask = out["crossattn"][0], out[NEGPIP_MASK_KEY][0]
check("claimed weight: cond == engine(bare sentence)", torch.equal(cond, upstream([bare])[0]))
check("mask shape follows cond", mask.shape == (cond.shape[0], 1, 1))

ids = adapter.templated_ids([bare])[0]
end = adapter.template_end(ids, len(ids))
flipped = [i for i, v in enumerate(mask.flatten().tolist()) if v != 1.0]
text = tokenizer.decode([ids[end + i] for i in flipped])
check(f"flipped tokens are the word ({text!r})", text.strip() == "watermark" and set(mask.flatten()[flipped].tolist()) == {-1.0})

# 3. split encode still runs and keeps the rows aligned with the conditioning
out = negpip([weighted], split)
cond, mask = out["crossattn"][0], out[NEGPIP_MASK_KEY][0]
check("split: rows aligned", mask.shape[0] == cond.shape[0] and int((mask != 1.0).sum()) > 0)
check("split: cond differs from one pass (repeated template)", cond.shape[0] > upstream([bare])[0].shape[0])

# 4. an unclaimed weight goes through EmphasisOriginal on the input embeddings
emph = "a (red:1.5) room, blue sofa"
out = negpip([emph], one_pass)
tokens, multipliers, _, _ = negpip_text._tokenize_line(adapter, emph, one_pass, "Original")
emb, mask_, num, info = engine.text_encoder.process_tokens([tokens], torch.device("cpu"))
emb = emphasis.EmphasisOriginal()(emb, torch.asarray([multipliers]).to(emb))
_, z = engine.text_encoder.transformer(None, mask_, embeds=emb, num_tokens=num, intermediate_output=TAPS, embeds_info=info)
z = z[:, :, adapter.template_end(tokens, z.shape[2]):]
expected = z.permute(0, 2, 1, 3).reshape(-1, 12, DIM)
check("unclaimed weight: EmphasisOriginal applied", torch.equal(out["crossattn"][0], expected))
check("unclaimed weight: differs from no weight", not torch.equal(out["crossattn"][0], upstream(["a red room, blue sofa"])[0]))
check("multiplier lands on the word", sorted(set(multipliers)) == [1.0, 1.5])

# 5. Emphasis "None": literal text, nothing claimed, identical to the engine
opts.emphasis = "None"
out = negpip([weighted], one_pass)
check("emphasis None: == engine(literal)", torch.equal(out["crossattn"][0], upstream([weighted])[0]) and bool((out[NEGPIP_MASK_KEY][0] == 1.0).all()))

# 6. Emphasis "Ignore": weights dropped from emphasis, but NegPiP still claims negatives
opts.emphasis = "Ignore"
out = negpip([emph], one_pass)
check("emphasis Ignore: unclaimed weight is a no-op", torch.equal(out["crossattn"][0], upstream(["a red room, blue sofa"])[0]))
opts.emphasis = "Original"

# 7. scheduling: several lines of different lengths in one call
lines = [plain, weighted, plain]
out = negpip(lines, one_pass)
check("schedule: one entry per line", len(out["crossattn"]) == 3 and len(out[NEGPIP_MASK_KEY]) == 3)
check("schedule: cache reuse is exact", torch.equal(out["crossattn"][0], out["crossattn"][2]))

# 8. the patch is a true inverse
negpip_text.patch_text_encoder(adapter, one_pass)
negpip_text.unpatch_text_encoder(model)
check("unpatch restores", not getattr(model.get_learned_conditioning, "_negpip", False))

# 9. the joined fallback: segments tokenised apart, whitespace moved to the following segment
original_single = negpip_text._tokenize_single
negpip_text._tokenize_single = lambda *a, **k: None
try:
    seam = "a cat, (red:-1.0) hat"
    tokens, _, factors, _ = negpip_text._tokenize_line(adapter, seam, one_pass, "Original")
    check("joined: tokens == the unweighted sentence's", tokens == adapter.templated_ids(["a cat, red hat"])[0])
    word = tokenizer.decode([t for t, f in zip(tokens, factors) if f != 1.0])
    check(f"joined: the flagged tokens are the word ({word!r})", word == " red")
    broken = "a cat, (red:-1.0) hat BREAK a dog"
    tokens, _, _, _ = negpip_text._tokenize_line(adapter, broken, one_pass, "Original")
    check("joined: BREAK keeps its spaces", tokens == adapter.templated_ids(["a cat, red hat BREAK a dog"])[0])
finally:
    negpip_text._tokenize_single = original_single

# 10. activation on any weight, through the real script
script_module = load_script()
Krea2Model = type("Krea2", (Krea2,), {})
krea = Krea2Model()


def fake_p(prompts, negatives=("",)):
    return types.SimpleNamespace(sd_model=krea, prompts=list(prompts), negative_prompts=list(negatives), extra_generation_params={}, clear_prompt_cache=lambda: None)


ui = dict(enable=True, value_strength=1.0, mean_reference=1.0, handle_deemphasis=False, handle_emphasis=False, emphasis_gain=2.0, handle_value_emphasis=False, single_pass=True, patch_refiners=False, block_start=0, block_end=41, block_stride=1)
script = script_module.NegPiP()
krea_adapter = detect(krea)

check("scan: no weight", scan(fake_p([plain]), one_pass, krea_adapter) == (False, False, False))
check("scan: positive weight only", scan(fake_p([emph]), one_pass, krea_adapter) == (True, False, False))
check("scan: negative weight", scan(fake_p([weighted]), one_pass, krea_adapter) == (True, True, False))
check("scan: BREAK alone is not a weight", scan(fake_p(["a cat BREAK a dog"]), one_pass, krea_adapter)[0] is False)

check("resolve: unweighted stands down", script._resolve(fake_p([plain]), **ui) is None)

resolved = script._resolve(fake_p([emph]), **ui)
check("resolve: positive weight activates, DiT untouched", resolved is not None and resolved[1] is None and not (resolved[3].uses_value or resolved[3].uses_bias))

p = fake_p([emph])
script.process_batch(p, **ui)
out = krea.get_learned_conditioning(Prompt([emph]))
check("emphasis-only: bare list, Forge's shape", isinstance(out, list) and torch.equal(out[0], expected))
check("emphasis-only: infotext recorded", p.extra_generation_params.get("NegPiP single pass") is True)
script.postprocess(p, None)
check("emphasis-only: postprocess unpatches", not getattr(krea.get_learned_conditioning, "_negpip", False))

zero = dict(ui, value_strength=0.0)
resolved = script._resolve(fake_p(["a cat, (dog:-2.0)"]), **zero)
check("strength 0: activates emphasis-only", resolved is not None and resolved[1] is None)
script.process_batch(fake_p(["a cat, (dog:-2.0)"]), **zero)
out = krea.get_learned_conditioning(Prompt(["a cat, (dog:-2.0)"]))
check("strength 0: negative weight is neutral", torch.equal(out[0], upstream(["a cat, dog"])[0]))
script.postprocess(None, None)

broken = "a cat, (red:-1.0) hat BREAK a dog"
out = negpip([broken], one_pass)
check("BREAK keeps its spaces in one pass", torch.equal(out["crossattn"][0], upstream(["a cat, red hat BREAK a dog"])[0]))
out = negpip([broken], split)
check("BREAK still encodes in split mode", out["crossattn"][0].shape[0] == out[NEGPIP_MASK_KEY][0].shape[0])

opts.emphasis = "None"
check("emphasis None: stands down", script._resolve(fake_p([weighted]), **ui) is None)
opts.emphasis = "Original"

# 11. defaults and infotext paste-back, old keys and new
import gradio as gr  # noqa: E402

with gr.Blocks():
    components = script.ui(False)
names = ["enable", "value_strength", "mean_reference", "deemphasis", "emphasis", "gain", "value_emphasis", "single_pass", "refiners", "block_start", "block_end", "block_stride"]
by_name = dict(zip(names, components))
check("default: value emphasis on, attention off, gain 1.0", by_name["value_emphasis"].value is True and by_name["emphasis"].value is False and by_name["gain"].value == 1.0)
check("default: block range covers 42", by_name["block_end"].value == 41 and by_name["block_end"].maximum == 41)

fields = {id(c): key for c, key in script.infotext_fields}


def pasted(component, params):
    key = fields[id(by_name[component])]
    return key(params) if callable(key) else params.get(key)


old_image = {"Krea2 NegPiP value strength": "1.0", "Krea2 NegPiP single pass": "True"}
old_emphasis = dict(old_image, **{"Krea2 NegPiP emphasis": "True", "Krea2 NegPiP emphasis gain": "1.0"})
new_image = {"NegPiP value strength": "1.0", "NegPiP single pass": "True", "NegPiP emphasis": "True", "NegPiP emphasis gain": "1.0", "NegPiP last block": "27"}
plain_image = {"Steps": "12"}
check("paste: old NegPiP image without emphasis key -> off", pasted("emphasis", old_image) is False and pasted("enable", old_image) is True)
check("paste: old emphasis recorded -> True", pasted("emphasis", old_emphasis) == "True" and pasted("value_strength", old_emphasis) == "1.0")
check("paste: new keys read", pasted("emphasis", new_image) == "True" and pasted("block_end", new_image) == "27" and pasted("enable", new_image) is True)
check("paste: non-NegPiP image leaves the panel alone", pasted("emphasis", plain_image) is None and pasted("block_end", plain_image) is None and pasted("enable", plain_image) is False)
check("paste: absent block range -> defaults", pasted("block_start", old_image) == 0 and pasted("block_end", old_image) == 41 and pasted("block_stride", old_image) == 1)

# 12. the DiT side on a Krea 2-shaped module: leading text positions, mask substitution, SDPA swap
from backend import attention as forge_attention  # noqa: E402
from backend.nn import krea  # noqa: E402

from lib_krea2_negpip import NEGPIP_BIAS_KEY  # noqa: E402
from lib_krea2_negpip import dit as negpip_dit  # noqa: E402

KDIM, HEADS = 32, 4


class FakeAttention(torch.nn.Module):
    """`krea.Attention`'s surface: `wv`, and `forward(x, freqs, mask, transformer_options)` through `attention_function`."""

    def __init__(self):
        super().__init__()
        self.wq, self.wk, self.wv, self.wo = (torch.nn.Linear(KDIM, KDIM, bias=False) for _ in range(4))
        self.seen_masks = []

    def forward(self, x, freqs=None, mask=None, transformer_options={}):
        self.seen_masks.append(None if mask is None else mask.clone())
        q, k, v = (w(x).reshape(x.shape[0], x.shape[1], HEADS, KDIM // HEADS).transpose(1, 2) for w in (self.wq, self.wk, self.wv))
        out = krea.attention_function(q, k, v, HEADS, mask=mask, skip_reshape=True, transformer_options=transformer_options)
        return self.wo(out)


class FakeBlock(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = FakeAttention()

    def forward(self, x, freqs, mask, transformer_options):
        return x + self.attn(x, freqs, mask, transformer_options=transformer_options)


class FakeSingleStreamDiT(torch.nn.Module):
    """Text first, as `SingleStreamDiT.forward` concatenates `cat(text, image)`."""

    txtlayers, txtdim = 12, 2560

    def __init__(self):
        super().__init__()
        self.blocks = torch.nn.ModuleList([FakeBlock() for _ in range(3)])
        self.txtfusion = types.SimpleNamespace(refiner_blocks=torch.nn.ModuleList([FakeBlock()]))
        self.txtmlp = torch.nn.Identity()

    def forward(self, x, timesteps, context, attention_mask=None, transformer_options=None, **kwargs):
        text = context
        transformer_options = transformer_options or {}  # Forge always hands the DiT a dict
        for block in self.txtfusion.refiner_blocks:
            text = block(text, None, None, transformer_options)
        tokens = torch.cat((text, x), 1)
        for block in self.blocks:
            tokens = block(tokens, None, None, transformer_options)
        return tokens[:, text.shape[1] :]


kdit = FakeSingleStreamDiT().eval()
check("krea dit: detection passes", krea_adapter.dit_problem(kdit) is None)
check("krea dit: a foreign module is refused with a reason", "SingleStreamDiT" in (krea_adapter.dit_problem(torch.nn.Linear(2, 2)) or ""))

ktxt, kimg, kb = 5, 7, 2
kctx = torch.randn(kb, ktxt, KDIM)
kx = torch.randn(kb, kimg, KDIM)
krow = torch.ones(kb, ktxt, 1, 1)
krow[:, 2] = -1.0
kbias = torch.zeros(kb, ktxt, 1, 1)
kbias[:, 2] = 2.0

forge_attention.attention_function = forge_attention.attention_pytorch  # the CPU path; the swap below is what is tested
krea.attention_function = forge_attention.attention_function

with torch.no_grad():
    kplain = kdit(kx, None, kctx)

negpip_dit.patch_dit(krea_adapter, kdit, block_start=0, block_end=41, block_stride=1, patch_refiners=True, mean_reference=0.0, force_pytorch_attention=True)
attn0 = kdit.blocks[0].attn
wv_original = getattr(attn0.wv, negpip_dit.ORIGINAL_VALUE)
ky = torch.randn(kb, ktxt + kimg, KDIM)
setattr(attn0, negpip_dit.ACTIVE_ATTR, krow.reshape(kb, ktxt, 1))
with torch.no_grad():
    kv_hooked, kv_orig = attn0.wv(ky), wv_original(ky)
setattr(attn0, negpip_dit.ACTIVE_ATTR, None)
kexpected = kv_orig.clone()
kexpected[:, 2] = -kv_orig[:, 2]
check("krea dit: the flagged token is flipped at its *leading* position, nothing else", torch.equal(kv_hooked, kexpected))

with torch.no_grad():
    ksame = kdit(kx, None, kctx, transformer_options={}, **{NEGPIP_MASK_KEY: torch.ones_like(krow), NEGPIP_BIAS_KEY: torch.zeros_like(kbias)})
    kbiased = kdit(kx, None, kctx, transformer_options={}, **{NEGPIP_MASK_KEY: torch.ones_like(krow), NEGPIP_BIAS_KEY: kbias})
check("krea dit: neutral rows reproduce the unpatched output", torch.allclose(ksame, kplain, rtol=1e-5, atol=1e-6))
check("krea dit: the bias changes the output", not torch.allclose(kbiased, kplain))
masks = [m for m in attn0.seen_masks if m is not None]
kexpected_mask = torch.zeros(kb, 1, ktxt + kimg)
kexpected_mask[:, 0, 2] = 2.0
check("krea dit: the bias reaches the mask argument padded to [text, image]", masks and torch.equal(masks[-1], kexpected_mask))
rmasks = [m for m in kdit.txtfusion.refiner_blocks[0].attn.seen_masks if m is not None]
check("krea dit: refiners see it over the text alone", rmasks and rmasks[-1].shape == (kb, 1, ktxt) and float(rmasks[-1][0, 0, 2]) == 2.0)

negpip_dit.unpatch_dit(krea_adapter, kdit)
check("krea dit: attention_function restored", krea.attention_function is forge_attention.attention_pytorch and not hasattr(krea, negpip_dit.ORIGINAL_ATTENTION))
check("krea dit: unpatch leaves no forward behind", all("forward" not in vars(m) for m in [kdit, *(b.attn for b in kdit.blocks), *(b.attn.wv for b in kdit.blocks), kdit.txtfusion.refiner_blocks[0].attn]))

finish()
