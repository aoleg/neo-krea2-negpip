"""Offline check of the MOSS-Image path against the real moss-image-forge-neo extension.

Real: the extension's `MossTextEngine` (Forge's `SDClipModel` and `SDTokenizer` inside it,
`layer="last"`), its `restore_layout_json`, its `MossImageDiT` with Comfy-Org's fused q/k/v at
a tiny size, its module-level `attention` and `kernels.attention`, Forge's parser and
emphasis classes, and the script.  Fake: the Qwen3.5 transformer (a deterministic causal
stand-in), the tokenizer's vocabulary (Qwen3-VL-4B's `tokenizer.json`, the same
`Qwen2TokenizerFast` class and BPE as the Qwen3.5 file the checkpoint ships, so offsets and
leading-space merges behave as they do there), and the diffusion engine around them.

What it proves: a layout prompt and an unweighted prompt are encoded byte-identically to the
extension on its own; a claimed weight leaves the conditioning of the unweighted sentence
and flags exactly the word's tokens, template tokens neutral; the rows land on the
*trailing* text positions of the joint sequence, in the value third of the fused projection
only; the logit bias reaches `attention_pytorch` padded to `[image, text]`; the uncond
block skip survives; unpatching is a true inverse; a tiny or foreign DiT stands the
extension down with a reason.

The extension is looked for at `FORGE/extensions/moss-image-forge-neo` or `MOSS_IMAGE_EXT`;
the fast tokenizer at `QWEN_FAST_TOKENIZER` or in the Hugging Face cache.
"""

import glob
import json
import os
import sys
import types
from pathlib import Path
from unittest import mock

import forge_env  # noqa: F401  (bootstraps Forge; must come first)
from forge_env import FORGE, Prompt, check, finish, load_script

import torch
from transformers import Qwen2TokenizerFast

from backend import attention
from backend.sampling import condition
from modules.shared import opts

from lib_krea2_negpip import NEGPIP_BIAS_KEY, NEGPIP_MASK_KEY, WeightConfig
from lib_krea2_negpip import dit as negpip_dit
from lib_krea2_negpip import moss as negpip_moss
from lib_krea2_negpip import text as negpip_text
from lib_krea2_negpip.adapters import detect
from lib_krea2_negpip.moss import MossAdapter, MossExtensionAdapter
from lib_krea2_negpip.prompts import scan

#   the tiny DiT runs on the CPU, where Forge's flash/sage pick fails and falls back with
#   an error line per call; this is the fallback it would take, chosen up front
attention.attention_function = attention.attention_pytorch

MOSS_EXT = Path(os.environ.get("MOSS_IMAGE_EXT", FORGE / "extensions" / "moss-image-forge-neo"))
if not (MOSS_EXT / "forge_moss" / "native" / "text_engine.py").exists():
    raise SystemExit(f"moss-image-forge-neo not found at {MOSS_EXT}; set MOSS_IMAGE_EXT")
sys.path.insert(0, str(MOSS_EXT))

TOKENIZER = os.environ.get("QWEN_FAST_TOKENIZER") or next(iter(sorted(glob.glob(os.path.expanduser("~/.cache/huggingface/hub/models--Qwen--Qwen3-VL-4B-Instruct/snapshots/*/tokenizer.json")))), None)
if TOKENIZER is None:
    raise SystemExit("no fast Qwen tokenizer found; set QWEN_FAST_TOKENIZER to a folder with tokenizer.json")

from forge_moss.native import settings, text_engine, transformer  # noqa: E402
from forge_moss.native.text_engine import MossTextEngine  # noqa: E402
from forge_moss.native.transformer import MossImageDiT  # noqa: E402

TEXT_DIM = 32
TINY = {"dim": 64, "ffn_dim": 96, "num_heads": 4, "num_layers": 4, "text_dim": TEXT_DIM, "freq_dim": 256, "latent_channels": 8, "rope_base": 10000.0}


# ------------------------------------------------------------------ text side -- #


class FakeEmbedding(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(1))
        self.freq = torch.linspace(0.001, 0.05, TEXT_DIM)

    def forward(self, ids):
        return torch.sin(ids.to(torch.float32)[..., None] * self.freq + 0.3)


class FakeQwen35(torch.nn.Module):
    """Causal and non-linear; returns the last hidden state, which `layer="last"` reads as `outputs[0]`."""

    num_layers = 32

    def __init__(self):
        super().__init__()
        self.embed = FakeEmbedding()

    def get_input_embeddings(self):
        return self.embed

    def forward(self, x, attention_mask=None, embeds=None, num_tokens=None, intermediate_output=None, final_layer_norm_intermediate=True, dtype=None, embeds_info=[]):
        assert x is None
        h = embeds.to(torch.float32)
        csum = torch.cumsum(h, dim=1) / torch.arange(1, h.shape[1] + 1).reshape(1, -1, 1)
        return torch.tanh(h + csum), None


tokenizer = Qwen2TokenizerFast.from_pretrained(str(Path(TOKENIZER).parent))
engine = MossTextEngine(text_encoder=FakeQwen35(), tokenizer=tokenizer)
check("real MossTextEngine built on Forge's SDClipModel", engine.text_encoder.layer == "last" and engine.tokenizer.tokenizer is tokenizer)

dit = MossImageDiT(**TINY, fused_qkv=True).eval()
generator = torch.Generator().manual_seed(0)
with torch.no_grad():
    for name, param in dit.named_parameters():
        param.copy_(torch.randn(param.shape, generator=generator) * (0.3 if "norm" in name else 0.1) + (1.0 if "norm" in name else 0.0))


class MossImageEngine:
    """Named as the extension's engine is, flagged as it flags itself."""

    is_moss_image = True
    text_engine_moss = engine
    forge_objects = types.SimpleNamespace(clip=types.SimpleNamespace(patcher=None), unet=types.SimpleNamespace(model=types.SimpleNamespace(diffusion_model=dit)))

    def get_learned_conditioning(self, prompt):
        return engine(prompt)


model = MossImageEngine()
adapter = detect(model)
check("detect: MossImageEngine -> MossExtensionAdapter", isinstance(adapter, MossExtensionAdapter))
check("detect: by flag alone", isinstance(detect(type("Other", (), {"is_moss_image": True, "text_engine_moss": engine})()), MossAdapter))
check("engine: no problem with the real text engine", adapter.engine_problem() is None)
check("engine: template and max length from the engine's module", adapter.template == text_engine.TEMPLATE and adapter.max_length == text_engine.MAX_LENGTH and adapter.max_length == 4096)
check("engine: a missing text engine is named", "text_engine_moss" in (MossExtensionAdapter(types.SimpleNamespace(is_moss_image=True)).engine_problem() or ""))


def negpip(lines, config):
    negpip_text.unpatch_text_encoder(model)
    negpip_text.patch_text_encoder(adapter, config)
    try:
        return model.get_learned_conditioning(Prompt(lines))
    finally:
        negpip_text.unpatch_text_encoder(model)


one_pass = WeightConfig(single_pass=True)
opts.emphasis = "Original"

# 1. unweighted and layout prompts: the extension's own encode, to the byte
plain = "a cute cat sitting on a sofa in a living room, a red apple on the table"
out = negpip([plain], one_pass)
check("unweighted == engine.encode(line)", torch.equal(out["crossattn"][0], engine.encode(plain)))
check("unweighted: conditioning is (T, dim), rows (T, 1)", out["crossattn"][0].ndim == 2 and out[NEGPIP_MASK_KEY][0].shape == (out["crossattn"][0].shape[0], 1))
check("unweighted rows are neutral", bool((out[NEGPIP_MASK_KEY][0] == 1.0).all()))
check("unweighted: the whole template is in the conditioning", out["crossattn"][0].shape[0] == len(engine.token_ids(plain)))

layout = json.dumps({"summary": {"meta": ["poster"], "canvas": [1024, 1024], "caption": "a vase and a teapot"}, "items": [{"label": "vase", "caption": "a blue vase", "bbox": [-400, -500, 100, -100]}, {"label": "teapot", "caption": "a red teapot", "bbox": [-400, 100, 100, 500]}], "bbox_coord_space": "abs_centralized", "bbox_format": "yxyx"})
stripped = layout.replace("{", "").replace("}", "")
check("layout: detected", adapter.is_verbatim(layout) and adapter.is_verbatim(f"  {layout}\n"))
check("layout: brace-stripped by sd-dynamic-prompts still detected", adapter.is_verbatim(stripped) and text_engine.restore_layout_json(stripped) == layout)
check("layout: an ordinary prompt is not one", not adapter.is_verbatim(plain) and not adapter.is_verbatim("(a red apple:-1.0) {red|blue} car"))
for name, line in (("layout", layout), ("stripped layout", stripped)):
    out = negpip([line], one_pass)
    check(f"{name}: encoded verbatim == engine.encode(line)", torch.equal(out["crossattn"][0], engine.encode(line)) and bool((out[NEGPIP_MASK_KEY][0] == 1.0).all()))
    check(f"{name}: bbox brackets are not de-emphasis", out["crossattn"][0].shape[0] == len(engine.token_ids(line)))

# 2. a claimed weight: the unweighted sentence's conditioning, the word's tokens flagged
weighted = "a cute cat sitting on a sofa in a living room, (a red apple:-1.0) on the table"
bare = "a cute cat sitting on a sofa in a living room, a red apple on the table"
out = negpip([weighted], one_pass)
cond, mask = out["crossattn"][0], out[NEGPIP_MASK_KEY][0]
check("claimed weight: cond == engine.encode(bare sentence)", torch.equal(cond, engine.encode(bare)))
check("claimed weight: rows 1:1 with the conditioning", mask.shape == (cond.shape[0], 1))
ids = engine.token_ids(bare)
flipped = [i for i, v in enumerate(mask.flatten().tolist()) if v != 1.0]
text = tokenizer.decode([ids[i] for i in flipped])
check(f"claimed weight: flipped tokens are the phrase ({text!r})", text == " a red apple" and set(mask.flatten()[flipped].tolist()) == {-1.0})
prefix_len = len(tokenizer.encode(text_engine.TEMPLATE.partition("{}")[0], add_special_tokens=False))
check("claimed weight: template tokens stay neutral", min(flipped) >= prefix_len and bool((mask[:prefix_len] == 1.0).all()) and bool((mask[-3:] == 1.0).all()))
check("one pass: the fast tokenizer reports offsets", adapter.reported_offsets(text_engine.TEMPLATE.format(bare)) is not None)

# 3. an unclaimed weight: the ComfyUI output lerp through the extension's own encoder
emph = "a cute cat sitting on a sofa in a living room, red theme, (blue theme:1.5)"
emph_bare = "a cute cat sitting on a sofa in a living room, red theme, blue theme"
out = negpip([emph], one_pass)
tokens, multipliers, _, _ = negpip_text._tokenize_line(adapter, emph, one_pass, "Original")
expected = engine.text_encoder.encode_token_weights([list(zip(tokens, multipliers))])[0][0]
check("unclaimed weight: encode_token_weights lerp applied", torch.equal(out["crossattn"][0], expected) and sorted(set(multipliers)) == [1.0, 1.5])
check("unclaimed weight: differs from no weight", not torch.equal(out["crossattn"][0], engine.encode(emph_bare)))
check("unclaimed weight: rows neutral", bool((out[NEGPIP_MASK_KEY][0] == 1.0).all()))
opts.emphasis = "Ignore"
out = negpip([emph], one_pass)
check("emphasis Ignore: unclaimed weight is a no-op", torch.equal(out["crossattn"][0], engine.encode(emph_bare)))
opts.emphasis = "Original"

# 4. the BREAK sentinel, and the option with no meaning here
broken = "a cat, (red:-1.0) hat BREAK a dog"
out = negpip([broken], one_pass)
check("BREAK keeps its spaces", torch.equal(out["crossattn"][0], engine.encode("a cat, red hat BREAK a dog")))
split = negpip([weighted], WeightConfig(single_pass=False))
check("single pass off: MOSS-Image is encoded in one pass regardless", torch.equal(split["crossattn"][0], cond) and torch.equal(split[NEGPIP_MASK_KEY][0], mask))

# 5. the joined fallback on a fast tokenizer: whitespace moves to the following segment
original_single = negpip_text._tokenize_single
negpip_text._tokenize_single = lambda *a, **k: None
try:
    tokens, _, factors, _ = negpip_text._tokenize_line(adapter, "a cat, (red:-1.0) hat", one_pass, "Original")
    check("joined: tokens == the unweighted sentence's", tokens == engine.token_ids("a cat, red hat"))
    check("joined: the flagged token is the space-carrying word", tokenizer.decode([t for t, f in zip(tokens, factors) if f != 1.0]) == " red")
finally:
    negpip_text._tokenize_single = original_single

# 6. several lines, the hand-back and the inverse
out = negpip([plain, weighted, plain], one_pass)
check("schedule: one entry per line, cache exact", len(out["crossattn"]) == 3 and torch.equal(out["crossattn"][0], out["crossattn"][2]))
negpip_text.patch_text_encoder(adapter, one_pass)
negpip_text.unpatch_text_encoder(model)
check("unpatch restores", not getattr(model.get_learned_conditioning, "_negpip", False) and "get_learned_conditioning" not in vars(model))

# ------------------------------------------------------------------- DiT side -- #

check("dit: the tiny model is refused with a reason naming the layers", adapter.dit_problem(dit) == "the DiT has 4 layers, not 42")
check("dit: a foreign model is refused", adapter.dit_problem(torch.nn.Linear(2, 2)) == "the DiT has no layers")

negpip_moss.MOSS_LAYERS, negpip_moss.MOSS_DIM = TINY["num_layers"], TINY["dim"]
check("dit: the real MossImageDiT passes once the shape constants match", adapter.dit_problem(dit) is None)

headless = MossImageDiT(**TINY, fused_qkv=True)
del headless.text_embedder.blocks
check("dit: missing refiners named", adapter.dit_problem(headless) == "the DiT has no text_embedder.blocks")

batch, height, width, txt = 2, 3, 5, len(engine.token_ids(bare))
imglen = height * width
generator = torch.Generator().manual_seed(1)
x = torch.randn(batch, TINY["latent_channels"], height, width, generator=generator)
context = torch.stack([engine.encode(bare)] * batch).float()
sigma = torch.full((batch,), 0.73)
assert context.shape == (batch, txt, TEXT_DIM)

flip_at = flipped[0]
mask_row = torch.ones(batch, txt, 1)
mask_row[:, flip_at, 0] = -1.0
neutral_row = torch.ones(batch, txt, 1)

with torch.no_grad():
    plain_out = dit(x, sigma, context)

options = dict(block_start=0, block_end=41, block_stride=1, patch_refiners=True, mean_reference=0.0, force_pytorch_attention=False)
hooked = negpip_dit.patch_dit(adapter, dit, **options)
check("patch: 4 blocks + 2 refiners hooked", hooked == 6)
check("patch: every block's fused qkv is wrapped, kernels.attention is not", all(getattr(layer.proj["qkv"].forward, "_negpip", False) for layer in dit.layers) and not getattr(transformer.kernels.attention, "_negpip", False))

# 7. the value lever: the trailing text rows of the value third, and nothing else
layer = dit.layers[0]
qkv = layer.proj["qkv"]
original_qkv = getattr(qkv, negpip_dit.ORIGINAL_VALUE)
y = torch.randn(batch, imglen + txt, TINY["dim"], generator=generator)
setattr(layer, negpip_dit.ACTIVE_ATTR, mask_row)
with torch.no_grad():
    hooked_out, original_out = qkv(y), original_qkv(y)
setattr(layer, negpip_dit.ACTIVE_ATTR, None)
dim = TINY["dim"]
q_k_same = torch.equal(hooked_out[..., : 2 * dim], original_out[..., : 2 * dim])
v_hooked, v_orig = hooked_out[..., 2 * dim :], original_out[..., 2 * dim :]
expected_v = v_orig.clone()
expected_v[:, imglen + flip_at, :] = -v_orig[:, imglen + flip_at, :]
check("value lever: only the value third changes", q_k_same)
check("value lever: exactly the flagged token, at its trailing position", torch.equal(v_hooked, expected_v))
check("value lever: the image rows are untouched", torch.equal(v_hooked[:, :imglen], v_orig[:, :imglen]))

refiner = dit.text_embedder.blocks[0]
original_rqkv = getattr(refiner.qkv, negpip_dit.ORIGINAL_VALUE)
yt = torch.randn(batch, txt, dim, generator=generator)
setattr(refiner, negpip_dit.ACTIVE_ATTR, mask_row)
with torch.no_grad():
    r_hooked, r_orig = refiner.qkv(yt), original_rqkv(yt)
setattr(refiner, negpip_dit.ACTIVE_ATTR, None)
expected_r = r_orig.clone()
expected_r[:, flip_at, 2 * dim :] = -r_orig[:, flip_at, 2 * dim :]
check("value lever: refiner (text only) flips the same token", torch.equal(r_hooked, expected_r))

# 8. end to end: neutral rows are an exact identity, a flip is not; the rows ride the kwargs
with torch.no_grad():
    same = dit(x, sigma, context, transformer_options={}, **{NEGPIP_MASK_KEY: neutral_row})
    different = dit(x, sigma, context, transformer_options={}, **{NEGPIP_MASK_KEY: mask_row})
    separate = dit(x[:1], sigma[:1], context[:1], transformer_options={}, **{NEGPIP_MASK_KEY: mask_row[:1]})
check("forward: neutral rows == unpatched output", torch.equal(same, plain_out))
check("forward: a flipped token changes the output", not torch.allclose(different, plain_out))
check("forward: cond and uncond as separate batches agree with the joint one", torch.allclose(separate[0], different[0], rtol=1e-4, atol=1e-5))

# 9. the uncond block skip keeps working under the hooks
with torch.no_grad(), mock.patch.object(settings, "uncond_skip_layers", return_value=frozenset({2})):
    hooked_skip = dit(x, sigma, context, transformer_options={"cond_or_uncond": [0, 1]}, **{NEGPIP_MASK_KEY: neutral_row})
negpip_dit.unpatch_dit(adapter, dit)
with torch.no_grad(), mock.patch.object(settings, "uncond_skip_layers", return_value=frozenset({2})):
    plain_skip = dit(x, sigma, context, transformer_options={"cond_or_uncond": [0, 1]})
check("uncond skip: hooked == unpatched, per sample", torch.equal(hooked_skip, plain_skip) and not torch.allclose(plain_skip[1], plain_out[1]))

# 10. the logit bias: kernels.attention swapped for attention_pytorch with the bias at the trailing keys
captured = []
real_attention_pytorch = negpip_moss.attention_pytorch


def spy(q, k, v, heads, mask=None, **kwargs):
    captured.append((int(q.shape[2]), mask.clone()))
    return real_attention_pytorch(q, k, v, heads, mask=mask, **kwargs)


bias_row = torch.zeros(batch, txt, 1)
bias_row[:, flip_at, 0] = 2.0
calls = []
real_kernels_attention = transformer.kernels.attention


def counting(q, k, v, transformer_options=None):
    calls.append(1)
    return real_kernels_attention(q, k, v, transformer_options=transformer_options)


transformer.kernels.attention = counting
negpip_dit.patch_dit(adapter, dit, **dict(options, block_start=0, block_end=1, force_pytorch_attention=True))
check("bias: kernels.attention swapped", getattr(transformer.kernels.attention, "_negpip", False) and getattr(transformer.kernels, negpip_dit.ORIGINAL_ATTENTION) is counting)
with torch.no_grad(), mock.patch.object(negpip_moss, "attention_pytorch", spy):
    biased = dit(x, sigma, context, transformer_options={}, **{NEGPIP_MASK_KEY: neutral_row, NEGPIP_BIAS_KEY: bias_row})
check("bias: selected blocks and refiners take the SDPA path, the rest the original", len(captured) == 4 and len(calls) == 2)
block_masks = [m for keys, m in captured if keys == imglen + txt]
refiner_masks = [m for keys, m in captured if keys == txt]
check("bias: two joint-sequence masks, two text-only masks", len(block_masks) == 2 and len(refiner_masks) == 2)
expected_block = torch.zeros(batch, 1, imglen + txt)
expected_block[:, 0, imglen + flip_at] = 2.0
expected_refiner = torch.zeros(batch, 1, txt)
expected_refiner[:, 0, flip_at] = 2.0
check("bias: padded to [image, text], the bias on the trailing text key", all(torch.equal(m, expected_block) for m in block_masks))
check("bias: refiners get it over the text keys alone", all(torch.equal(m, expected_refiner) for m in refiner_masks))
check("bias: changes the output", not torch.allclose(biased, plain_out))
with torch.no_grad():
    unbiased = dit(x, sigma, context, transformer_options={}, **{NEGPIP_MASK_KEY: neutral_row, NEGPIP_BIAS_KEY: torch.zeros_like(bias_row)})
check("bias: a zero bias through attention_pytorch matches the eager path", torch.allclose(unbiased, plain_out, rtol=1e-4, atol=1e-5))

# 11. unpatching is a true inverse
negpip_dit.unpatch_dit(adapter, dit)
check("unpatch: kernels.attention restored", transformer.kernels.attention is counting and not hasattr(transformer.kernels, negpip_dit.ORIGINAL_ATTENTION))
transformer.kernels.attention = real_kernels_attention
clean = all("forward" not in vars(m) for m in [dit, *dit.layers, *dit.text_embedder.blocks, *(layer.proj["qkv"] for layer in dit.layers), *(b.qkv for b in dit.text_embedder.blocks)])
check("unpatch: no forward left in any instance dict", clean)
check("unpatch: no attributes left on the blocks", not any(hasattr(layer, negpip_dit.ACTIVE_ATTR) or hasattr(layer, negpip_dit.ORIGINAL_FORWARD) for layer in dit.layers))
with torch.no_grad():
    check("unpatch: output == before patching", torch.equal(dit(x, sigma, context), plain_out))

# 12. compile_conditions learns the dict shape, with the rows as plain Conditions
negpip_dit._hook_compile_conditions()
out = negpip([weighted], one_pass)
compiled = condition.compile_conditions({"crossattn": out["crossattn"][0], NEGPIP_MASK_KEY: out[NEGPIP_MASK_KEY][0]})
check("compile_conditions: crossattn + rows as ConditionCrossAttn, no pooled vector", isinstance(compiled[0]["model_conds"]["c_crossattn"], condition.ConditionCrossAttn) and isinstance(compiled[0]["model_conds"][NEGPIP_MASK_KEY], condition.ConditionCrossAttn))

# 13. the script: activation, a layout never activates, infotext, a bad DiT stands down with a reason
script_module = load_script()
script = script_module.NegPiP()


def fake_p(prompts, negatives=("",)):
    return types.SimpleNamespace(sd_model=model, prompts=list(prompts), negative_prompts=list(negatives), extra_generation_params={}, clear_prompt_cache=lambda: None)


ui = dict(enable=True, value_strength=1.0, mean_reference=1.0, handle_deemphasis=False, handle_emphasis=True, emphasis_gain=1.0, handle_value_emphasis=False, single_pass=True, patch_refiners=False, block_start=0, block_end=41, block_stride=1)

check("scan: layout alone is not weighted", scan(fake_p([layout]), one_pass, adapter) == (False, False, False) and scan(fake_p([stripped]), one_pass, adapter) == (False, False, False))
check("scan: negative weight", scan(fake_p([weighted]), one_pass, adapter) == (True, True, False))
check("scan: emphasis needs the bias", scan(fake_p([emph]), WeightConfig(emphasis=True), adapter) == (True, True, True))
check("resolve: layout stands down", script._resolve(fake_p([layout]), **ui) is None)

p = fake_p([weighted])
script.process_batch(p, **ui)
out = model.get_learned_conditioning(Prompt([weighted]))
check("script: text hooked, rows produced", isinstance(out, dict) and NEGPIP_MASK_KEY in out and torch.equal(out["crossattn"][0], engine.encode(bare)))
check("script: DiT hooked", getattr(dit.forward, "_negpip", False) and getattr(dit.layers[0].forward, "_negpip", False) and not getattr(dit.text_embedder.blocks[0].forward, "_negpip", False))
check("script: infotext under the model-neutral key", p.extra_generation_params.get("NegPiP value strength") == 1.0 and "Krea2 NegPiP value strength" not in p.extra_generation_params)
check("script: no bias in the prompt, so SDPA not forced", not getattr(transformer.kernels.attention, "_negpip", False))
script.postprocess(p, None)
check("script: postprocess unpatches", not getattr(model.get_learned_conditioning, "_negpip", False) and not getattr(dit.forward, "_negpip", False))

p = fake_p([emph])
script.process_batch(p, **ui)
check("script: emphasis in the prompt forces the SDPA swap", getattr(transformer.kernels.attention, "_negpip", False))
script.postprocess(p, None)
check("script: swap undone", not getattr(transformer.kernels.attention, "_negpip", False))

negpip_moss.MOSS_LAYERS = 42
warnings = []
with mock.patch.object(script_module.logger, "warning", lambda msg, *a, **k: warnings.append(msg)):
    check("resolve: the real constants refuse the tiny DiT", script._resolve(fake_p([weighted]), **ui) is None)
check("resolve: the refusal is logged with the reason", any("4 layers, not 42" in w and "MOSS-Image" in w for w in warnings))

finish()
