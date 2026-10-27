"""Offline check of the MOSS-Image path against Forge Neo's own MOSS-Image support (`be6fa5d4`).

Real: Forge's `Qwen359BEngine` (its `SDClipModel` and `SDTokenizer` inside it, `layer="last"`),
its `backend.nn.moss_image.MOSSImage` with Comfy-Org's fused q/k/v at a tiny size, built under
`using_forge_operations` so its norms and linears are Forge's own, its module-level `attention`
and the `attention_function` it ends in, Forge's parser and emphasis classes, and the script.
Fake: the Qwen3.5 transformer (a deterministic causal stand-in), the tokenizer's vocabulary
(Qwen3-VL-4B's `tokenizer.json`, the same `Qwen2TokenizerFast` class and BPE as the Qwen3.5
file the checkpoint ships), and the diffusion engine around them.

What it proves: an unweighted prompt, a layout and a line of raw chat markup are encoded
byte-identically to the engine on its own; the JSON Builder's output — a layout with escaped
parentheses and an *Extra* line after it — keeps the layout as data and parses the rest; a
claimed weight leaves the conditioning of the unweighted sentence and flags exactly the word's
tokens, template tokens neutral, through the decoded-offsets path the slow tokenizer forces as
well as the reported one; the rows land on the *trailing* text positions of the joint sequence,
in the value third of the fused projection only; the logit bias reaches `attention_pytorch`
padded to `[image, text]` through the swapped `attention_function`; Skip Layer Guidance's block
replacement survives; unpatching is a true inverse; a tiny or foreign DiT stands the extension
down with a reason.

The fast tokenizer is looked for at `QWEN_FAST_TOKENIZER` or in the Hugging Face cache.
"""

import glob
import json
import os
import types
from pathlib import Path
from unittest import mock

import forge_env  # noqa: F401  (bootstraps Forge; must come first)
from forge_env import Prompt, check, finish, load_script

import torch
from transformers import Qwen2TokenizerFast

from backend import attention, operations
from backend.nn import moss_image
from backend.sampling import condition
from backend.text_processing.moss_image_engine import Qwen359BEngine
from modules.shared import opts

from lib_krea2_negpip import NEGPIP_BIAS_KEY, NEGPIP_MASK_KEY, WeightConfig
from lib_krea2_negpip import dit as negpip_dit
from lib_krea2_negpip import moss as negpip_moss
from lib_krea2_negpip import text as negpip_text
from lib_krea2_negpip.adapters import detect
from lib_krea2_negpip.moss import MossExtensionAdapter, MossNativeAdapter
from lib_krea2_negpip.prompts import scan

#   the tiny DiT runs on the CPU, where Forge's flash/sage pick fails and falls back with
#   an error line per call; this is the fallback it would take, chosen up front.  The DiT
#   module bound its own copy of the name at import, which is the one the swap is tested on
attention.attention_function = attention.attention_pytorch
moss_image.attention_function = attention.attention_pytorch

TOKENIZER = os.environ.get("QWEN_FAST_TOKENIZER") or next(iter(sorted(glob.glob(os.path.expanduser("~/.cache/huggingface/hub/models--Qwen--Qwen3-VL-4B-Instruct/snapshots/*/tokenizer.json")))), None)
if TOKENIZER is None:
    raise SystemExit("no fast Qwen tokenizer found; set QWEN_FAST_TOKENIZER to a folder with tokenizer.json")

TEXT_DIM = 32
TINY = {"dim": 64, "ffn_dim": 96, "num_heads": 4, "num_layers": 4, "text_dim": TEXT_DIM, "latent_channels": 8, "fused_qkv": True}


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
engine = Qwen359BEngine(text_encoder=FakeQwen35(), tokenizer=tokenizer)
check("real Qwen359BEngine built on Forge's SDClipModel", engine.text_encoder.layer == "last" and engine.tokenizer.tokenizer is tokenizer and "{}" in engine.llama_template)

with operations.using_forge_operations(device=torch.device("cpu"), dtype=torch.float32, manual_cast_enabled=False):
    dit = moss_image.MOSSImage(**TINY).eval()
check("real MOSSImage built from Forge operations", type(dit.layers[0].norm1).__qualname__.startswith("ForgeOperations") and not hasattr(dit, "dim"))

generator = torch.Generator().manual_seed(0)
with torch.no_grad():
    for name, param in dit.named_parameters():
        param.copy_(torch.randn(param.shape, generator=generator) * (0.3 if "norm" in name else 0.1) + (1.0 if "norm" in name else 0.0))


class MOSSImage:
    """Named as Forge's engine is, with the engine where Forge keeps it."""

    text_processing_engine_qwen = engine
    forge_objects = types.SimpleNamespace(clip=types.SimpleNamespace(patcher=None), unet=types.SimpleNamespace(model=types.SimpleNamespace(diffusion_model=dit)))

    def get_learned_conditioning(self, prompt):
        return engine(prompt)


model = MOSSImage()
adapter = detect(model)
check("detect: MOSSImage -> MossNativeAdapter", isinstance(adapter, MossNativeAdapter) and not isinstance(adapter, MossExtensionAdapter))
check("detect: the extension's engine is still the extension's", isinstance(detect(type("MossImageEngine", (), {"is_moss_image": True})()), MossExtensionAdapter))
check("detect: the name alone is not enough", detect(type("MOSSImage", (), {})()) is None)
check("engine: no problem with the real text engine", adapter.engine_problem() is None)
check("engine: template from the engine, no truncation", adapter.template == engine.llama_template and adapter.max_length is None)
check("engine: a missing text engine is named", "text_processing_engine_qwen" in (MossNativeAdapter(types.SimpleNamespace()).engine_problem() or ""))


def negpip(lines, config):
    negpip_text.unpatch_text_encoder(model)
    negpip_text.patch_text_encoder(adapter, config)
    try:
        return model.get_learned_conditioning(Prompt(lines))
    finally:
        negpip_text.unpatch_text_encoder(model)


def own(line):
    return engine([line])[0]


one_pass = WeightConfig(single_pass=True)
opts.emphasis = "Original"

# 1. unweighted prompts: the engine's own encode, to the byte, template and all
plain = "a cute cat sitting on a sofa in a living room, a red apple on the table"
out = negpip([plain], one_pass)
check("unweighted == engine(line)", torch.equal(out["crossattn"][0], own(plain)))
check("unweighted: conditioning is (T, dim), rows (T, 1)", out["crossattn"][0].ndim == 2 and out[NEGPIP_MASK_KEY][0].shape == (out["crossattn"][0].shape[0], 1))
check("unweighted rows are neutral", bool((out[NEGPIP_MASK_KEY][0] == 1.0).all()))
check("unweighted: the whole template is in the conditioning", out["crossattn"][0].shape[0] == len(engine.tokenize(plain)) and adapter.plain_ids(plain) == list(engine.tokenize(plain)))
check("empty prompt == engine's single space", torch.equal(negpip([""], one_pass)["crossattn"][0], own("")))

markup = "<|im_start|>user\nhello there<|im_end|>\n<|im_start|>assistant\n<think>\n"
check("chat markup: verbatim", adapter.is_verbatim(markup) and adapter.split_verbatim(markup) == (markup, ""))
#   be6fa5d4 let such a line through untemplated, c6c2b43d templates it like any other; either
#   way the ids come from the engine's own call, which the byte-equal conditioning proves
check("chat markup: verbatim, == engine(line)", torch.equal(negpip([markup], one_pass)["crossattn"][0], own(markup)))

# 2. layouts: the whole line, and the JSON Builder's output with its Extra line
layout = json.dumps({"summary": {"meta": ["poster"], "canvas": [1024, 1024], "caption": "a vase (blue) and a teapot"}, "items": [{"label": "vase", "caption": "a blue vase", "bbox": [-400, -500, 100, -100]}, {"label": "teapot", "caption": "a red teapot", "bbox": [-400, 100, 100, 500]}], "bbox_coord_space": "abs_centralized", "bbox_format": "yxyx"})
escaped = layout.replace("(", "\\(").replace(")", "\\)")  # as the JSON Builder escapes
builder = f"{escaped}\nextra text, (red:-1.0)"
builder_bare = f"{layout}\nextra text, red"

check("layout: detected whole", adapter.is_verbatim(layout) and adapter.is_verbatim(f"  {layout}\n"))
check("layout: escaped parentheses still detected, unescaped as the engine does", adapter.split_verbatim(escaped) == (layout, "") and json.loads(adapter.split_verbatim(escaped)[0])["summary"]["caption"] == "a vase (blue) and a teapot")
check("layout: an ordinary prompt is not one", not adapter.is_verbatim(plain) and adapter.split_verbatim("(a red apple:-1.0) {red|blue} car")[0] is None)
check("builder: the layout is the prefix, the Extra line the rest", adapter.split_verbatim(builder) == (layout, "\nextra text, (red:-1.0)") and not adapter.is_verbatim(builder))
for name, line in (("layout", layout), ("escaped layout", escaped)):
    out = negpip([line], one_pass)
    check(f"{name}: encoded verbatim == engine(line)", torch.equal(out["crossattn"][0], own(line)) and bool((out[NEGPIP_MASK_KEY][0] == 1.0).all()))
    check(f"{name}: bbox brackets are not de-emphasis", out["crossattn"][0].shape[0] == len(engine.tokenize(adapter.unescape(line))))

out = negpip([builder], one_pass)
cond, mask = out["crossattn"][0], out[NEGPIP_MASK_KEY][0]
check("builder: cond == engine(layout + unweighted Extra)", torch.equal(cond, own(builder_bare)))
layout_len = len(tokenizer.encode(engine.llama_template.partition("{}")[0] + layout, add_special_tokens=False))
flagged = [i for i, v in enumerate(mask.flatten().tolist()) if v != 1.0]
check("builder: the layout's tokens are all neutral", flagged and min(flagged) >= layout_len)
check("builder: the Extra line's weight is flagged", tokenizer.decode([adapter.plain_ids(builder_bare)[i] for i in flagged]) == " red")

# 3. a claimed weight: the unweighted sentence's conditioning, the word's tokens flagged
weighted = "a cute cat sitting on a sofa in a living room, (a red apple:-1.0) on the table"
bare = "a cute cat sitting on a sofa in a living room, a red apple on the table"
out = negpip([weighted], one_pass)
cond, mask = out["crossattn"][0], out[NEGPIP_MASK_KEY][0]
check("claimed weight: cond == engine(bare sentence)", torch.equal(cond, own(bare)))
check("claimed weight: rows 1:1 with the conditioning", mask.shape == (cond.shape[0], 1))
ids = adapter.plain_ids(bare)
flipped = [i for i, v in enumerate(mask.flatten().tolist()) if v != 1.0]
text = tokenizer.decode([ids[i] for i in flipped])
check(f"claimed weight: flipped tokens are the phrase ({text!r})", text == " a red apple" and set(mask.flatten()[flipped].tolist()) == {-1.0})
prefix_len = len(tokenizer.encode(engine.llama_template.partition("{}")[0], add_special_tokens=False))
check("claimed weight: template tokens stay neutral", min(flipped) >= prefix_len and bool((mask[:prefix_len] == 1.0).all()) and bool((mask[-3:] == 1.0).all()))

fast = negpip_text._tokenize_line(adapter, weighted, one_pass, "Original")
with mock.patch.object(MossNativeAdapter, "reported_offsets", return_value=None):
    slow = negpip_text._tokenize_line(adapter, weighted, one_pass, "Original")
check("slow tokenizer: the decoded-offsets path gives the same stream and rows", fast == slow)

# 4. an unclaimed weight: the ComfyUI output lerp through Forge's own encoder
emph = "a cute cat sitting on a sofa in a living room, red theme, (blue theme:1.5)"
emph_bare = "a cute cat sitting on a sofa in a living room, red theme, blue theme"
out = negpip([emph], one_pass)
tokens, multipliers, _, _ = negpip_text._tokenize_line(adapter, emph, one_pass, "Original")
expected = engine.text_encoder.encode_token_weights([list(zip(tokens, multipliers))])[0][0]
check("unclaimed weight: encode_token_weights lerp applied", torch.equal(out["crossattn"][0], expected) and sorted(set(multipliers)) == [1.0, 1.5])
check("unclaimed weight: differs from no weight", not torch.equal(out["crossattn"][0], own(emph_bare)))
check("unclaimed weight: rows neutral", bool((out[NEGPIP_MASK_KEY][0] == 1.0).all()))
opts.emphasis = "Ignore"
out = negpip([emph], one_pass)
check("emphasis Ignore: unclaimed weight is a no-op", torch.equal(out["crossattn"][0], own(emph_bare)))
opts.emphasis = "Original"

# 5. the BREAK sentinel, and the option with no meaning here
out = negpip(["a cat, (red:-1.0) hat BREAK a dog"], one_pass)
check("BREAK keeps its spaces", torch.equal(out["crossattn"][0], own("a cat, red hat BREAK a dog")))
split = negpip([weighted], WeightConfig(single_pass=False))
check("single pass off: MOSS-Image is encoded in one pass regardless", torch.equal(split["crossattn"][0], cond) and torch.equal(split[NEGPIP_MASK_KEY][0], mask))

# 5b. the reference/Edit path (c6c2b43d): the positive prompt is handed back, the state kept clean
#   a registered option lives in opts.data, an unregistered one (older Forge) on the instance;
#   set it the way the UI does and put it back the same way
registered = "moss_do_reference" in getattr(opts, "data_labels", {})
previous = getattr(opts, "moss_do_reference", None)
model.ref_latents = [torch.zeros(1, 8, 2, 2)]
model.ini_latent = None
opts.moss_do_reference = "Moss-Image-Edit"
check("edit: referencing is read from the setting", adapter._referencing() and adapter.hand_back(Prompt([weighted])) is not None)
negative = Prompt([weighted])
negative.is_negative_prompt = True
check("edit: the negative prompt is not handed back", adapter.hand_back(negative) is None)
handed = negpip([weighted], one_pass)
check("edit: the positive prompt goes to the engine whole (no rows)", isinstance(handed, list) and torch.equal(handed[0], own(weighted)))
opts.moss_do_reference = "Disable"
check("edit: Disable never hands back", adapter.hand_back(Prompt([weighted])) is None)
if registered:
    opts.moss_do_reference = previous
else:
    delattr(opts, "moss_do_reference")
model.ref_latents = []
check("edit: no references, no hand-back", adapter.hand_back(Prompt([weighted])) is None)

from backend.args import dynamic_args  # noqa: E402

model.ini_latent = torch.zeros(1)
dynamic_args.ref_latents.append(torch.zeros(1))
negpip([weighted], one_pass)
check("plain encode: the img2img latent is consumed and the reference list dropped", model.ini_latent is None and not dynamic_args.ref_latents)

# 6. several lines and the inverse
out = negpip([plain, weighted, plain], one_pass)
check("schedule: one entry per line, cache exact", len(out["crossattn"]) == 3 and torch.equal(out["crossattn"][0], out["crossattn"][2]))
negpip_text.patch_text_encoder(adapter, one_pass)
negpip_text.unpatch_text_encoder(model)
check("unpatch restores", not getattr(model.get_learned_conditioning, "_negpip", False) and "get_learned_conditioning" not in vars(model))

# ------------------------------------------------------------------- DiT side -- #

check("dit: the tiny model is refused with a reason naming the layers", adapter.dit_problem(dit) == "the DiT has 4 layers, not 42")
check("dit: a foreign model is refused", adapter.dit_problem(torch.nn.Linear(2, 2)) == "the DiT has no layers")

negpip_moss.MOSS_LAYERS, negpip_moss.MOSS_DIM = TINY["num_layers"], TINY["dim"]
check("dit: the real MOSSImage passes once the shape constants match; width read off video_embedder", adapter.dit_problem(dit) is None and negpip_moss._dit_dim(dit) == TINY["dim"])

with operations.using_forge_operations(device=torch.device("cpu"), dtype=torch.float32, manual_cast_enabled=False):
    headless = moss_image.MOSSImage(**TINY)
del headless.text_embedder.blocks
check("dit: missing refiners named", adapter.dit_problem(headless) == "the DiT has no text_embedder.blocks")
with mock.patch.object(moss_image, "attention_function", None):
    check("dit: a module with nothing to swap is named", adapter.dit_problem(dit) == "the DiT's module has no attention_function to swap")

batch, height, width, txt = 2, 3, 5, len(adapter.plain_ids(bare))
imglen = height * width
generator = torch.Generator().manual_seed(1)
x = torch.randn(batch, TINY["latent_channels"], height, width, generator=generator)
context = torch.stack([own(bare)] * batch).float()
sigma = torch.full((batch,), 0.73)
assert context.shape == (batch, txt, TEXT_DIM)

flip_at = flipped[0]
mask_row = torch.ones(batch, txt, 1)
mask_row[:, flip_at, 0] = -1.0
neutral_row = torch.ones(batch, txt, 1)

with torch.no_grad():
    plain_out = dit(x, sigma, context, transformer_options={})

options = dict(block_start=0, block_end=41, block_stride=1, patch_refiners=True, mean_reference=0.0, force_pytorch_attention=False)
hooked = negpip_dit.patch_dit(adapter, dit, **options)
check("patch: 4 blocks + 2 refiners hooked", hooked == 6)
check("patch: every block's fused qkv is wrapped, attention_function is not", all(getattr(layer.proj["qkv"].forward, "_negpip", False) for layer in dit.layers) and not getattr(moss_image.attention_function, "_negpip", False))

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
v_hooked, v_orig = hooked_out[..., 2 * dim :], original_out[..., 2 * dim :]
expected_v = v_orig.clone()
expected_v[:, imglen + flip_at, :] = -v_orig[:, imglen + flip_at, :]
check("value lever: only the value third changes", torch.equal(hooked_out[..., : 2 * dim], original_out[..., : 2 * dim]))
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

# 9. Skip Layer Guidance replaces a block through patches_replace; the hooked block is still the block
slg = {"patches_replace": {"dit": {("single_block", 2): lambda args, extra: args}}}
with torch.no_grad():
    hooked_skip = dit(x, sigma, context, transformer_options=dict(slg), **{NEGPIP_MASK_KEY: neutral_row})
    hooked_skip_flip = dit(x, sigma, context, transformer_options=dict(slg), **{NEGPIP_MASK_KEY: mask_row})
negpip_dit.unpatch_dit(adapter, dit)
with torch.no_grad():
    plain_skip = dit(x, sigma, context, transformer_options=dict(slg))
check("SLG: hooked == unpatched with the block skipped, and the skip is real", torch.equal(hooked_skip, plain_skip) and not torch.allclose(plain_skip, plain_out))
check("SLG: the flip still acts in the blocks that run", not torch.allclose(hooked_skip_flip, plain_skip))

# 9b. the block wrapper takes whatever the block takes: c6c2b43d adds ref, ref_kv and kv_out positionally
import inspect  # noqa: E402

block_params = list(inspect.signature(moss_image.Block.forward).parameters)
negpip_dit.patch_dit(adapter, dit, **options)
layer = dit.layers[0]
tokens = torch.randn(batch, imglen + txt, TINY["dim"], generator=generator)
with torch.no_grad():
    mod_vec = dit.shared_block_modulation(dit.time_embedding(moss_image.timestep_embedding(sigma, 256))).unflatten(-1, (6, -1)).unsqueeze(2).unbind(1)
    pe = dit.rope(height, width, txt, tokens.device)
    hooked_pos = layer(tokens, mod_vec, pe, {})
    hooked_kw = layer(tokens, mod_vec, pe, transformer_options={})
check("wrapper: positional and keyword transformer_options agree", torch.equal(hooked_pos, hooked_kw))
if "ref_kv" in block_params:
    with torch.no_grad():
        extra = layer(tokens, mod_vec, pe, {}, None, None)
    check("wrapper: the Edit path's extra positionals pass through", torch.equal(extra, hooked_pos))
negpip_dit.unpatch_dit(adapter, dit)

if hasattr(dit, "preprocess_refs"):
    #   a reference image through the isolated method: its K/V are appended after the
    #   sequence's own inside attention, the sequence itself stays [image, text]
    ref = torch.randn(1, TINY["latent_channels"], 2, 3, generator=generator)
    reflen = 2 * 3

    def with_reference(fn):
        dynamic_args.ref_latents[:] = [ref]
        dynamic_args.moss_reference_method = "index_timestep_zero_text_start"
        dynamic_args.moss_ref_kv = None
        try:
            return fn()
        finally:
            dynamic_args.ref_latents.clear()
            dynamic_args.moss_reference_method = None
            dynamic_args.moss_ref_kv = None

    with torch.no_grad():
        plain_ref = with_reference(lambda: dit(x, sigma, context, transformer_options={}))
    check("edit: the reference changes the plain output", not torch.allclose(plain_ref, plain_out))
    negpip_dit.patch_dit(adapter, dit, **options)
    with torch.no_grad():
        same_ref = with_reference(lambda: dit(x, sigma, context, transformer_options={}, **{NEGPIP_MASK_KEY: neutral_row}))
        flip_ref = with_reference(lambda: dit(x, sigma, context, transformer_options={}, **{NEGPIP_MASK_KEY: mask_row}))
    check("edit: neutral rows == unpatched output with the reference", torch.equal(same_ref, plain_ref))
    check("edit: the flip still acts with the reference", not torch.allclose(flip_ref, plain_ref))
    negpip_dit.unpatch_dit(adapter, dit)

    captured_ref = []

    def spy_ref(q, k, v, heads, mask=None, **kwargs):
        captured_ref.append((int(q.shape[1]), int(k.shape[1]), mask.clone()))
        return negpip_moss.attention_pytorch.__wrapped__(q, k, v, heads, mask=mask, **kwargs) if hasattr(negpip_moss.attention_pytorch, "__wrapped__") else attention.attention_pytorch(q, k, v, heads, mask=mask, **kwargs)

    bias_ref = torch.zeros(batch, txt, 1)
    bias_ref[:, flip_at, 0] = 2.0
    negpip_dit.patch_dit(adapter, dit, **dict(options, block_start=0, block_end=0, patch_refiners=False, force_pytorch_attention=True))
    with torch.no_grad(), mock.patch.object(negpip_moss, "attention_pytorch", spy_ref):
        with_reference(lambda: dit(x, sigma, context, transformer_options={}, **{NEGPIP_MASK_KEY: neutral_row, NEGPIP_BIAS_KEY: bias_ref}))
    negpip_dit.unpatch_dit(adapter, dit)
    joint = [(q, k, m) for q, k, m in captured_ref if q == imglen + txt]
    expected_ref = torch.zeros(batch, 1, imglen + txt + reflen)
    expected_ref[:, 0, imglen + flip_at] = 2.0
    check("edit: the keys are the sequence plus the reference", len(joint) == 1 and joint[0][1] == imglen + txt + reflen)
    check("edit: the bias sits on the text inside the sequence, zero on the reference keys", len(joint) == 1 and torch.equal(joint[0][2], expected_ref))
else:
    check("edit: this Forge has no reference path for MOSS-Image (before c6c2b43d); nothing to check", True)

# 10. the logit bias: attention_function swapped for attention_pytorch with the bias at the trailing keys
captured = []
real_attention_pytorch = negpip_moss.attention_pytorch


def spy(q, k, v, heads, mask=None, **kwargs):
    captured.append((int(k.shape[1]), mask.clone()))
    return real_attention_pytorch(q, k, v, heads, mask=mask, **kwargs)


bias_row = torch.zeros(batch, txt, 1)
bias_row[:, flip_at, 0] = 2.0
calls = []
real_attention_function = moss_image.attention_function


def counting(q, k, v, heads, *args, **kwargs):
    calls.append(1)
    return real_attention_function(q, k, v, heads, *args, **kwargs)


moss_image.attention_function = counting
negpip_dit.patch_dit(adapter, dit, **dict(options, block_start=0, block_end=1, force_pytorch_attention=True))
check("bias: attention_function swapped", getattr(moss_image.attention_function, "_negpip", False) and getattr(moss_image, negpip_dit.ORIGINAL_ATTENTION) is counting)
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
check("unpatch: attention_function restored", moss_image.attention_function is counting and not hasattr(moss_image, negpip_dit.ORIGINAL_ATTENTION))
moss_image.attention_function = real_attention_function
clean = all("forward" not in vars(m) for m in [dit, *dit.layers, *dit.text_embedder.blocks, *(layer.proj["qkv"] for layer in dit.layers), *(b.qkv for b in dit.text_embedder.blocks)])
check("unpatch: no forward left in any instance dict", clean)
check("unpatch: no attributes left on the blocks", not any(hasattr(layer, negpip_dit.ACTIVE_ATTR) or hasattr(layer, negpip_dit.ORIGINAL_FORWARD) for layer in dit.layers))
with torch.no_grad():
    check("unpatch: output == before patching", torch.equal(dit(x, sigma, context, transformer_options={}), plain_out))

# 12. compile_conditions learns the dict shape, with the rows as plain Conditions
negpip_dit._hook_compile_conditions()
out = negpip([weighted], one_pass)
compiled = condition.compile_conditions({"crossattn": out["crossattn"][0], NEGPIP_MASK_KEY: out[NEGPIP_MASK_KEY][0]})
check("compile_conditions: crossattn + rows as ConditionCrossAttn, no pooled vector", isinstance(compiled[0]["model_conds"]["c_crossattn"], condition.ConditionCrossAttn) and isinstance(compiled[0]["model_conds"][NEGPIP_MASK_KEY], condition.ConditionCrossAttn))

# 12b. cond and uncond of different token counts batch into one pass, rows tiled like the tokens
#   a 2:1 pair cut from the weighted prompt's own conditioning, so the tiling branch is the one exercised
even = 2 * (out["crossattn"][0].shape[0] // 2)
long_cond, long_rows = out["crossattn"][0][:even], out[NEGPIP_MASK_KEY][0][:even]
short_cond, short_rows = long_cond[: even // 2], long_rows[: even // 2]
pair_cond = (condition.ConditionCrossAttn(long_cond[None]), condition.ConditionCrossAttn(short_cond[None]))
pair_rows = (condition.ConditionCrossAttn(long_rows[None]), condition.ConditionCrossAttn(short_rows[None]))
check("batching: a plain Condition would refuse the pair, ConditionCrossAttn takes it", not condition.Condition(long_rows[None]).can_concat(condition.Condition(short_rows[None])) and pair_rows[0].can_concat(pair_rows[1]) == pair_cond[0].can_concat(pair_cond[1]))
if pair_cond[0].can_concat(pair_cond[1]):
    tiled_cond = pair_cond[0].concat([pair_cond[1]])
    tiled_rows = pair_rows[0].concat([pair_rows[1]])
    check("batching: rows tiled to the same length as the tokens", tiled_rows.shape[:2] == tiled_cond.shape[:2])
    period = short_cond.shape[0]
    flagged_long = [i for i, v in enumerate(long_rows.flatten().tolist()) if v != 1.0]
    flagged_tiled = [i for i, v in enumerate(tiled_rows[0].flatten().tolist()) if v != 1.0]
    tiles = tiled_cond.shape[1] // long_cond.shape[0]
    check("batching: the long row's flags repeat with its tokens", flagged_tiled == [i + k * long_cond.shape[0] for k in range(tiles) for i in flagged_long] and torch.equal(tiled_cond[0, : long_cond.shape[0]], long_cond) and torch.equal(tiled_cond[1, :period], short_cond) and torch.equal(tiled_cond[1, period : 2 * period], short_cond))
else:
    check("batching: a 2:1 pair must be one Forge batches", False)

# 13. the script: activation, a layout never activates, the builder's Extra does, infotext, a bad DiT stands down with a reason
script_module = load_script()
script = script_module.NegPiP()


def fake_p(prompts, negatives=("",)):
    return types.SimpleNamespace(sd_model=model, prompts=list(prompts), negative_prompts=list(negatives), extra_generation_params={}, clear_prompt_cache=lambda: None)


ui = dict(enable=True, value_strength=1.0, mean_reference=1.0, handle_deemphasis=False, handle_emphasis=True, emphasis_gain=1.0, handle_value_emphasis=False, single_pass=True, patch_refiners=False, block_start=0, block_end=41, block_stride=1)

check("scan: layout alone is not weighted", scan(fake_p([layout]), one_pass, adapter) == (False, False, False) and scan(fake_p([escaped]), one_pass, adapter) == (False, False, False))
check("scan: the builder's Extra line is", scan(fake_p([builder]), one_pass, adapter) == (True, True, False))
check("scan: negative weight", scan(fake_p([weighted]), one_pass, adapter) == (True, True, False))
check("scan: emphasis needs the bias", scan(fake_p([emph]), WeightConfig(emphasis=True), adapter) == (True, True, True))
check("resolve: layout stands down, the builder's prompt does not", script._resolve(fake_p([layout]), **ui) is None and script._resolve(fake_p([builder]), **ui) is not None)

p = fake_p([weighted])
script.process_batch(p, **ui)
out = model.get_learned_conditioning(Prompt([weighted]))
check("script: text hooked, rows produced", isinstance(out, dict) and NEGPIP_MASK_KEY in out and torch.equal(out["crossattn"][0], own(bare)))
check("script: DiT hooked", getattr(dit.forward, "_negpip", False) and getattr(dit.layers[0].forward, "_negpip", False) and not getattr(dit.text_embedder.blocks[0].forward, "_negpip", False))
check("script: infotext under the model-neutral key", p.extra_generation_params.get("NegPiP value strength") == 1.0 and "Krea2 NegPiP value strength" not in p.extra_generation_params)
check("script: no bias in the prompt, so SDPA not forced", not getattr(moss_image.attention_function, "_negpip", False))
script.postprocess(p, None)
check("script: postprocess unpatches", not getattr(model.get_learned_conditioning, "_negpip", False) and not getattr(dit.forward, "_negpip", False))

p = fake_p([emph])
script.process_batch(p, **ui)
check("script: emphasis in the prompt forces the SDPA swap", getattr(moss_image.attention_function, "_negpip", False))
script.postprocess(p, None)
check("script: swap undone", not getattr(moss_image.attention_function, "_negpip", False) and moss_image.attention_function is real_attention_function)

negpip_moss.MOSS_LAYERS = 42
warnings = []
with mock.patch.object(script_module.logger, "warning", lambda msg, *a, **k: warnings.append(msg)):
    check("resolve: the real constants refuse the tiny DiT", script._resolve(fake_p([weighted]), **ui) is None)
check("resolve: the refusal is logged with the reason", any("4 layers, not 42" in w and "MOSS-Image" in w for w in warnings))

finish()
