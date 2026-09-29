# Pic2Word

[Pic2Word](https://github.com/google-research/composed_image_retrieval) (CVPR'23). The lightest
architecture in the study: **stock OpenAI CLIP ViT-L/14 plus one ~2M-parameter mapper**
(`IM2TEXT`), trained against frozen CLIP and composed by splicing the mapper's output into a
text prompt at a placeholder.

```
t = mapper(CLIP.encode_image(reference))     # one pseudo-word token embedding
q = CLIP.encode_text("a photo of * , {caption}", with t spliced at the * position)
candidate = CLIP.encode_image(candidate)     # a plain CLIP image embedding
```

No tower conversion (CLIP is never fine-tuned upstream) and no trainable candidate path.

## Why it cannot reuse SEARLE's `Phi`

Both hold three `Linear` layers, which invites reuse. They are not interchangeable:

| | `IM2TEXT` (Pic2Word) | `Phi` (SEARLE / LinCIR) |
|---|---|---|
| block order | `Linear -> Dropout -> ReLU` | `Linear -> GELU -> Dropout` |
| activation | **ReLU** | **GELU** |
| structure | `n_layer` blocks, then a separate `fc_out` | one flat `nn.Sequential` |
| parameter names | `layers.{0,1}.0.*`, `fc_out.*` | `layers.{0,3,6}.*` |

The activation alone makes it a different function, so no state-dict remap is possible.
`mapper.py` transcribes upstream's module, and is checked against an independent copy of
upstream's definition — `torch.equal`, not a tolerance.

## The prompt, and why it is the risky part

This is where a careless port silently produces plausible-but-wrong numbers. Upstream's
templates, from `src/data.py`:

| benchmark | upstream template |
|---|---|
| CIRR | `'a photo of * , {caption}'` |
| FashionIQ | `'a photo of * , {cap2} and {cap1}'` (both captions, cap2 first) |
| Shoes | no upstream eval; inherits CIRR's single-caption form |

The placeholder is `*`, CLIP token id **265**, where SEARLE's `$` is **259**. The literal
template, the id, and the fact that the query actually moves when the mapper output changes are
all pinned.

## Prerequisite: fetch and convert the checkpoint

The release is a 1.72 GB Google Drive training checkpoint, of which only `img2text` is needed.

```bash
uv pip install gdown --python .venv/lincir/bin/python
.venv/lincir/bin/python -m gdown <file-id> -O resources/pretrained/pic2word/pic2word_raw.pt
.venv/lincir/bin/python -m peft_cir.models.pic2word.convert \
    --checkpoint resources/pretrained/pic2word/pic2word_raw.pt
```

That writes `pic2word_large.pt` holding the mapper plus a `meta` dict of dimensions read off
the tensor shapes, so nothing hardcodes ViT-L/14's numbers. The converter accepts the layouts a
release plausibly takes (nested under `img2text`, `module.`-prefixed from a DataParallel save,
or a bare state dict) and raises rather than guessing on anything else.

> The upstream README's own link (id `1IxRi2Cj81RxMu0ViT4q4nkfyjbSHm1dF`) returns **404** — that
> file was removed. Use the id that actually resolves.

## Fidelity gate — PASSED

Frozen Pic2Word, measured 2026-09-03, against the paper (arXiv 2302.03084v2). **FashionIQ val
is the like-for-like check** — the paper's table and ours are both val:

| subtask | published R@10 / R@50 | measured | delta |
|---|---|---|---|
| dress | 20.0 / 40.2 | 20.03 / 40.80 | +0.03 / +0.60 |
| shirt | 26.2 / 43.6 | 25.56 / 43.62 | −0.64 / +0.02 |
| toptee | 27.9 / 47.4 | 27.49 / 46.86 | −0.41 / −0.54 |
| **avg** | **24.7 / 43.7** | **24.36 / 43.76** | **−0.34 / +0.06** |

Sub-point agreement on every subtask. **CIRR is not like-for-like**: the paper's Table 3 is
captioned "Evaluation on CIRR test set" while the harness reports val (CIRR's test GT is
server-side). Recorded for reference — the offsets are the size and direction a val/test change
produces: R@1 23.9 → 23.03, R@5 51.7 → 51.42, R@10 65.3 → 64.20, R@50 87.8 → 86.41.

Checked at conversion time rather than assumed: **all 446 CLIP tensors in the release are
bit-identical to `clip.load("ViT-L/14")`** (max abs deviation 0.0), confirming Pic2Word froze
CLIP and that this port is right to use the harness's stock towers.

## Running

```bash
PY=.venv/lincir/bin/python

# zero-shot baseline -- the fidelity gate; run and check this before spending GPU on the grid
PYTHONPATH=src $PY tools/evaluate.py --arch pic2word --benchmark cirr --split val \
    --model_tag pic2word --output outputs/CIRR/pic2word_peft

# one cell of the grid
PYTHONPATH=src $PY tools/train.py --arch pic2word --dataset cirr --method lora \
    --target text_vision --batch_size 16 --lr 1e-4 --epochs 10 --patience 3 \
    --output outputs/CIRR/pic2word_peft

# cross-dataset transfer: score that checkpoint, unmodified, on another benchmark
PYTHONPATH=src $PY tools/evaluate.py --benchmark circo --split val \
    --peft_checkpoint outputs/CIRR/pic2word_peft/lora_text_vision \
    --output outputs/cross/pic2word_peft
```

Recipe: the shared harness one, deliberately, so these columns stay comparable to SEARLE-XL's
and MTI's. Only the *prompt* follows upstream, because a wrong prompt makes the architecture
wrong rather than merely differently tuned.
