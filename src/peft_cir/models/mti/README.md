# MTI

[MTI](https://github.com/Chen-Junyang-cn/PLI) — *"Pretrain like Your Inference: Masked Tuning
Improves Zero-Shot Composed Image Retrieval"* ([arXiv 2311.07622](https://arxiv.org/abs/2311.07622),
ICME 2025). Upstream calls the project **PLI**, after the paper title; this study calls the
architecture **`mti`**, after the method.

## What is different about this model

Every other arch here carries a trainable mapping network — SEARLE-XL a `Phi` MLP, Pic2Word an
image-to-pseudo-word mapper, MagicLens a multimodal head. **MTI has none.** Masked tuning is a
*pretraining* procedure (mask most of a reference image's patches and make (masked image +
caption) retrieve the unmasked image), and what it ships is a set of fine-tuned CLIP ViT-L/14
weights and nothing else — the released checkpoint's `compositor` entry is literally empty.
Composition at inference is a parameter-free weighted sum:

```
query     = normalize( normalize(text(caption)) + img_weight * normalize(image(reference)) )
candidate = normalize( image(candidate) )
```

with `img_weight = 0.25` (upstream's ViT-L/14 setting) and no prompt template — the caption is
tokenized bare. Two consequences for this benchmark:

* **This is the cleanest PEFT arm in the study.** Under any method the only trainable tensors
  are the adapters plus the temperature, so the adapter comparison is not confounded by a
  co-trained mapper. It is also the cheapest to train — one text pass, one image pass, no
  patch-token sequence.
* **The weights *are* the method**, so a control arm exists here that the others have no
  analogue for: the identical architecture and composition with *stock* CLIP-L weights, which
  isolates what masked tuning bought.

`img_weight` stays a fixed hyperparameter rather than a learned scalar, so the frozen and PEFT
arms differ in exactly one thing — the adapters — and the frozen arm remains the paper's
zero-shot model.

## Port

The released checkpoint is in OpenAI-CLIP layout, which fuses QKV into one `in_proj_weight`
inside `nn.MultiheadAttention` — and `peft` cannot inject adapters there. `convert.py` relabels
the weights into the HuggingFace CLIP layout the rest of the benchmark uses, so
`peft_cir.adapters.attach_peft` applies unchanged. Every architecture dimension is inferred
from tensor shapes, and the conversion raises on any unrecognized or missing key rather than
dropping a tensor silently. Because the release is a *training* checkpoint (weights + optimizer
moments + step), only `model_state_dict` is read, and its empty `compositor` is asserted rather
than assumed.

```bash
.venv/lincir/bin/python -m peft_cir.models.mti.convert            # -> mti_clip_large_torch.pt
.venv/lincir/bin/python -m peft_cir.models.mti.convert --stock    # -> mti_clip_large_stock.pt
```

Conversion fidelity is checked *exactly*, not against published numbers: a miniature real
OpenAI CLIP is converted through the same mapping, and the HF towers are required to reproduce
its `encode_image` / `encode_text` to 1e-4.

## Preprocessing caveat

This study feeds every arch identical `CLIPImageProcessor` pixels, whereas the released MTI
checkpoint was tuned under CLIP4Cir's `targetpad` transform. Cross-arch comparability is the
point here, so expect the frozen numbers to land somewhat below the published ones. Published
ViT-L/14 reference points, for orientation: CIRR test R@1 26.15 / R@5 56.82 / R@10 69.30 /
R_subset@1 56.22; FashionIQ val R@10 36.37 / R@50 58.78. Stock-CLIP baseline: CIRR R@1 12.40 /
R@5 36.20 / R@10 49.10; FashionIQ R@10 19.80 / R@50 35.70.

## Running

```bash
PY=.venv/lincir/bin/python

# zero-shot baseline -- the fidelity gate; run and check this before spending GPU on the grid
PYTHONPATH=src $PY tools/evaluate.py --arch mti --benchmark cirr --split val \
    --model_tag mti --output outputs/CIRR/mti_peft

# one cell of the grid
PYTHONPATH=src $PY tools/train.py --arch mti --dataset cirr --method lora \
    --target text_vision --batch_size 16 --lr 1e-4 --epochs 10 --patience 3 \
    --output outputs/CIRR/mti_peft

# cross-dataset transfer: score that checkpoint, unmodified, on another benchmark
PYTHONPATH=src $PY tools/evaluate.py --benchmark circo --split val \
    --peft_checkpoint outputs/CIRR/mti_peft/lora_text_vision \
    --output outputs/cross/mti_peft_asymmetric
```

The stock-CLIP control is the frozen eval pointed at the other checkpoint:

```bash
python tools/evaluate.py --benchmark cirr --split val --arch mti \
    --phi_checkpoint resources/pretrained/mti/mti_clip_large_stock.pt \
    --model_tag mti_stock --output outputs/CIRR/mti_peft
```

Recipe: the shared harness one, identical to SEARLE-XL's and Pic2Word's, so the three columns
stay comparable.
