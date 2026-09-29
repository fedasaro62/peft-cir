# PEFT-CIR

A parameter-efficient fine-tuning benchmark for CLIP-based **composed image retrieval**: given
a reference image and a modification instruction, retrieve the image the instruction describes.

Four retrievers are wrapped with the same PEFT methods, adapted on the same datasets, and
evaluated cross-dataset against a held-out target. The design principle is that the four
differ **only in how they compose a query** — adapters are injected by one shared function
against the same CLIP attention modules, on the same HuggingFace tower pair, under the same
loss and the same preprocessing. A difference between two columns is therefore a difference
between two architectures, not between two harnesses.

---

## Models

All four sit on CLIP ViT-L/14 and split into two families by *where retrieval happens*.

![The two model families: textual-inversion-based, which maps the reference image to a
pseudo-word token spliced into a prompt, and pseudo-triplet-based, which fuses the two unimodal
embeddings directly. Blue flames mark what PEFT makes trainable, snowflakes what stays frozen.](resources/methods_ft.png)

Both families are trained the same way: an InfoNCE loss between the composed query `q` and the
target `c`, with the visual encoder shared between the reference and candidate paths. Only the
composition differs, and only the adapters (plus each architecture's mapping network) train.

### Textual inversion

The reference image is mapped to a **pseudo-word token**, spliced into a text prompt, and
encoded by the text tower. Retrieval matches CLIP *text* embeddings against CLIP *image*
embeddings, so a candidate is a plain normalized image embedding and nothing on the candidate
side is trainable.

| arch | mapping network | prompt |
|---|---|---|
| `pic2word` | `IM2TEXT` mapper (~2M params) | `"a photo of * , {caption}"`, token 265 |
| `searle` | `Phi` MLP | `"a photo of $ that {caption}"`, token 259 |

Both leave CLIP untouched upstream, so the harness's stock HuggingFace towers are already the
correct backbone — no checkpoint conversion, only the mapping network is loaded.

### Pseudo-triplet

No pseudo-word. The query is formed by combining the two unimodal embeddings directly, and both
models ship their own fine-tuned CLIP weights, so their towers come from a converted checkpoint
rather than from stock CLIP.

| arch | mapping network | composition |
|---|---|---|
| `magiclens` | multimodal head (16.8M base / 37.8M large) | `head(image, text)` on **both** sides |
| `mti` | none | `normalize(text + 0.25 · image)`, parameter-free |

MagicLens is the one architecture whose **candidate** vector depends on a trainable module (the
head encodes candidates with an empty instruction), which is why the harness finishes candidate
vectors inside the training step rather than normalizing a cached table up front. MTI is the
opposite extreme, and the study's cleanest arm: with no mapping network, the only trainable
tensors under any method are the adapters themselves plus the temperature, so its adapter
comparison is not confounded by a co-trained mapper.

Each model's own notes — how to obtain and convert its checkpoint, and how the port was
validated against its paper — are in `src/peft_cir/models/<arch>/README.md`, with its training
hyperparameters beside them in `recipe.sh`. The textual-inversion pair and MTI share one recipe,
which is what keeps their columns comparable; MagicLens overrides it with its paper's own, for
reasons that file explains.

---

## PEFT methods

`--method` selects the technique and `--target` selects which CLIP tower(s) receive it
(`text`, `vision` or `text_vision`). The mapping network and the temperature are always
trainable; `vision` is the ablation arm that leaves the text encoder untouched.

![The adapted methods. Reparameterisation methods (LoRA, DoRA, VeRA, AdaLoRA) add a low-rank
branch alongside the frozen weight W; additive methods rescale activations ((IA)³) or insert a
bottleneck block (Adapter). Colour marks the cost tier: expressive in blue, minimal in
orange.](resources/peft_methods.png)

| | expressive | minimal |
|---|---|---|
| **reparameterisation** | `lora`, `dora`, `adalora` | `vera` |
| **additive** | `adapter` | `ia3` |

Plus two baselines — `frozen` (zero-shot, eval-only) and `full` (full fine-tuning) — and
`prompt`, soft prompts prepended to the text sequence, which is text-stream only because there
is no vision analogue.

A checkpoint stores **only the trainable tensors**, alongside a `meta.json` recording
everything needed to rebuild the model — which is what the evaluator reads back.

---

## Datasets

### Adaptation

`cirr`, `fiq` (FashionIQ, pooling its three subtasks into one jointly trained model) and
`shoes`. Each is selected on its own metric: CIRR on ½(R@5 + R_subset@1), FashionIQ on its
standard Avg ½(R@10 + R@50), Shoes on ½(R@10 + R@50).

### Cross-dataset transfer

Every `text_vision` checkpoint is also scored unmodified — no further training — on the
benchmarks it was *not* trained on: a checkpoint trained on `cirr` is evaluated on `fiq`,
`shoes` and `circo`, and likewise for the other two. `circo` only ever appears on this side: it
is eval-only, so no checkpoint is ever trained on it.

### Protocols

Shoes ships no validation split. `--protocol carved` (the default) carves one out of train with
a seeded, image-disjoint split — whole connected components of the reference/target graph stay
on one side, so no image appears in both — and reports test once. `--protocol literature`
instead trains on the full split and selects on **test**, reproducing ARTEMIS, whose released
code has no val split; those numbers are best-epoch-on-test and optimistic by construction, so
they are written to a separate `_lit` tree.

---

## Setup

Each architecture gets its own virtualenv: they pin different, sometimes incompatible versions,
and `requirements.txt` pins a trio (`transformers` 4.34.1 / `huggingface_hub` 0.17.3 /
`accelerate` 0.30.1) that a resolver left to itself gets wrong.

```bash
python3 -m venv .venv/lincir
.venv/lincir/bin/pip install -r requirements.txt
.venv/lincir/bin/pip install -e .          # adds peft_cir without touching those pins
```

That environment runs everything except the MagicLens checkpoint conversion, which needs
`flax` and gets its own `.venv/magiclens` (see that model's README).

`data/`, `resources/` and `outputs/` are tracked as directories so every path in the code
resolves, but their contents are not. Link or place benchmark data under `data/<name>/` and
converted checkpoints under `resources/pretrained/<arch>/`:

```bash
ln -s /path/to/datasets/cirr data/cirr
```

---

## Running

```bash
PY=.venv/lincir/bin/python

# zero-shot baseline -- run this first (see below)
PYTHONPATH=src $PY tools/evaluate.py --arch pic2word --benchmark cirr --split val \
    --model_tag pic2word --output outputs/CIRR/pic2word_peft

# adapt one architecture to one dataset with one method
PYTHONPATH=src $PY tools/train.py --arch pic2word --dataset cirr --method lora \
    --target text_vision --batch_size 16 --lr 1e-4 --epochs 10 --patience 3 \
    --output outputs/CIRR/pic2word_peft

# score a trained checkpoint, in-domain or on any other benchmark
PYTHONPATH=src $PY tools/evaluate.py --benchmark circo --split val \
    --peft_checkpoint outputs/CIRR/pic2word_peft/lora_text_vision \
    --output outputs/cross/pic2word_peft
```

**Run the zero-shot baseline first.** Those rows must land near the numbers published for that
architecture before any GPU time is spent on a grid. A misconfigured recipe produces plausible
but wrong results, which is expensive to notice late and cheap to catch here.

Results land in `outputs/<benchmark>/<arch>_peft*/`: one directory per
`<method>_<target>[_<loss>]` cell, plus a `benchmarks.csv` of scored rows.
`tools/plot_cross_results.py` reads that tree and draws the cross-dataset figures.
