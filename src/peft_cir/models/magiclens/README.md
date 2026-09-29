# MagicLens (PyTorch port)

[MagicLens](https://open-vision-language.github.io/MagicLens/) (ICML'24). The upstream release
is JAX/Flax on scenic CLIP, and `peft` cannot inject adapters into Flax modules — so the model
is reimplemented here in PyTorch on the same HuggingFace CLIP towers every other architecture
uses, and the released checkpoints are converted.

## What is different about this model

The pseudo-word architectures map the reference to a token, splice it into a prompt, and
retrieve CLIP *text* embeddings against CLIP *image* embeddings. MagicLens has no pseudo-word.
It has a head — 4 pre-LN transformer layers over the two modality tokens, then attention
pooling with one learned query token — and **both sides of the retrieval pass through it**:

```
query     = head(image_embeds(reference), text_embeds(instruction))
candidate = head(image_embeds(candidate),  text_embeds(""))
```

The empty instruction on the candidate side is upstream's own recipe. Two consequences:

* **A candidate vector depends on a trainable module.** This is why the harness asks every
  architecture for `candidates_from_features` and calls it inside the step (and once per epoch
  for the val gallery) rather than normalizing a cached table up front. For the pseudo-word
  archs that method is just `F.normalize`, so their numerics are unchanged.
* **The head is this arch's mapping network** — always trainable except under `frozen`, saved
  in the trainable state dict, and much larger than `Phi` (16.8M parameters at base, 37.8M at
  large), so trainable-parameter counts differ from SEARLE-XL's by a constant across methods.

The head carries three conventions that do not match PyTorch defaults:
`LayerNorm` scales by `1 + scale`, attention logits are capped at `50·tanh(logits/50)`, and the
pooler scales queries by `softplus(w)·1.442695/√H` rather than `H^-0.5`. Note that the *CLIP*
LayerNorms in the same checkpoint use the ordinary convention — both live in `head.py`.

**The temperature starts at 100, not 14.3.** This checkpoint carries CLIP's trained
`logit_scale` (4.6052, which is also the value the trainer clamps at), whereas SEARLE-XL and
Pic2Word start from `log(1/0.07)`. Ranking is unaffected, so the frozen rows are unaffected; the
fine-tuning loss just starts sharper. Passing `logit_scale_init` to `build` overrides it.

## Prerequisite: convert the checkpoints

The released `.pkl` files are Flax parameter trees. Conversion needs `flax`, which only the
`magiclens` venv has, so it runs with that interpreter (which also has `torch`).
`convert.py` imports nothing from `peft_cir`, so the package need not be installed there:

```bash
for SIZE in base large; do
    .venv/magiclens/bin/python src/peft_cir/models/magiclens/convert.py \
        --checkpoint resources/pretrained/magiclens/magic_lens_clip_${SIZE}.pkl
done
```

This writes `magic_lens_clip_{base,large}_torch.pt` next to the originals — the CLIP towers,
the head, the temperature, and a `meta` dict of every architecture dimension read off the
tensor shapes. Nothing hardcodes base/large sizes; `--clip_model_name` only selects a file.

| `--clip_model_name` | backbone | embed dim | head |
|---|---|---|---|
| `base` | ViT-B/16, text width 512 | 512 | 4 layers, 8 heads |
| `large` | ViT-L/14, text width 768 | 768 | 4 layers, 16 heads |

## Port fidelity

Verified against a fixture of the upstream Flax model's activations for a fixed seeded input,
comparing the vision tower, the text tower and the head separately so a failure localises.
Measured on the base checkpoint, converted weights against Flax:

| tensor | max abs deviation | scale |
|---|---|---|
| `img_embed` | 3.8e-06 | 11.1 |
| `txt_embed` | 4.8e-06 | 10.5 |
| pooled multimodal embed | 1.2e-05 | — |
| normalized (the retrieval vector) | 5.6e-07 | 1.0 |

Cosine similarity to the Flax retrieval vectors is 1.000000 to six decimals. Both towers and
the head load with `strict=True`, and the converter refuses a checkpoint with any key it does
not recognise rather than dropping a tensor silently.

The one intentional deviation is preprocessing: the harness uses the same `CLIPImageProcessor`
as every other arch instead of upstream's `process_img`, which rescales each image by its own
maximum and resizes bilinearly. Identical pixels across archs is worth more here than
bit-matching a quirk.

## Why this model has its own recipe

The first CIRR grid ran under the shared harness recipe and turned up an anomaly: full
fine-tuning landed far *below* frozen (R@1 20.83 vs 33.51, `cirr_avg` 47.75 vs 67.99). Its
`cirr_avg` peaked at **epoch 0** and fell afterwards while its **training loss kept going down**
— a generalisation failure, not divergence or underfitting.

`recipe.sh` implements MagicLens's own training setup instead: asymmetric loss (the reference's
`loss_q2t` alone), fixed temperature 0.07, no weight decay, warmup 300 / min_lr 2e-6, batch 32,
and adafactor at lr 2e-4 for `full` only. That recovers `cirr_avg` 73.13 / R@1 44.80 with a flat
val curve. The comments in `recipe.sh` explain each part and why adafactor must *not* be used
for the adapter methods (a zero-initialised LoRA B gets an effective lr of 2e-7 under it).

Only the faithful recipe survives the reorganisation: the earlier symmetric grid that produced
the table above wrote to a separate `magiclens_peft` tree, which is legacy and is not
regenerated. Passing `symmetric` now keeps the faithful recipe and writes its cross-eval rows
to `outputs/cross/magiclens_peft_faithful_symmetric`, beside the default tree rather than over
it.

The mechanism behind the anomaly is specific to this arch: every candidate is encoded through
the head with an *empty* instruction, so the `candidate->query` direction of the symmetric loss
asks a caption-free vector to pick its own instruction out of the batch — weaker and partly
ill-posed. Adapters, being constrained, survive it; `full` does not. Passing `symmetric` as the
loss keeps the rest of the recipe and isolates that one variable.

## Running

Every command carries the recipe's own flags, which is what `recipe.sh` exists to supply:

```bash
PY=.venv/lincir/bin/python
RECIPE="--loss asymmetric --temperature 0.07 --weight_decay 0 --warmup_steps 300 --min_lr 2e-6"

# zero-shot baseline; frozen rows go to the plain magiclens_peft tree, not the recipe's own
PYTHONPATH=src $PY tools/evaluate.py --arch magiclens --benchmark cirr --split val \
    --model_tag magiclens --output outputs/CIRR/magiclens_peft

# an adapter cell: AdamW at 1e-4
PYTHONPATH=src $PY tools/train.py --arch magiclens --dataset cirr --method lora \
    --target text_vision $RECIPE --optimizer adamw --lr 1e-4 \
    --batch_size 32 --epochs 12 --patience 8 --output outputs/CIRR/magiclens_peft_faithful

# full fine-tuning: adafactor at 2e-4, and see the memory note below
PYTHONPATH=src $PY tools/train.py --arch magiclens --dataset cirr --method full \
    --target text_vision $RECIPE --optimizer adafactor --lr 2e-4 \
    --batch_size 32 --epochs 12 --patience 8 --output outputs/CIRR/magiclens_peft_faithful

# cross-dataset transfer
PYTHONPATH=src $PY tools/evaluate.py --benchmark circo --split val \
    --peft_checkpoint outputs/CIRR/magiclens_peft_faithful/lora_text_vision_asymmetric \
    --output outputs/cross/magiclens_peft_faithful
```

`full text_vision` measures ~39GB at batch 32, so it wants an 80GB card. Adding
`--grad_checkpointing` fits it in 40GB at ~30% more time: checkpointing is verified to give
bit-identical gradients and, unlike gradient accumulation, leaves the contrastive negative pool
untouched.

`--query_modality text` is refused for this arch in `tools/inference.py`: upstream answers
text-only queries by feeding a *blank image* through the vision tower, which is not wired here.

`base` and `large` must not share an output tree. A checkpoint directory is named after the
(method, target) pair alone, so the two sizes would overwrite each other's `best.pt` — and
cross-dataset evaluation reads those checkpoints back, so a lost one costs a whole transfer
matrix.
