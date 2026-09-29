# SEARLE-XL

[SEARLE](https://github.com/miccunifi/SEARLE) (ICCV'23) — zero-shot CIR by textual inversion.
A frozen CLIP ViT-L/14 pair plus `Phi`, an MLP mapping a CLIP image embedding to one
pseudo-word token, spliced into `"a photo of $ that {caption}"` at the `$` position (token 259).
Candidates are plain CLIP image embeddings, so `candidates_from_features` is a bare
`F.normalize` and the trainer's cached-feature fast path stays exact.

> The public release ships the **SEARLE / SEARLE-XL** `Phi` weights. There are no separately
> released iSEARLE weights (iSEARLE improves the training procedure), so SEARLE-XL (ViT-L/14)
> is the available off-the-shelf zero-shot model.

## Checkpoint

`Phi` is vendored at `resources/pretrained/searle/SEARLE_ViT-L14.pt`; the CLIP backbone is
downloaded from HuggingFace on first run. Nothing needs converting — SEARLE never fine-tunes
CLIP, so the harness's stock HF towers already are the right backbone.

`Phi` is also **LinCIR's** mechanism: the two differ only in how the released checkpoint was
trained, so `--phi_checkpoint .../lincir_large.pt` loads here unchanged. The harness registered
the pair under a single `lincir` arch before this split, and `peft_cir.registry` still resolves
that name to `searle` so older `meta.json` files keep working.

## Caveat

Running SEARLE's `Phi` on the HuggingFace CLIP implementation rather than its native
OpenAI-CLIP `clip.load` can shift the frozen baseline slightly against the published zero-shot
number. Every PEFT row here is measured on that same HF backbone, so they are internally
comparable — and comparable to the other three architectures, which is the point.

## Running

```bash
PY=.venv/lincir/bin/python

# zero-shot baseline -- the fidelity gate; run and check this before spending GPU on the grid
PYTHONPATH=src $PY tools/evaluate.py --arch searle --benchmark cirr --split val \
    --model_tag searle --output outputs/CIRR/searle_peft

# one cell of the grid
PYTHONPATH=src $PY tools/train.py --arch searle --dataset cirr --method lora \
    --target text_vision --batch_size 16 --lr 1e-4 --epochs 10 --patience 3 \
    --output outputs/CIRR/searle_peft

# cross-dataset transfer: score that checkpoint, unmodified, on another benchmark
PYTHONPATH=src $PY tools/evaluate.py --benchmark circo --split val \
    --peft_checkpoint outputs/CIRR/searle_peft/lora_text_vision \
    --output outputs/cross/searle_peft
```

Recipe: the shared harness one (`recipe.sh`) — symmetric loss, AdamW, learned `logit_scale`,
batch 16 / lr 1e-4, and 8 / 1e-5 for `full`. Pic2Word and MTI use it identically, which is what
keeps the three columns comparable; only MagicLens overrides it.
