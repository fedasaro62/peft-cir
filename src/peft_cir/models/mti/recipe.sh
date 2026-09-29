# MTI's training recipe: the shared harness one.
#
# Symmetric loss, AdamW, learned logit_scale, batch 16 (8 for full fine-tuning, which needs
# both the smaller batch and the smaller step). SEARLE-XL and Pic2Word use the identical
# recipe, so their columns stay comparable to this one; only MagicLens overrides it.
recipe() {
    MODEL_TAG="mti_large_${DATASET}"
    OUT_NAME="mti_peft"
    EVAL_BATCH=128
    EPOCHS=10
    PATIENCE=3
    OPTIMIZER=adamw
    DEFAULT_LOSS=symmetric
    EXTRA=""
    if [ "$METHOD" = "full" ]; then
        BATCH=8;  LR=1e-5
    else
        BATCH=16; LR=1e-4
    fi
}
