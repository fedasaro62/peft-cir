# MagicLens's training recipe: the paper's own, not the shared harness one.
#
# Validated on CIRR with --method full: cirr_avg 73.13 / R@1 44.80, against frozen's
# 67.99 / 33.51 and the harness recipe's 47.75 / 20.83. Its val curve is flat and stable
# (71-73 across 12 epochs) instead of peaking at epoch 0 and collapsing.
#
# It splits in two, by what each part is FOR:
#
# (a) Architecture-level, applied to EVERY method -- these are how MagicLens defines its
#     objective, so they are not method-specific:
#       --loss asymmetric      loss_q2t only                     (training/loss.py)
#       --temperature 0.07     FIXED, logit_scale frozen; the reference never reads
#                              clip/logit_scale, which holds a vestigial ln(100)
#       --weight_decay 0       optax adafactor weight_decay_rate=None
#       --warmup_steps 300 / --min_lr 2e-6                       (configs/*.yaml)
#       batch 32                                                 (configs/*.yaml)
#
# (b) Optimizer, which is ONLY meaningful for `full`. optax.adafactor's
#     multiply_by_parameter_scale=True makes each step proportional to a parameter's own RMS,
#     which is exactly what protects pretrained tower weights. Applied to a zero-initialised
#     adapter it does the opposite: LoRA's B starts at RMS 0, so param_scale floors at
#     eps2=1e-3 and B's effective lr becomes 1e-3 * 2e-4 = 2e-7. Measured over 30 steps on the
#     tiny model, max|delta| for lora_B was 1.12e-05 under adafactor against 2.74e-03 under
#     AdamW -- 245x less, i.e. LoRA would silently not train.
#     So: full -> adafactor lr 2e-4 (the reference's). Adapters -> AdamW lr 1e-4 (the
#     harness's proven setting, so those rows differ from the others only in (a)).
#
# Step budget follows the reference's max_steps 10000 rather than a fixed epoch count, so
# every dataset gets the same amount of optimisation:
#     cirr  28,225 triplets / 32 = 882 steps/epoch -> 12 epochs = 10,584
#     fiq   18,000            / 32 = 562           -> 18 epochs = 10,116
#     shoes  8,990            / 32 = 280           -> 36 epochs = 10,080
#
# `full text_vision` needs the 80GB card (batch 32 measured at ~39GB), or GRAD_CKPT=1 to fit a
# 3g.40gb slice: checkpointing is verified to give bit-identical gradients and leaves the
# contrastive negative pool untouched, at ~30% more time.
recipe() {
    MODEL_TAG="magiclens_large_${DATASET}_faithful"
    OUT_NAME="magiclens_peft_faithful"
    EVAL_BATCH=256
    BATCH=32
    PATIENCE=8
    DEFAULT_LOSS=asymmetric      # the reference's own loss_q2t
    case "$DATASET" in
        cirr)  EPOCHS=12 ;;
        fiq)   EPOCHS=18 ;;
        shoes) EPOCHS=36 ;;
        *)     EPOCHS=12 ;;
    esac
    if [ "$METHOD" = "full" ]; then
        OPTIMIZER=adafactor; LR=2e-4
    else
        OPTIMIZER=adamw;     LR=1e-4
    fi
    EXTRA="--temperature 0.07 --weight_decay 0 --warmup_steps 300 --min_lr 2e-6"
    # not `[ ... ] && ...`: a false test as the function's last command would make `recipe`
    # return 1, which `set -e` in the caller turns into an abort
    if [ "${GRAD_CKPT:-0}" = "1" ]; then
        EXTRA="$EXTRA --grad_checkpointing"
    fi
}
