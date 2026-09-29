"""Adapt one retriever to one dataset with one PEFT method.

    python tools/train.py --arch pic2word --dataset cirr --method lora --target text_vision

Writes ``<output>/<method>_<target>[_<loss>]/{best.pt,meta.json}``, where ``best.pt`` holds
only the trainable tensors and ``meta.json`` records everything needed to rebuild the model --
which is what ``tools/evaluate.py --peft_checkpoint`` reads back.
"""

import json
from argparse import ArgumentParser
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from peft_cir import ROOT
from peft_cir.adapters import adalora_step, enable_gradient_checkpointing, set_adalora_total_step
from peft_cir.data.benchmarks import VAL_FRACTION
from peft_cir.data.datasets import SELECTION_KEYS, build_train, build_val, validate
from peft_cir.registry import ARCHS, DEFAULT_CHECKPOINT, build_model, resolve_arch
from peft_cir.training import (
    LOSSES,
    OPTIMIZERS,
    build_optimizer,
    clip_gradients,
    contrastive_loss,
    loss_tag_suffix,
    mapper_param_groups,
    scaled_logits,
    warmup_cosine,
)

DATASETS = ["cirr", "fiq", "shoes"]
METHODS = ["full", "lora", "dora", "ia3", "adapter", "prompt", "adalora", "vera"]


def parse_args():
    p = ArgumentParser(description=__doc__)
    p.add_argument("--arch", default="searle", choices=ARCHS,
                   help="which retriever to adapt.")
    p.add_argument("--method", required=True, choices=METHODS)
    p.add_argument("--target", default="text", choices=["text", "vision", "text_vision"],
                   help="which CLIP towers receive the adapter; the mapping network and the "
                        "temperature are always trainable. `vision` is the ablation arm that "
                        "leaves the text encoder untouched.")
    p.add_argument("--loss", default="symmetric", choices=LOSSES,
                   help="in-batch contrastive loss. symmetric (default): 0.5*(query->candidate "
                        "+ candidate->query), CLIP's usual loss and what every existing "
                        "checkpoint was trained with. asymmetric: query->candidate only, "
                        "matching MagicLens's own paper.")
    p.add_argument("--dataset", default="cirr", choices=DATASETS,
                   help="that benchmark's train split from --data_root. FashionIQ pools its "
                        "three subtasks; shoes carves val out of train with a seeded, "
                        "image-disjoint split.")
    p.add_argument("--data_root", default=str(ROOT / "data"),
                   help="parent dir holding cirr/, fashioniq/, shoes/.")
    p.add_argument("--protocol", default="carved", choices=["carved", "literature"],
                   help="shoes only. carved: val carved out of train, report test once "
                        "(default, no test-set selection). literature: train on the full train "
                        "split and select on TEST, reproducing ARTEMIS, whose released code has "
                        "no val split. 'literature' numbers are best-epoch-on-test and are "
                        "optimistic.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--clip_model_name", default="large", choices=["base", "large", "huge", "giga"],
                   help="CLIP backbone. searle/pic2word take large/huge/giga; magiclens ships "
                        "base (ViT-B/16) and large (ViT-L/14); mti only large.")
    p.add_argument("--phi_checkpoint", default=None,
                   help="mapping-network checkpoint; defaults to the one --arch ships with.")
    p.add_argument("--model_tag", default="searle_large",
                   help="label stored in the checkpoint meta and used as the CSV 'model' column.")
    p.add_argument("--cache_dir", default=str(ROOT / "resources" / "pretrained" / "hf_models"))
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--grad_clip", type=float, default=0.0,
                   help="clip the global gradient norm before each step. 0 (default) disables "
                        "clipping; full fine-tuning is the arm that wants it.")
    p.add_argument("--grad_checkpointing", action="store_true",
                   help="recompute encoder activations during backward instead of retaining "
                        "them. ~30%% slower but cuts activation memory enough for batch 32 to "
                        "fit a 40GB slice; leaves the contrastive negative pool unchanged.")
    p.add_argument("--optimizer", default="adamw", choices=OPTIMIZERS,
                   help="adamw (default) is the harness's. adafactor matches the MagicLens "
                        "reference: update scaled by each parameter's RMS and clipped.")
    p.add_argument("--temperature", type=float, default=None,
                   help="FIXED contrastive temperature, as the MagicLens reference uses (0.07). "
                        "None (default) keeps the model's learned logit_scale clamped at 100. "
                        "Setting this freezes logit_scale.")
    p.add_argument("--warmup_steps", type=int, default=None,
                   help="absolute warmup steps. None (default) uses 5%% of total steps.")
    p.add_argument("--min_lr", type=float, default=0.0,
                   help="floor the cosine decays to. 0 (default) decays to zero.")
    p.add_argument("--mapper_lr", type=float, default=None,
                   help="separate learning rate for the mapping network. None (default) puts "
                        "everything in one group. Set it for --method full, where one lr cannot "
                        "serve both the pretrained towers and the task head.")
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--eval_batch_size", type=int, default=64)
    p.add_argument("--rank", type=int, default=16)
    p.add_argument("--alpha", type=int, default=32)
    p.add_argument("--dropout", type=float, default=0.05)
    p.add_argument("--adapter_dim", type=int, default=64)
    p.add_argument("--num_virtual_tokens", type=int, default=10)
    p.add_argument("--output", default=str(ROOT / "outputs" / "CIRR" / "searle_peft"))
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def build_data(args, model, preprocess, device):
    """Returns ``(train_dataset, validate_fn)`` for the requested dataset."""
    literature = args.protocol == "literature"
    train_ds = build_train(args.dataset, args.data_root, preprocess, model.tokenize,
                           val_frac=0.0 if literature else VAL_FRACTION)
    # literature protocol: no held-out val exists, so selection happens on test -- exactly what
    # ARTEMIS does, and the reason those numbers are optimistic.
    bench = build_val(args.dataset, args.data_root, "test" if literature else "val")
    if literature:
        print("*** protocol=literature: selecting the best checkpoint on the TEST split. "
              "Reported numbers are best-epoch-on-test and are NOT comparable to the "
              "carved-val protocol. ***")
    def _validate() -> dict:
        return validate(args.dataset, model, preprocess, bench, device, args.eval_batch_size)

    return train_ds, _validate


def run_epoch(model, loader, optim, sched, args, device, global_step: int):
    """One training epoch; returns ``(mean loss, next global step)``."""
    model.train()
    running = 0.0
    for batch in tqdm(loader, desc="train"):
        ref_pix, tgt_pix, tokens = (b.to(device) for b in batch)
        ref_feats = model.image_features(ref_pix)
        targets = model.encode_candidates(tgt_pix)
        q = model.encode_query(tokens, ref_feats)
        logits = scaled_logits(q, targets, model.logit_scale, args.temperature)
        loss = contrastive_loss(logits, torch.arange(q.shape[0], device=device), args.loss)
        optim.zero_grad()
        loss.backward()
        clip_gradients(model.trainable_parameters(), args.grad_clip)
        optim.step()
        adalora_step(model, global_step)  # rank reallocation; no-op for other methods
        sched.step()
        global_step += 1
        running += loss.item()
    return running / max(1, len(loader)), global_step


def checkpoint_meta(args, metrics: dict, select_key: str, epoch: int, n_trainable: int) -> dict:
    """Everything needed to rebuild this model, plus the numbers that selected it."""
    literature = args.dataset == "shoes" and args.protocol == "literature"
    return {
        "method": args.method, "target": args.target, "loss": args.loss, "lr": args.lr,
        "weight_decay": args.weight_decay, "grad_clip": args.grad_clip,
        "mapper_lr": args.mapper_lr, "optimizer": args.optimizer,
        "grad_checkpointing": args.grad_checkpointing, "temperature": args.temperature,
        "warmup_steps": args.warmup_steps, "min_lr": args.min_lr,
        "batch_size": args.batch_size, "model_tag": args.model_tag, "arch": args.arch,
        "clip_model_name": args.clip_model_name,
        "phi_checkpoint": args.phi_checkpoint or DEFAULT_CHECKPOINT[resolve_arch(args.arch)],
        "rank": args.rank, "alpha": args.alpha, "dropout": args.dropout,
        "adapter_dim": args.adapter_dim, "num_virtual_tokens": args.num_virtual_tokens,
        "dataset": args.dataset, "selection_metric": select_key, "protocol": args.protocol,
        "selected_on": "test" if literature else "val",
        "best_epoch": epoch, "val_score": metrics[select_key],
        "val_recall@1": metrics.get("Recall@1"), "trainable_params": n_trainable,
    }


def main() -> None:
    args = parse_args()
    device = args.device
    torch.manual_seed(args.seed)

    print(f"Building {args.arch} + {args.method} ({args.target})...")
    model, preprocess = build_model(
        args.arch, args.clip_model_name, args.phi_checkpoint, args.method, args.target,
        cache_dir=args.cache_dir, rank=args.rank, alpha=args.alpha, dropout=args.dropout,
        adapter_dim=args.adapter_dim, num_virtual_tokens=args.num_virtual_tokens)
    model.to(device)
    n_trainable = model.num_trainable()
    print(f"Trainable parameters: {n_trainable:,}")

    train_ds, validate_fn = build_data(args, model, preprocess, device)
    select_key = SELECTION_KEYS.get(args.dataset, "Recall@1")
    loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4,
                        drop_last=True)
    print(f"{args.dataset}: {len(train_ds)} train triplets, selecting on {select_key}")

    if args.grad_checkpointing:
        print(f"gradient checkpointing on: {enable_gradient_checkpointing(model)}")
    if args.temperature is not None:
        # a fixed temperature makes logit_scale dead weight; freeze it so it is not an
        # untrained parameter sitting in the optimizer with no gradient
        model.logit_scale.requires_grad_(False)
        print(f"fixed temperature {args.temperature} (logit_scale frozen)")

    # one group unless --mapper_lr asks for two; the scheduler scales each group's own base lr
    # by the same factor, so warmup/cosine apply to both proportionally
    groups = mapper_param_groups(model, args.lr, args.mapper_lr)
    optim = build_optimizer(args.optimizer, groups, args.lr, args.weight_decay)
    if len(groups) > 1:
        print(f"param groups: {len(groups[0]['params'])} tensors @ lr={args.lr} (towers), "
              f"{len(groups[1]['params'])} @ lr={args.mapper_lr} (mapping network)")
    total_steps = args.epochs * max(1, len(loader))
    warmup = args.warmup_steps if args.warmup_steps is not None else int(0.05 * total_steps)
    sched = torch.optim.lr_scheduler.LambdaLR(
        optim, warmup_cosine(total_steps, warmup, args.min_lr / args.lr if args.lr else 0.0))
    print(f"{args.optimizer}: {total_steps} steps, warmup {warmup}, "
          f"lr {args.lr} -> {args.min_lr}, weight_decay {args.weight_decay}")
    set_adalora_total_step(model, total_steps)  # no-op unless method == adalora

    ckpt_dir = Path(args.output) / f"{args.method}_{args.target}{loss_tag_suffix(args.loss)}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_score, best_epoch, bad, global_step = -1.0, -1, 0, 0

    for epoch in range(args.epochs):
        loss, global_step = run_epoch(model, loader, optim, sched, args, device, global_step)
        metrics = validate_fn()
        score = metrics[select_key]
        print(f"epoch {epoch}: loss={loss:.4f}  val {select_key}={score:.2f}")

        if score <= best_score:
            bad += 1
            if bad >= args.patience:
                print(f"early stop at epoch {epoch} "
                      f"(best {select_key}={best_score:.2f} @ epoch {best_epoch})")
                break
            continue

        best_score, best_epoch, bad = score, epoch, 0
        torch.save(model.trainable_state_dict(), ckpt_dir / "best.pt")
        (ckpt_dir / "meta.json").write_text(json.dumps(
            checkpoint_meta(args, metrics, select_key, epoch, n_trainable), indent=2))

    print(f"Best val {select_key} = {best_score:.2f} (epoch {best_epoch}); checkpoint in {ckpt_dir}")


if __name__ == "__main__":
    main()
