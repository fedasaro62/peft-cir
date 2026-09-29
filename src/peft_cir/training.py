"""Objective, optimizer and schedule -- everything the training loop needs that is not a model.

Nothing here is architecture-specific: the four retrievers are trained by the same loss on the
same in-batch negatives, which is what makes their PEFT columns comparable.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

LOSSES = ["symmetric", "asymmetric"]
OPTIMIZERS = ["adamw", "adafactor"]
# the attribute each arch stores its mapping network under: Phi for SEARLE-XL, the
# image-to-pseudo-word mapper for Pic2Word, the multimodal head for MagicLens. MTI has none --
# it adapts the two towers directly -- so it simply never matches.
MAPPER_ATTRS = ("phi", "mapper", "head")


def contrastive_loss(logits: torch.Tensor, labels: torch.Tensor, loss: str) -> torch.Tensor:
    """In-batch InfoNCE over a temperature-scaled (query, candidate) similarity matrix.

    Args:
        logits: shape ``(B, B)``; row i holds query i's similarity to every candidate in the
            batch, so the diagonal is the positive pair.
        labels: ``torch.arange(B)``, the diagonal's index for each row.
        loss: ``"symmetric"`` averages the query->candidate direction with its
            candidate->query transpose (CLIP's usual loss, and what every arch trained with
            before this option existed). ``"asymmetric"`` uses query->candidate alone, matching
            MagicLens's own paper.

    Returns:
        A scalar loss.
    """
    assert loss in LOSSES, loss
    if loss == "symmetric":
        return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels))
    return F.cross_entropy(logits, labels)


def loss_tag_suffix(loss: str) -> str:
    """Checkpoint-dir / CSV-method suffix for a non-default loss; "" for "symmetric".

    Keeps ``"symmetric"`` runs at their original, pre-existing paths (``lora_text_vision``),
    and gives every other loss its own (``lora_text_vision_asymmetric``) so a re-run never
    overwrites the baseline it is being compared against.
    """
    return "" if loss == "symmetric" else f"_{loss}"


def scaled_logits(query: torch.Tensor, candidates: torch.Tensor, logit_scale: torch.Tensor,
                  temperature: float | None = None) -> torch.Tensor:
    """The (query, candidate) similarity matrix, temperature-scaled.

    Args:
        query: [B, D], L2-normalized.
        candidates: [B, D], L2-normalized.
        logit_scale: the model's learned log-temperature, used when ``temperature`` is None.
        temperature: a FIXED temperature, dividing the similarities. This is what MagicLens's
            own training does (``logits = q @ t.T / 0.07``); its ``clip/logit_scale`` is never
            read by the reference at all. ``None`` (the default) keeps the harness's learned,
            clamped scale, as every existing run used.

    Returns:
        [B, B] logits whose diagonal is the positive pair.
    """
    sim = query @ candidates.T
    if temperature is not None:
        return sim / temperature
    return logit_scale.exp().clamp(max=100.0) * sim


def mapper_param_groups(model: nn.Module, lr: float,
                        mapper_lr: float | None = None) -> list[dict]:
    """AdamW parameter groups, optionally putting the mapping network on its own learning rate.

    One learning rate has to serve two very different things under ``--method full``: the
    pretrained CLIP towers, which tolerate only a small step before the representation
    degrades, and the mapping network, which is this arch's task head and needs a large one.
    Splitting them lets each get what it wants.

    Args:
        model: the wrapped retriever.
        lr: learning rate for everything outside the mapping network.
        mapper_lr: learning rate for the mapping network. ``None`` (the default) returns a
            single group, so a run that does not ask for this is identical to one from before
            it existed -- as is any arch with no mapping network at all.

    Returns:
        A list of AdamW param-group dicts.
    """
    trainable = [p for p in model.parameters() if p.requires_grad]
    if mapper_lr is None:
        return [{"params": trainable, "lr": lr}]

    mapper = next((m for m in (getattr(model, a, None) for a in MAPPER_ATTRS)
                   if isinstance(m, nn.Module)), None)
    mapper_ids = {id(p) for p in mapper.parameters() if p.requires_grad} if mapper else set()
    if not mapper_ids:
        return [{"params": trainable, "lr": lr}]
    return [{"params": [p for p in trainable if id(p) not in mapper_ids], "lr": lr},
            {"params": [p for p in trainable if id(p) in mapper_ids], "lr": mapper_lr}]


def build_optimizer(name: str, groups: list[dict], lr: float, weight_decay: float):
    """AdamW (the harness default) or Adafactor matched to the MagicLens reference.

    The reference trains with ``optax.adafactor(learning_rate=schedule)``, whose defaults are
    ``multiply_by_parameter_scale=True``, ``clipping_threshold=1.0``, ``momentum=None`` and
    ``weight_decay_rate=None``. So its step is *relative* to each parameter's own RMS and
    clipped, where AdamW's is ~lr in absolute terms regardless of weight scale -- which is why
    the reference can fully fine-tune CLIP at lr 2e-4 while AdamW at 1e-5 damages the towers.
    ``transformers.Adafactor`` reproduces those defaults exactly with ``scale_parameter=True``
    and ``relative_step=False`` (so an external LR schedule drives it).
    """
    assert name in OPTIMIZERS, name
    if name == "adamw":
        return torch.optim.AdamW(groups, lr=lr, weight_decay=weight_decay)
    from transformers.optimization import Adafactor
    return Adafactor(groups, lr=lr, weight_decay=weight_decay, scale_parameter=True,
                     relative_step=False, warmup_init=False, clip_threshold=1.0,
                     decay_rate=-0.8, beta1=None)


def warmup_cosine(total_steps: int, warmup_steps: int, min_frac: float = 0.0):
    """LambdaLR multiplier: linear warmup to 1.0, then cosine decay to ``min_frac``.

    With ``min_frac=0`` this is exactly the schedule every existing run used. ``min_frac``
    exists because the reference's cosine decays to a floor (``min_lr`` 2e-6 against a 2e-4
    peak) rather than to zero.
    """
    def f(step: int) -> float:
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        t = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return min_frac + (1.0 - min_frac) * 0.5 * (1.0 + math.cos(math.pi * min(t, 1.0)))
    return f


def clip_gradients(params: list[torch.Tensor], max_norm: float) -> None:
    """Clip ``params``' gradients to ``max_norm`` in L2 norm; ``max_norm <= 0`` disables it.

    Off by default, so a run that does not ask for it is bit-identical to one from before this
    existed. Full fine-tuning is the arm that wants it: it is the only method whose trainable
    set is the pretrained CLIP towers themselves, where one outsized gradient moves weights
    that every adapter method leaves frozen.
    """
    if max_norm > 0:
        torch.nn.utils.clip_grad_norm_(params, max_norm)
