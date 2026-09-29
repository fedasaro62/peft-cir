"""Architecture registry: the one place an ``--arch`` name maps to a model.

Both entry points dispatch through :func:`build_model`, so adding a retriever means adding one
row here and nothing else.
"""

from typing import Any

from peft_cir.models import magiclens, mti, pic2word, searle

# each entry is (builder, default mapping-network checkpoint). MagicLens and MTI have no
# mapping network to load separately -- both *are* their fine-tuned CLIP weights -- so their
# default is the whole converted checkpoint, with a {size} placeholder filled in from
# --clip_model_name.
_REGISTRY = {
    "searle": (searle.build, searle.DEFAULT_CHECKPOINT),
    "pic2word": (pic2word.build, pic2word.DEFAULT_CHECKPOINT),
    "magiclens": (magiclens.build, magiclens.DEFAULT_CHECKPOINT),
    "mti": (mti.build, mti.DEFAULT_CHECKPOINT),
}

ARCHS = list(_REGISTRY)
DEFAULT_CHECKPOINT = {name: ckpt for name, (_, ckpt) in _REGISTRY.items()}

# SEARLE-XL and LinCIR are the same Phi mechanism with different pretrained weights, and this
# harness registered the pair under "lincir" before the split. Checkpoints written then record
# arch="lincir" in their meta.json, so the name still resolves.
_ALIASES = {"lincir": "searle"}


def resolve_arch(arch: str) -> str:
    """Canonical arch name, following the legacy aliases recorded in older meta.json files."""
    name = _ALIASES.get(arch, arch)
    if name not in _REGISTRY:
        raise ValueError(f"unknown arch {arch!r} (want one of {ARCHS})")
    return name


def build_model(arch: str, clip_model_name: str, mapper_checkpoint: str | None, method: str,
                target: str, cache_dir: str | None = None, **hparams) -> tuple[Any, Any]:
    """Build the wrapped retriever for ``arch``; returns ``(model, preprocess)``.

    ``mapper_checkpoint`` falls back to the architecture's default when None.
    """
    builder, default = _REGISTRY[resolve_arch(arch)]
    return builder(clip_model_name, mapper_checkpoint or default, method, target,
                   cache_dir=cache_dir, **hparams)
