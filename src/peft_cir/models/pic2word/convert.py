# Converts the released Pic2Word checkpoint into a bare `Pic2WordMapper` state dict.
#
# Pic2Word trains only its img2text mapping network against frozen, stock OpenAI CLIP -- verified:
# all 446 CLIP tensors in the release are bit-identical to clip.load("ViT-L/14"), max abs
# deviation 0.0 -- so --
# unlike magiclens_convert.py and mti_convert.py -- there are no CLIP towers to remap. The
# harness's existing stock HF CLIP ViT-L/14 is already the right backbone. All this does is
# find the mapper's tensors inside whatever wrapper the release used and store them next to a
# `meta` dict of dimensions read off their shapes.
#
#   .venv/lincir/bin/python -m peft_cir.models.pic2word.convert \
#       --checkpoint resources/pretrained/pic2word/pic2word_raw.pt

import argparse
from collections.abc import Mapping
from pathlib import Path

import torch

from peft_cir import ROOT
from peft_cir.models.pic2word.mapper import build_mapper, infer_meta

# keys the release might nest the mapper under; upstream saves it as "img2text"
# the released checkpoint uses "state_dict_img2text" with "module."-prefixed keys (verified
# against the 1.72GB release); the others are accepted in case a mirror differs
MAPPER_KEYS = ("img2text", "img2text_state_dict", "state_dict_img2text")
# prefixes to strip -- a DataParallel/DDP save prefixes every key with "module."
STRIP_PREFIXES = ("module.",)


def _strip(state: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    out = {}
    for k, v in state.items():
        for p in STRIP_PREFIXES:
            if k.startswith(p):
                k = k[len(p):]
        out[k] = v
    return out


def find_mapper_state(checkpoint: Mapping) -> dict[str, torch.Tensor]:
    """Locate the img2text tensors inside a released checkpoint.

    Handles the shapes a release plausibly takes: the mapper nested under one of
    ``MAPPER_KEYS``, or a bare state dict that already is the mapper. Raises rather than
    guessing if neither matches, so a layout change fails loudly.
    """
    for key in MAPPER_KEYS:
        if isinstance(checkpoint, Mapping) and key in checkpoint:
            inner = checkpoint[key]
            if isinstance(inner, Mapping):
                return _strip(inner)
    if isinstance(checkpoint, Mapping):
        flat = _strip({k: v for k, v in checkpoint.items() if torch.is_tensor(v)})
        if "fc_out.weight" in flat:
            return flat
    raise ValueError(
        "could not find img2text tensors; top-level keys were "
        f"{sorted(checkpoint)[:12] if isinstance(checkpoint, Mapping) else type(checkpoint)}")


def convert(checkpoint: Mapping) -> dict:
    """``{mapper, meta}``, with every mapper parameter present and nothing left over."""
    state = find_mapper_state(checkpoint)
    meta = infer_meta(state)
    # strict load is the completeness check: it fails on a missing OR an unexpected key
    build_mapper(meta).load_state_dict(state, strict=True)
    return {"mapper": state, "meta": meta}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True, help="released Pic2Word .pt")
    p.add_argument("--output", default=None,
                   help="defaults to resources/pretrained/pic2word/pic2word_large.pt")
    args = p.parse_args()

    # weights_only=False is required and safe enough here: the release pickles numpy scalars
    # (epoch bookkeeping), which torch>=2.6's weights_only loader refuses, and the literal name
    # `numpy.core.multiarray.scalar` cannot be allowlisted because modern numpy resolves it to
    # `numpy._core.*`. The file is the canonical Google Research release for the paper.
    raw = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    converted = convert(raw)
    out = Path(args.output or ROOT / "resources" / "pretrained" / "pic2word" / "pic2word_large.pt")
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(converted, out)
    n = sum(v.numel() for v in converted["mapper"].values())
    print(f"wrote {out}  ({n:,} mapper parameters)")
    print(f"meta: {converted['meta']}")


if __name__ == "__main__":
    main()
