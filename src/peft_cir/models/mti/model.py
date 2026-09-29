"""MTI: masked tuning for zero-shot CIR ("Pretrain like Your Inference", upstream PLI).

The study's simplest arch, and its cleanest PEFT arm. Masked tuning is a *pretraining*
procedure -- mask most of a reference image's patches and make (masked image + caption)
retrieve the unmasked image -- and what it ships is a set of fine-tuned CLIP weights, nothing
more: the released checkpoint's ``compositor`` entry is empty. Composition at inference is a
parameter-free weighted sum of the two unimodal features::

    query     = normalize( normalize(text(caption)) + img_weight * normalize(image(reference)) )
    candidate = normalize( image(candidate) )

So where the other archs carry a trainable mapping network, MTI has none: under any PEFT
method the only trainable tensors are the adapters themselves plus the temperature. That makes
it the one arch whose adapter comparison is not confounded by a co-trained mapper.

``img_weight`` stays a fixed hyperparameter (upstream pairs 0.25 with CLIP ViT-L/14) rather
than a learned scalar, so the frozen and PEFT arms differ in exactly one thing -- the adapters
-- and the frozen arm remains the paper's zero-shot model.
"""

from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from transformers import CLIPImageProcessor

from peft_cir import PRETRAINED
from peft_cir.base import PeftRetriever
from peft_cir.encoders import build_preprocess, encode_text, towers_from_meta

# only ViT-L/14 is released for MTI; the {size} placeholder mirrors the magiclens default
MODEL_SIZES = ["large"]
DEFAULT_CHECKPOINT = str(PRETRAINED / "mti" / "mti_clip_{size}_torch.pt")
# upstream's CLIP ViT-L/14 setting: every clip-L command pairs --mask_ratio=0.75 with this
DEFAULT_IMG_WEIGHT = 0.25


class PeftMTI(PeftRetriever):
    """MTI wrapped with a selectable PEFT method (no mapping network to train).

    ``PROMPT_TEMPLATE`` stays None and ``MAPPER_ATTR`` None: MTI was trained on the bare
    caption and has no extra parameters at all.
    """

    def __init__(self, *args, img_weight: float = DEFAULT_IMG_WEIGHT, **kwargs):
        super().__init__(*args, **kwargs)
        self.img_weight = img_weight

    def text_features(self, tokens: torch.Tensor) -> torch.Tensor:
        """CLIP text embeddings, via the shared encoder so soft prompts apply here too."""
        return encode_text(self.text_model, tokens, None, self.prefix_embeds)

    def encode_query(self, tokens: torch.Tensor, ref_image_features: torch.Tensor) -> torch.Tensor:
        """Upstream's weighted sum of separately normalized text and image features.

        The outer normalization does not change the ranking (both sides are normalized before
        the dot product either way), but keeps the contract that every arch returns unit
        vectors.
        """
        text = F.normalize(self.text_features(tokens), dim=-1)
        image = F.normalize(ref_image_features, dim=-1)
        return F.normalize(text + self.img_weight * image, dim=-1)


def resolve_checkpoint(clip_model_name: str, checkpoint: str | None = None) -> Path:
    """Pick the converted checkpoint for ``clip_model_name``.

    A path carrying a ``{size}`` placeholder (as the default does) is filled in from the model
    size; an explicit path is used as given, which is how the stock-CLIP control arm is pointed
    at ``mti_clip_large_stock.pt``.
    """
    path = str(checkpoint or DEFAULT_CHECKPOINT)
    if "{size}" in path:
        if clip_model_name not in MODEL_SIZES:
            raise ValueError(f"mti has no {clip_model_name!r} checkpoint "
                             f"(want one of {MODEL_SIZES})")
        path = path.format(size=clip_model_name)
    return Path(path)


def load_base(checkpoint) -> tuple[Any, CLIPImageProcessor, Any]:
    """Load the CLIP towers from a converted checkpoint (see ``convert.py``)."""
    converted = torch.load(checkpoint, map_location="cpu", weights_only=True)
    meta = converted["meta"]
    vision_model, text_model = towers_from_meta(meta)
    vision_model.load_state_dict(converted["clip_vision"], strict=True)
    text_model.load_state_dict(converted["clip_text"], strict=True)
    return vision_model, build_preprocess(meta["image_size"]), text_model


def build(clip_model_name: str, checkpoint: str | None, method: str, target: str,
          cache_dir: str | None = None, **hparams) -> tuple[PeftMTI, CLIPImageProcessor]:
    del cache_dir  # no HuggingFace download: every weight comes from the converted checkpoint
    path = resolve_checkpoint(clip_model_name, checkpoint)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found; convert the released MTI checkpoint first:\n"
            f"  .venv/lincir/bin/python -m peft_cir.models.mti.convert --checkpoint "
            f"resources/pretrained/mti/best.pth")
    vision_model, preprocess, text_model = load_base(path)
    return PeftMTI(vision_model, text_model, method, target, **hparams), preprocess
