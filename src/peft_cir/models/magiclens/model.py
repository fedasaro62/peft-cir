"""MagicLens: retrieval through a shared multimodal head (ICML'24).

What differs from the pseudo-word archs is where retrieval happens. SEARLE-XL and Pic2Word
compose in CLIP's *text* space and index candidates as CLIP *image* embeddings. MagicLens has
a head of its own that both sides pass through::

    query     = head(image_embeds(reference), text_embeds(instruction))
    candidate = head(image_embeds(candidate),  text_embeds(""))

so a candidate vector depends on the (always trainable) head -- which is why the harness asks
for ``candidates_from_features`` instead of normalizing a cached feature table itself.

The head plays the role Phi plays for SEARLE-XL, and is loaded from the converted checkpoint
together with the CLIP towers MagicLens fine-tuned (see ``convert.py``).
"""

from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from transformers import CLIPImageProcessor

from peft_cir import PRETRAINED
from peft_cir.base import PeftRetriever
from peft_cir.encoders import build_preprocess, encode_text, towers_from_meta
from peft_cir.models.magiclens.head import MagicLensHead, build_head

MODEL_SIZES = ["base", "large"]
# one checkpoint per model size, so the default carries a {size} placeholder
DEFAULT_CHECKPOINT = str(PRETRAINED / "magiclens" / "magic_lens_clip_{size}_torch.pt")


class PeftMagicLens(PeftRetriever):
    """MagicLens wrapped with a selectable PEFT method (the head is always trainable).

    ``PROMPT_TEMPLATE`` stays None: MagicLens was trained on the bare instruction.
    """

    MAPPER_ATTR = "head"

    def post_init(self) -> None:
        # the empty instruction that turns the head into a candidate encoder; kept as token ids
        # rather than an embedding because the text tower itself changes during training
        self.register_buffer("null_tokens", self.tokenize([""]), persistent=False)

    def text_features(self, tokens: torch.Tensor) -> torch.Tensor:
        """CLIP text embeddings, via the shared encoder so soft prompts apply here too."""
        return encode_text(self.text_model, tokens, None, self.prefix_embeds)

    def candidates_from_features(self, image_features: torch.Tensor) -> torch.Tensor:
        """Finish a candidate from cached image features: the head with an empty instruction."""
        null_text = self.text_features(self.null_tokens.to(image_features.device))
        fused = self.head(image_features, null_text.expand(image_features.shape[0], -1))
        return F.normalize(fused, dim=-1)

    def encode_query(self, tokens: torch.Tensor, ref_image_features: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.head(ref_image_features, self.text_features(tokens)), dim=-1)


def resolve_checkpoint(clip_model_name: str, checkpoint: str | None = None) -> Path:
    """Pick the converted checkpoint for ``clip_model_name``.

    A path carrying a ``{size}`` placeholder (as the default does) is filled in from the model
    size; an explicit path is used as given.
    """
    if clip_model_name not in MODEL_SIZES:
        raise ValueError(f"magiclens has no {clip_model_name!r} checkpoint "
                         f"(want one of {MODEL_SIZES})")
    path = str(checkpoint or DEFAULT_CHECKPOINT)
    return Path(path.format(size=clip_model_name) if "{size}" in path else path)


def load_base(checkpoint) -> tuple[Any, CLIPImageProcessor, Any, MagicLensHead, float]:
    """Load the CLIP towers, the head and the temperature from a converted checkpoint."""
    converted = torch.load(checkpoint, map_location="cpu")
    meta = converted["meta"]
    vision_model, text_model = towers_from_meta(meta)
    vision_model.load_state_dict(converted["clip_vision"], strict=True)
    text_model.load_state_dict(converted["clip_text"], strict=True)
    head = build_head(meta)
    head.load_state_dict(converted["head"], strict=True)
    return (vision_model, build_preprocess(meta["image_size"]), text_model, head,
            float(converted["logit_scale"]))


def build(clip_model_name: str, checkpoint: str | None, method: str, target: str,
          cache_dir: str | None = None, **hparams) -> tuple[PeftMagicLens, CLIPImageProcessor]:
    del cache_dir  # no HuggingFace download: every weight comes from the converted checkpoint
    path = resolve_checkpoint(clip_model_name, checkpoint)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found; convert the released MagicLens checkpoint first:\n"
            f"  .venv/magiclens/bin/python -m peft_cir.models.magiclens.convert --checkpoint "
            f"resources/pretrained/magiclens/magic_lens_clip_{clip_model_name}.pkl")
    vision_model, preprocess, text_model, head, logit_scale = load_base(path)
    # the checkpoint's own temperature is the default, but an explicit logit_scale_init wins --
    # the escape hatch for starting every arch from the same temperature
    hparams.setdefault("logit_scale_init", logit_scale)
    return PeftMagicLens(vision_model, text_model, method, target, mapper=head,
                         **hparams), preprocess
