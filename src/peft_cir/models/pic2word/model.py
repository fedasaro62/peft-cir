"""Pic2Word: stock CLIP plus one image-to-pseudo-word mapper (CVPR'23).

The lightest arch in the benchmark. CLIP is never fine-tuned upstream, so the harness's stock
towers are already the correct backbone, and a candidate is a plain CLIP image embedding.

What is genuinely Pic2Word-specific is the prompt. Upstream splices the mapper output at a
``*`` placeholder inside its own template, where SEARLE uses ``$``: the placeholder token id
differs (265 vs 259) and so does the surrounding wording. A wrong prompt would make the arch
wrong rather than merely differently tuned -- the latent bug this study found for Context-I2W
-- so the template and the id are transcribed from upstream ``src/data.py`` and pinned by tests.
"""

from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from peft_cir import PRETRAINED
from peft_cir.base import PeftRetriever
from peft_cir.encoders import build_clip_towers, encode_text
from peft_cir.models.pic2word.mapper import Pic2WordMapper, build_mapper

DEFAULT_CHECKPOINT = str(PRETRAINED / "pic2word" / "pic2word_large.pt")
# upstream's placeholder is "*" (src/eval_utils.py: id_split = tokenize(["*"])[0][1]), not
# SEARLE's "$". Its embedding is overwritten by the mapper output, so only the position
# matters -- but the literal template is kept so the code matches the paper.
PSEUDO_TOKEN = "*"
PSEUDO_TOKEN_ID = 265
# src/data.py. CIRR is the single-caption form; FashionIQ concatenates both captions with cap2
# first. Shoes has no upstream Pic2Word eval, so it takes the single-caption form.
PROMPT_TEMPLATE = "a photo of * , {caption}"


class PeftPic2Word(PeftRetriever):
    """Pic2Word wrapped with a selectable PEFT method (the mapper is always trainable)."""

    PROMPT_TEMPLATE = PROMPT_TEMPLATE
    PSEUDO_TOKEN_ID = PSEUDO_TOKEN_ID
    MAPPER_ATTR = "mapper"

    def encode_query(self, tokens: torch.Tensor, ref_image_features: torch.Tensor) -> torch.Tensor:
        pseudo = self.mapper(ref_image_features)
        composed = encode_text(self.text_model, tokens, pseudo, self.prefix_embeds,
                               pseudo_token_id=self.PSEUDO_TOKEN_ID)
        return F.normalize(composed, dim=-1)


def load_base(clip_model_name: str, checkpoint: str,
              cache_dir: str | None = None) -> tuple[Any, Any, Any, Pic2WordMapper]:
    """Stock CLIP towers plus the converted mapper. No tower weights are replaced."""
    vision_model, preprocess, text_model = build_clip_towers(clip_model_name, cache_dir)
    converted = torch.load(checkpoint, map_location="cpu")
    mapper = build_mapper(converted["meta"])
    mapper.load_state_dict(converted["mapper"], strict=True)
    return vision_model, preprocess, text_model, mapper


def build(clip_model_name: str, checkpoint: str | None, method: str, target: str,
          cache_dir: str | None = None, **hparams) -> tuple[PeftPic2Word, Any]:
    path = checkpoint or DEFAULT_CHECKPOINT
    if not Path(path).exists():
        raise FileNotFoundError(
            f"{path} not found; fetch and convert the released Pic2Word checkpoint first "
            f"(see src/peft_cir/models/pic2word/README.md)")
    vision_model, preprocess, text_model, mapper = load_base(clip_model_name, path, cache_dir)
    return PeftPic2Word(vision_model, text_model, method, target, mapper=mapper,
                        **hparams), preprocess
