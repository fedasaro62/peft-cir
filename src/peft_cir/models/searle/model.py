"""SEARLE-XL: zero-shot CIR by textual inversion (ICCV'23).

A frozen CLIP pair plus ``Phi``, an MLP mapping a CLIP image embedding to one pseudo-word
token embedding, which is spliced into ``"a photo of $ that {caption}"`` at the ``$`` position
and encoded by the text tower. Candidates are plain CLIP image embeddings.

Phi is also LinCIR's mechanism -- the two differ only in how the released checkpoint was
trained -- so pointing ``--phi_checkpoint`` at a LinCIR checkpoint loads here unchanged.
"""

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from peft_cir import PRETRAINED
from peft_cir.base import PeftRetriever
from peft_cir.encoders import build_clip_towers, encode_text

DEFAULT_CHECKPOINT = str(PRETRAINED / "searle" / "SEARLE_ViT-L14.pt")
PSEUDO_TOKEN_ID = 259  # id of "$" in the CLIP tokenizer
PROMPT_TEMPLATE = "a photo of $ that {caption}"
PHI_DROPOUT = 0.5


class Phi(nn.Module):
    """Textual-inversion network: CLIP image features -> one pseudo-word embedding.

    Transcribed from SEARLE's ``src/phi.py`` (``Linear -> GELU -> Dropout`` blocks), so the
    released checkpoint's ``Phi`` entry loads without renaming.
    """

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, dropout: float):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.GELU(), nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class PeftSearle(PeftRetriever):
    """SEARLE-XL wrapped with a selectable PEFT method (Phi always trainable)."""

    PROMPT_TEMPLATE = PROMPT_TEMPLATE
    PSEUDO_TOKEN_ID = PSEUDO_TOKEN_ID
    MAPPER_ATTR = "phi"

    def encode_query(self, tokens: torch.Tensor, ref_image_features: torch.Tensor) -> torch.Tensor:
        pseudo = self.phi(ref_image_features)
        composed = encode_text(self.text_model, tokens, pseudo, self.prefix_embeds,
                               pseudo_token_id=self.PSEUDO_TOKEN_ID)
        return F.normalize(composed, dim=-1)


def load_base(clip_model_name: str, phi_checkpoint: str,
              cache_dir: str | None = None) -> tuple[Any, Any, Any, Phi]:
    """Stock CLIP towers plus Phi from a released checkpoint. No tower weights are replaced."""
    vision_model, preprocess, text_model = build_clip_towers(clip_model_name, cache_dir)
    phi = Phi(input_dim=text_model.config.projection_dim,
              hidden_dim=text_model.config.projection_dim * 4,
              output_dim=text_model.config.hidden_size, dropout=PHI_DROPOUT)
    phi.load_state_dict(torch.load(phi_checkpoint, map_location="cpu")["Phi"])
    return vision_model, preprocess, text_model, phi


def build(clip_model_name: str, checkpoint: str | None, method: str, target: str,
          cache_dir: str | None = None, **hparams) -> tuple[PeftSearle, Any]:
    path = checkpoint or DEFAULT_CHECKPOINT
    if not Path(path).exists():
        raise FileNotFoundError(f"{path} not found; fetch SEARLE-XL's Phi checkpoint first "
                                f"(see src/peft_cir/models/searle/README.md)")
    vision_model, preprocess, text_model, phi = load_base(clip_model_name, path, cache_dir)
    return PeftSearle(vision_model, text_model, method, target, mapper=phi,
                      context_length=77, **hparams), preprocess
