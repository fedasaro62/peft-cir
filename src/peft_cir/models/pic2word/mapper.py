# Pic2Word's image-to-pseudo-word mapping network (`IM2TEXT` upstream, model/model.py in
# google-research/composed_image_retrieval).
#
# Pic2Word trains ONLY this module: CLIP stays frozen and stock, and composition happens by
# splicing the module's output into a text prompt at a placeholder token. So this file plus a
# checkpoint is the whole model, which makes it the lightest arch in this benchmark.
#
# It is deliberately NOT a reuse of searle's Phi, despite both holding three Linear layers.
# Upstream's block is `Linear -> Dropout -> ReLU` where Phi's is `Linear -> GELU -> Dropout`:
# a different activation and a different order, hence a different function. Reusing Phi would
# silently compute something Pic2Word never trained.

from collections.abc import Mapping

import torch
import torch.nn as nn

MIDDLE_DIM = 512  # upstream default (src/params.py --middle_dim)
N_LAYER = 2       # upstream default (src/params.py --n-layer)
DROPOUT = 0.1     # upstream default (IM2TEXT.__init__)


class Pic2WordMapper(nn.Module):
    """Maps a CLIP image embedding to one pseudo-word token embedding.

    A transcription of upstream's ``IM2TEXT``: ``n_layer`` blocks of
    ``Linear -> Dropout -> ReLU``, then a separate ``fc_out`` projection. The module and
    parameter names match upstream (``layers.<i>.0.*``, ``fc_out.*``) so the released
    checkpoint loads without renaming.

    Args:
        embed_dim: CLIP joint embedding dim, the input width (768 for ViT-L/14).
        middle_dim: hidden width.
        output_dim: width of the emitted token embedding (768, the token-embedding width).
        n_layer: number of hidden blocks.
        dropout: dropout probability inside each block.
    """

    def __init__(self, embed_dim: int = 512, middle_dim: int = MIDDLE_DIM,
                 output_dim: int = 512, n_layer: int = N_LAYER, dropout: float = DROPOUT):
        super().__init__()
        self.fc_out = nn.Linear(middle_dim, output_dim)
        layers = []
        dim = embed_dim
        for _ in range(n_layer):
            layers.append(nn.Sequential(nn.Linear(dim, middle_dim), nn.Dropout(dropout),
                                        nn.ReLU()))
            dim = middle_dim
        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return self.fc_out(x)


def infer_meta(state: Mapping[str, torch.Tensor]) -> dict:
    """Read the mapper's dimensions off a state dict's tensor shapes.

    Nothing in this port hardcodes ViT-L/14's numbers: a checkpoint for a different backbone
    loads by virtue of its own shapes, and a malformed one fails loudly here rather than
    silently reshaping later.
    """
    blocks = sorted({int(k.split(".")[1]) for k in state if k.startswith("layers.")})
    if not blocks or "fc_out.weight" not in state:
        raise ValueError(f"not a Pic2Word img2text state dict (keys: {sorted(state)[:6]})")
    first = state[f"layers.{blocks[0]}.0.weight"]
    return {"embed_dim": first.shape[1], "middle_dim": first.shape[0],
            "output_dim": state["fc_out.weight"].shape[0], "n_layer": len(blocks),
            "dropout": DROPOUT}


def build_mapper(meta: Mapping[str, int]) -> Pic2WordMapper:
    """A mapper sized by ``infer_meta``'s output."""
    return Pic2WordMapper(embed_dim=meta["embed_dim"], middle_dim=meta["middle_dim"],
                          output_dim=meta["output_dim"], n_layer=meta["n_layer"],
                          dropout=meta.get("dropout", DROPOUT))
