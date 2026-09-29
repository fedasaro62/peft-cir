# PyTorch port of the MagicLens multimodal head (ICML'24), whose reference implementation is
# Flax and lives in src/magiclens. The head takes the CLIP image and text embeddings of one
# query, treats them as a two-token sequence, runs a small pre-LN transformer over it and pools
# the result with one learned query token.
#
# Three conventions are inherited from the Praxis-style layers upstream uses, and none of them
# match the PyTorch defaults:
#   * LayerNorm scales by (1 + scale), not scale -- note that scenic's *CLIP* LayerNorms in the
#     same checkpoint use the ordinary convention, so the two coexist;
#   * attention logits are capped by 50 * tanh(logits / 50) before the softmax;
#   * the pooler scales queries per dimension with softplus(w) * 1.442695 / sqrt(H) instead of
#     the usual H**-0.5, which the encoder does use.
#
# Attention projections keep flax's (D, N, H) parameter layout rather than being folded into
# nn.Linear, so converted weights are a straight copy and stay readable next to the reference.
# The head is never adapted by `peft` (it is this arch's always-trainable mapping network), so
# nothing downstream needs it to be a Linear.

import math
from collections.abc import Mapping
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

LOGIT_CAP = 50.0
R_SOFTPLUS_0 = 1.442695041  # 1 / softplus(0), so an unloaded PerDimScale is a plain 1/sqrt(H)


def cap_logits(logits: torch.Tensor, cap: float = LOGIT_CAP) -> torch.Tensor:
    """Squash attention logits into (-cap, cap), as upstream's `_dot_atten` does."""
    return cap * torch.tanh(logits / cap)


def _lecun_normal(*shape: int, fan_in: int) -> torch.Tensor:
    """flax's default kernel init, so an unloaded head is a usable random map, not zeros.

    Initialization is otherwise irrelevant here -- every deployed head is loaded from the
    released checkpoint -- so the scale is matched, not flax's exact truncated sampler.
    """
    return torch.randn(*shape) / math.sqrt(fan_in)


class MagicLensLayerNorm(nn.Module):
    """LayerNorm with MagicLens's `x * (1 + scale) + bias` parameterization."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.zeros(dim))
        self.bias = nn.Parameter(torch.zeros(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=-1, keepdim=True)
        var = (x - mean).pow(2).mean(dim=-1, keepdim=True)
        normed = (x - mean) * torch.rsqrt(var + self.eps)
        return normed * (1 + self.scale) + self.bias


class PerDimScale(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.per_dim_scale = nn.Parameter(torch.zeros(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = R_SOFTPLUS_0 / math.sqrt(self.dim) * F.softplus(self.per_dim_scale)
        return x * scale


class HeadProjection(nn.Module):
    """Per-head projection in flax's layout: (D, N, H) weights either splitting or merging heads."""

    def __init__(self, input_dim: int, num_heads: int, dim_per_head: int, output_proj: bool = False):
        super().__init__()
        self.output_proj = output_proj
        fan_in = num_heads * dim_per_head if output_proj else input_dim
        self.w = nn.Parameter(_lecun_normal(input_dim, num_heads, dim_per_head, fan_in=fan_in))
        self.b = nn.Parameter(torch.zeros(input_dim if output_proj else (num_heads, dim_per_head)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.output_proj:
            return torch.einsum("...nh,dnh->...d", x, self.w) + self.b
        return torch.einsum("...d,dnh->...nh", x, self.w) + self.b


class HeadAttention(nn.Module):
    """Dot-product attention with capped logits; the pooler variant scales per dimension."""

    def __init__(self, input_dim: int, hidden_dim: int, num_heads: int,
                 use_per_dim_scale: bool = False):
        super().__init__()
        if hidden_dim % num_heads:
            raise ValueError(f"hidden_dim {hidden_dim} is not divisible by num_heads {num_heads}")
        self.dim_per_head = hidden_dim // num_heads
        self.query = HeadProjection(input_dim, num_heads, self.dim_per_head)
        self.key = HeadProjection(input_dim, num_heads, self.dim_per_head)
        self.value = HeadProjection(input_dim, num_heads, self.dim_per_head)
        self.post = HeadProjection(input_dim, num_heads, self.dim_per_head, output_proj=True)
        self.per_dim_scale = PerDimScale(self.dim_per_head) if use_per_dim_scale else None

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        query, key, value = self.query(q), self.key(k), self.value(v)
        query = (self.per_dim_scale(query) if self.per_dim_scale is not None
                 else query * self.dim_per_head ** -0.5)
        logits = cap_logits(torch.einsum("btnh,bsnh->bnts", query, key))
        probs = torch.softmax(logits, dim=-1)
        encoded = torch.einsum("bnts,bsnh->btnh", probs, value)
        return self.post(encoded)


class _Linear(nn.Module):
    """Affine map keeping flax's (in, out) weight orientation, as the FFN weights are stored."""

    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.w = nn.Parameter(_lecun_normal(input_dim, output_dim, fan_in=input_dim))
        self.b = nn.Parameter(torch.zeros(output_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.w + self.b


class TransformerFFN(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.layer_norm = MagicLensLayerNorm(input_dim)
        self.ffn_layer1 = _Linear(input_dim, hidden_dim)
        self.ffn_layer2 = _Linear(hidden_dim, input_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.ffn_layer2(F.relu(self.ffn_layer1(self.layer_norm(x))))
        return x + h


class TransformerLayer(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_heads: int):
        super().__init__()
        self.layer_norm = MagicLensLayerNorm(input_dim)
        self.self_attention = HeadAttention(input_dim, input_dim, num_heads)
        self.ff_layer = TransformerFFN(input_dim, hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        normed = self.layer_norm(x)
        return self.ff_layer(self.self_attention(normed, normed, normed) + x)


class MultimodalEncoder(nn.Module):
    def __init__(self, embed_dim: int, num_layers: int, num_heads: int, ff_hidden: int):
        super().__init__()
        self.layers = nn.ModuleList(
            [TransformerLayer(embed_dim, ff_hidden, num_heads) for _ in range(num_layers)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


class AttenTokenPooler(nn.Module):
    """Attention pooling over the modality tokens, driven by learned query tokens."""

    def __init__(self, embed_dim: int, num_heads: int, num_query_tokens: int = 1):
        super().__init__()
        self.pooling_attn_query = nn.Parameter(
            _lecun_normal(num_query_tokens, embed_dim, fan_in=embed_dim))
        self.pool_attn = HeadAttention(embed_dim, 4 * embed_dim, num_heads, use_per_dim_scale=True)
        self.pool_attn_ln = MagicLensLayerNorm(embed_dim)

    def forward(self, embeds: torch.Tensor) -> torch.Tensor:
        query = self.pooling_attn_query.unsqueeze(0).expand(embeds.shape[0], -1, -1)
        return self.pool_attn_ln(self.pool_attn(query, embeds, embeds))


class MagicLensHead(nn.Module):
    """Fuses one CLIP image embedding and one CLIP text embedding into a retrieval vector."""

    def __init__(self, embed_dim: int, num_layers: int, num_heads: int, ff_hidden: int,
                 num_query_tokens: int = 1):
        super().__init__()
        self.embed_dim = embed_dim
        self.encoder = MultimodalEncoder(embed_dim, num_layers, num_heads, ff_hidden)
        self.pooler = AttenTokenPooler(embed_dim, num_heads, num_query_tokens)

    def forward(self, image_embeds: torch.Tensor, text_embeds: torch.Tensor) -> torch.Tensor:
        """Returns the unnormalized pooled embedding (upstream's ``multimodal_embed``)."""
        tokens = torch.stack([image_embeds, text_embeds], dim=1)
        return self.pooler(self.encoder(tokens))[:, 0]


def build_head(meta: Mapping[str, Any]) -> MagicLensHead:
    return MagicLensHead(embed_dim=meta["embed_dim"], num_layers=meta["head_layers"],
                         num_heads=meta["head_heads"], ff_hidden=meta["head_ff"],
                         num_query_tokens=meta["num_query_tokens"])
