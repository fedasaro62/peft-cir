# Converts a released MagicLens Flax checkpoint (resources/pretrained/magiclens/*.pkl) into
# PyTorch state dicts: the two HuggingFace CLIP towers plus the MagicLens head of head.py.
#
# Reading the .pkl needs flax, which the lincir venv does not have, so run the CLI with the
# magiclens venv (it has flax, numpy and torch):
#
#   .venv/magiclens/bin/python src/peft_cir/models/magiclens/convert.py \
#       --checkpoint resources/pretrained/magiclens/magic_lens_clip_base.pkl
#
# It imports nothing from peft_cir, so it runs in that venv without the package installed.
#
# `convert` itself only needs numpy and torch, so the mapping is testable in the lincir venv.
# Everything about the architecture is inferred from tensor shapes and stored in the result's
# `meta`, so nothing here or in head.py hardcodes base/large dimensions.

# `from __future__` keeps the modern annotations importable under the magiclens venv's
# Python 3.9, which is where the conversion CLI has to run (flax lives there).
from __future__ import annotations

import argparse
import math
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch

FLAX_ROOT = "params"
LOGIT_SCALE_KEY = f"{FLAX_ROOT}/clip/logit_scale"

# (flax suffix, torch suffix, transform) for one CLIP transformer block. CLIP's q/k/v kernels
# are (D, N, H) while its out kernel is (N, H, D) -- the two need different reshapes.
_CLIP_BLOCK = [
    ("attn/query/kernel", "self_attn.q_proj.weight", "qkv_kernel"),
    ("attn/query/bias", "self_attn.q_proj.bias", "flatten"),
    ("attn/key/kernel", "self_attn.k_proj.weight", "qkv_kernel"),
    ("attn/key/bias", "self_attn.k_proj.bias", "flatten"),
    ("attn/value/kernel", "self_attn.v_proj.weight", "qkv_kernel"),
    ("attn/value/bias", "self_attn.v_proj.bias", "flatten"),
    ("attn/out/kernel", "self_attn.out_proj.weight", "out_kernel"),
    ("attn/out/bias", "self_attn.out_proj.bias", "copy"),
    ("ln_1/scale", "layer_norm1.weight", "copy"),
    ("ln_1/bias", "layer_norm1.bias", "copy"),
    ("ln_2/scale", "layer_norm2.weight", "copy"),
    ("ln_2/bias", "layer_norm2.bias", "copy"),
    ("mlp/c_fc/kernel", "mlp.fc1.weight", "transpose"),
    ("mlp/c_fc/bias", "mlp.fc1.bias", "copy"),
    ("mlp/c_proj/kernel", "mlp.fc2.weight", "transpose"),
    ("mlp/c_proj/bias", "mlp.fc2.bias", "copy"),
]

# the head keeps flax's (D, N, H) layout, so all of it converts by copy
_HEAD_LAYER = [
    ("layer_norm/scale", "layer_norm.scale"),
    ("layer_norm/bias", "layer_norm.bias"),
    ("self_attention/query/w", "self_attention.query.w"),
    ("self_attention/query/b", "self_attention.query.b"),
    ("self_attention/key/w", "self_attention.key.w"),
    ("self_attention/key/b", "self_attention.key.b"),
    ("self_attention/value/w", "self_attention.value.w"),
    ("self_attention/value/b", "self_attention.value.b"),
    ("self_attention/post/w", "self_attention.post.w"),
    ("self_attention/post/b", "self_attention.post.b"),
    ("ff_layer/layer_norm/scale", "ff_layer.layer_norm.scale"),
    ("ff_layer/layer_norm/bias", "ff_layer.layer_norm.bias"),
    ("ff_layer/ffn_layer1/linear/w", "ff_layer.ffn_layer1.w"),
    ("ff_layer/ffn_layer1/bias/b", "ff_layer.ffn_layer1.b"),
    ("ff_layer/ffn_layer2/linear/w", "ff_layer.ffn_layer2.w"),
    ("ff_layer/ffn_layer2/bias/b", "ff_layer.ffn_layer2.b"),
]

_POOLER = [
    ("pooling_attn_query", "pooler.pooling_attn_query"),
    ("pool_attn/query/w", "pooler.pool_attn.query.w"),
    ("pool_attn/query/b", "pooler.pool_attn.query.b"),
    ("pool_attn/key/w", "pooler.pool_attn.key.w"),
    ("pool_attn/key/b", "pooler.pool_attn.key.b"),
    ("pool_attn/value/w", "pooler.pool_attn.value.w"),
    ("pool_attn/value/b", "pooler.pool_attn.value.b"),
    ("pool_attn/post/w", "pooler.pool_attn.post.w"),
    ("pool_attn/post/b", "pooler.pool_attn.post.b"),
    ("pool_attn/per_dim_scale/per_dim_scale", "pooler.pool_attn.per_dim_scale.per_dim_scale"),
    ("pool_attn_ln/scale", "pooler.pool_attn_ln.scale"),
    ("pool_attn_ln/bias", "pooler.pool_attn_ln.bias"),
]


def flatten_params(tree: Mapping[str, Any], prefix: str = "") -> dict[str, np.ndarray]:
    """Flatten a nested flax parameter tree into ``scope/scope/name`` keys."""
    flat: dict[str, np.ndarray] = {}
    for key, value in tree.items():
        path = f"{prefix}/{key}" if prefix else str(key)
        if isinstance(value, Mapping):
            flat.update(flatten_params(value, path))
        else:
            flat[path] = np.asarray(value)
    return flat


def _count_indexed(flat: Mapping[str, np.ndarray], pattern: str) -> int:
    regex = re.compile(pattern)
    return len({int(m.group(1)) for k in flat if (m := regex.search(k))})


def infer_meta(flat: Mapping[str, np.ndarray]) -> dict[str, int]:
    """Read every architecture dimension off the checkpoint's tensor shapes."""

    def shape(key: str) -> tuple[int, ...]:
        if key not in flat:
            raise ValueError(f"checkpoint is missing {key!r}, cannot infer the architecture")
        return tuple(np.asarray(flat[key]).shape)

    vocab_size, text_width = shape(f"{FLAX_ROOT}/clip/text/token_embedding/embedding")
    vision_positions, vision_width = shape(f"{FLAX_ROOT}/clip/visual/positional_embedding")
    patch_size = shape(f"{FLAX_ROOT}/clip/visual/conv1/kernel")[0]
    grid = math.isqrt(vision_positions - 1)
    if 1 + grid * grid != vision_positions:
        raise ValueError(f"{vision_positions} position embeddings is not 1 + a square grid")

    return {
        "text_width": text_width,
        "text_heads": shape(f"{FLAX_ROOT}/clip/text/transformer/resblocks_0/attn/query/kernel")[1],
        "text_layers": _count_indexed(flat, r"clip/text/transformer/resblocks_(\d+)/"),
        "text_intermediate": shape(f"{FLAX_ROOT}/clip/text/transformer/resblocks_0/mlp/c_fc/kernel")[1],
        "vocab_size": vocab_size,
        "context_length": shape(f"{FLAX_ROOT}/clip/text/positional_embedding")[0],
        "vision_width": vision_width,
        "vision_heads": shape(f"{FLAX_ROOT}/clip/visual/transformer/resblocks_0/attn/query/kernel")[1],
        "vision_layers": _count_indexed(flat, r"clip/visual/transformer/resblocks_(\d+)/"),
        "vision_intermediate": shape(f"{FLAX_ROOT}/clip/visual/transformer/resblocks_0/mlp/c_fc/kernel")[1],
        "patch_size": patch_size,
        "image_size": grid * patch_size,
        "embed_dim": shape(f"{FLAX_ROOT}/clip/text/text_projection/kernel")[1],
        "head_layers": _count_indexed(flat, r"multimodal_encoder/x_layers_(\d+)/"),
        "head_heads": shape(f"{FLAX_ROOT}/multimodal_encoder/x_layers_0/self_attention/query/w")[1],
        "head_ff": shape(f"{FLAX_ROOT}/multimodal_encoder/x_layers_0/ff_layer/ffn_layer1/linear/w")[1],
        "num_query_tokens": shape(f"{FLAX_ROOT}/contrastive_multimodal_pooler/pooling_attn_query")[0],
    }


def _entries(meta: Mapping[str, int]) -> list[tuple[str, str, str, str]]:
    """(flax key, bucket, torch key, transform) for every parameter in the checkpoint."""
    entries: list[tuple[str, str, str, str]] = [
        (f"{FLAX_ROOT}/clip/text/token_embedding/embedding", "clip_text",
         "text_model.embeddings.token_embedding.weight", "copy"),
        (f"{FLAX_ROOT}/clip/text/positional_embedding", "clip_text",
         "text_model.embeddings.position_embedding.weight", "copy"),
        (f"{FLAX_ROOT}/clip/text/ln_final/scale", "clip_text", "text_model.final_layer_norm.weight", "copy"),
        (f"{FLAX_ROOT}/clip/text/ln_final/bias", "clip_text", "text_model.final_layer_norm.bias", "copy"),
        (f"{FLAX_ROOT}/clip/text/text_projection/kernel", "clip_text", "text_projection.weight", "transpose"),
        (f"{FLAX_ROOT}/clip/visual/class_embedding", "clip_vision",
         "vision_model.embeddings.class_embedding", "copy"),
        (f"{FLAX_ROOT}/clip/visual/conv1/kernel", "clip_vision",
         "vision_model.embeddings.patch_embedding.weight", "conv"),
        (f"{FLAX_ROOT}/clip/visual/positional_embedding", "clip_vision",
         "vision_model.embeddings.position_embedding.weight", "copy"),
        (f"{FLAX_ROOT}/clip/visual/ln_pre/scale", "clip_vision", "vision_model.pre_layrnorm.weight", "copy"),
        (f"{FLAX_ROOT}/clip/visual/ln_pre/bias", "clip_vision", "vision_model.pre_layrnorm.bias", "copy"),
        (f"{FLAX_ROOT}/clip/visual/ln_post/scale", "clip_vision", "vision_model.post_layernorm.weight", "copy"),
        (f"{FLAX_ROOT}/clip/visual/ln_post/bias", "clip_vision", "vision_model.post_layernorm.bias", "copy"),
        (f"{FLAX_ROOT}/clip/visual/proj/kernel", "clip_vision", "visual_projection.weight", "transpose"),
        (LOGIT_SCALE_KEY, "scalar", "logit_scale", "copy"),
    ]

    towers = [("text", "clip_text", "text_model", meta["text_layers"]),
              ("visual", "clip_vision", "vision_model", meta["vision_layers"])]
    for flax_tower, bucket, torch_tower, layers in towers:
        for i in range(layers):
            for flax_suffix, torch_suffix, kind in _CLIP_BLOCK:
                entries.append((
                    f"{FLAX_ROOT}/clip/{flax_tower}/transformer/resblocks_{i}/{flax_suffix}",
                    bucket, f"{torch_tower}.encoder.layers.{i}.{torch_suffix}", kind))

    for i in range(meta["head_layers"]):
        for flax_suffix, torch_suffix in _HEAD_LAYER:
            entries.append((f"{FLAX_ROOT}/multimodal_encoder/x_layers_{i}/{flax_suffix}",
                            "head", f"encoder.layers.{i}.{torch_suffix}", "copy"))
    for flax_suffix, torch_suffix in _POOLER:
        entries.append((f"{FLAX_ROOT}/contrastive_multimodal_pooler/{flax_suffix}",
                        "head", torch_suffix, "copy"))
    return entries


def _transform(array: np.ndarray, kind: str) -> torch.Tensor:
    tensor = torch.from_numpy(np.asarray(array, dtype=np.float32).copy())
    if kind == "copy":
        return tensor
    if kind == "transpose":
        return tensor.t().contiguous()
    if kind == "qkv_kernel":  # (D, N, H) -> Linear weight (N*H, D)
        d = tensor.shape[0]
        return tensor.reshape(d, -1).t().contiguous()
    if kind == "out_kernel":  # (N, H, D) -> Linear weight (D, N*H)
        d = tensor.shape[-1]
        return tensor.reshape(-1, d).t().contiguous()
    if kind == "flatten":  # (N, H) -> (N*H,)
        return tensor.reshape(-1).contiguous()
    if kind == "conv":  # (H, W, I, O) -> (O, I, H, W)
        return tensor.permute(3, 2, 0, 1).contiguous()
    raise ValueError(f"unknown transform {kind!r}")


def convert(tree: Mapping[str, Any]) -> dict[str, Any]:
    """Convert a flax parameter tree (nested or flattened) to PyTorch state dicts.

    Returns a dict with ``clip_text``, ``clip_vision`` and ``head`` state dicts, the scalar
    ``logit_scale`` and the inferred ``meta``. Raises if the checkpoint has keys this mapping
    does not know, or lacks keys it expects: a silently dropped tensor would surface much later
    as a quietly wrong retrieval score.
    """
    flat = flatten_params(tree)
    meta = infer_meta(flat)
    entries = _entries(meta)

    expected = {flax_key for flax_key, _, _, _ in entries}
    if missing := sorted(expected - set(flat)):
        raise ValueError(f"checkpoint is missing {len(missing)} expected key(s): {missing[:5]}")
    if unexpected := sorted(set(flat) - expected):
        raise ValueError(f"checkpoint has {len(unexpected)} unrecognized key(s): {unexpected[:5]}")

    out: dict[str, Any] = {"clip_text": {}, "clip_vision": {}, "head": {}, "meta": meta}
    for flax_key, bucket, torch_key, kind in entries:
        tensor = _transform(flat[flax_key], kind)
        if bucket == "scalar":
            out[torch_key] = tensor
        else:
            out[bucket][torch_key] = tensor
    return out


def load_flax_checkpoint(path: str | Path) -> dict[str, np.ndarray]:
    """Read a released MagicLens .pkl into a flat parameter dict (needs flax)."""
    import pickle

    from flax import serialization

    with open(path, "rb") as handle:
        model_bytes = pickle.load(handle)
    return flatten_params(serialization.msgpack_restore(model_bytes))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parents[4]
    parser.add_argument("--checkpoint", required=True, help="released MagicLens .pkl")
    parser.add_argument("--output", default=None,
                        help="destination .pt (default: <checkpoint stem>_torch.pt beside it)")
    args = parser.parse_args()

    checkpoint = Path(args.checkpoint)
    output = Path(args.output) if args.output else checkpoint.with_name(f"{checkpoint.stem}_torch.pt")
    print(f"reading {checkpoint.relative_to(root) if checkpoint.is_relative_to(root) else checkpoint}")
    converted = convert(load_flax_checkpoint(checkpoint))
    torch.save(converted, output)
    counts = {k: len(v) for k, v in converted.items() if isinstance(v, dict) and k != "meta"}
    print(f"wrote {output} ({counts})")
    for key, value in converted["meta"].items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
