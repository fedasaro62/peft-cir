# Converts the released MTI checkpoint (the "Pretrain like Your Inference: Masked Tuning"
# model, upstream repo Chen-Junyang-cn/PLI) from OpenAI-CLIP layout into the HuggingFace CLIP
# layout the rest of this benchmark uses.
#
# MTI ships no head of its own: masked tuning is a pretraining procedure whose product is a set
# of fine-tuned CLIP ViT-L/14 weights (the released checkpoint's `compositor` entry is empty),
# and composition at inference is a parameter-free weighted sum. So unlike magiclens_convert.py
# there is no head to port -- only a tower relabelling, needed because `peft` targets HF module
# names (q_proj/k_proj/v_proj/out_proj) while OpenAI's CLIP fuses QKV into one `in_proj_weight`
# inside nn.MultiheadAttention.
#
#   .venv/lincir/bin/python -m peft_cir.models.mti.convert \
#       --checkpoint resources/pretrained/mti/best.pth
#
# `--stock` instead converts unmodified OpenAI CLIP ViT-L/14 through the identical mapping,
# producing the control arm that isolates what masked tuning bought.
#
# Every architecture dimension is inferred from tensor shapes and stored in the result's `meta`,
# so nothing here or in mti_peft.py hardcodes ViT-L/14's numbers.

import argparse
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch

from peft_cir import ROOT

# MTI's own key prefixes: the released image_encoder nests the tower under `visual_encoder.`,
# while its text_encoder is stored bare (as OpenAI CLIP stores the text side).
MTI_VISION_PREFIX = "visual_encoder."
STOCK_VISION_PREFIX = "visual."
# a masked-pretraining artifact: the learned token substituted for masked patches. Unused at
# inference, so deliberately dropped rather than silently ignored.
DROPPED_VISION_KEYS = ("vision_mask_token",)
CLIP_HEAD_DIM = 64  # OpenAI CLIP derives head counts as width // 64 for both towers

# (source suffix, target suffix, transform) for one transformer block, shared by both towers.
# OpenAI CLIP's c_fc/c_proj are nn.Linear, already [out, in] like HF's fc1/fc2, so they copy.
_BLOCK = [
    ("attn.in_proj_weight", "self_attn.q_proj.weight", "qkv:0"),
    ("attn.in_proj_weight", "self_attn.k_proj.weight", "qkv:1"),
    ("attn.in_proj_weight", "self_attn.v_proj.weight", "qkv:2"),
    ("attn.in_proj_bias", "self_attn.q_proj.bias", "qkv:0"),
    ("attn.in_proj_bias", "self_attn.k_proj.bias", "qkv:1"),
    ("attn.in_proj_bias", "self_attn.v_proj.bias", "qkv:2"),
    ("attn.out_proj.weight", "self_attn.out_proj.weight", "copy"),
    ("attn.out_proj.bias", "self_attn.out_proj.bias", "copy"),
    ("ln_1.weight", "layer_norm1.weight", "copy"),
    ("ln_1.bias", "layer_norm1.bias", "copy"),
    ("ln_2.weight", "layer_norm2.weight", "copy"),
    ("ln_2.bias", "layer_norm2.bias", "copy"),
    ("mlp.c_fc.weight", "mlp.fc1.weight", "copy"),
    ("mlp.c_fc.bias", "mlp.fc1.bias", "copy"),
    ("mlp.c_proj.weight", "mlp.fc2.weight", "copy"),
    ("mlp.c_proj.bias", "mlp.fc2.bias", "copy"),
]

# note HF's own misspelling of `pre_layrnorm`, which the state dict key has to match
_VISION_TOP = [
    ("class_embedding", "vision_model.embeddings.class_embedding", "copy"),
    ("conv1.weight", "vision_model.embeddings.patch_embedding.weight", "copy"),
    ("positional_embedding", "vision_model.embeddings.position_embedding.weight", "copy"),
    ("ln_pre.weight", "vision_model.pre_layrnorm.weight", "copy"),
    ("ln_pre.bias", "vision_model.pre_layrnorm.bias", "copy"),
    ("ln_post.weight", "vision_model.post_layernorm.weight", "copy"),
    ("ln_post.bias", "vision_model.post_layernorm.bias", "copy"),
    ("proj", "visual_projection.weight", "transpose"),
]

_TEXT_TOP = [
    ("token_embedding.weight", "text_model.embeddings.token_embedding.weight", "copy"),
    ("positional_embedding", "text_model.embeddings.position_embedding.weight", "copy"),
    ("ln_final.weight", "text_model.final_layer_norm.weight", "copy"),
    ("ln_final.bias", "text_model.final_layer_norm.bias", "copy"),
    ("text_projection", "text_projection.weight", "transpose"),
]


def _count_blocks(state: Mapping[str, torch.Tensor]) -> int:
    return 1 + max(int(k.split(".")[2]) for k in state if k.startswith("transformer.resblocks."))


def infer_meta(vision: Mapping[str, torch.Tensor],
               text: Mapping[str, torch.Tensor]) -> dict[str, int]:
    """Derive every architecture dimension from tensor shapes.

    Nothing about ViT-L/14 is assumed, so a ViT-B checkpoint would convert unchanged.
    """
    vision_width, _, _, patch_size = vision["conv1.weight"].shape
    num_patches = vision["positional_embedding"].shape[0] - 1  # minus the class token
    grid = round(num_patches ** 0.5)
    if grid * grid != num_patches:
        raise ValueError(f"vision position embedding is not a square grid + 1: {num_patches + 1}")
    text_width, embed_dim = text["text_projection"].shape
    return {
        "embed_dim": embed_dim,
        "image_size": grid * patch_size,
        "patch_size": patch_size,
        "vision_width": vision_width,
        "vision_layers": _count_blocks(vision),
        "vision_heads": vision_width // CLIP_HEAD_DIM,
        "vision_intermediate": vision["transformer.resblocks.0.mlp.c_fc.weight"].shape[0],
        "vocab_size": text["token_embedding.weight"].shape[0],
        "context_length": text["positional_embedding"].shape[0],
        "text_width": text_width,
        "text_layers": _count_blocks(text),
        "text_heads": text_width // CLIP_HEAD_DIM,
        "text_intermediate": text["transformer.resblocks.0.mlp.c_fc.weight"].shape[0],
    }


def _entries(meta: Mapping[str, int]) -> list[tuple[str, str, str, str]]:
    """(source key, bucket, target key, transform) for every tensor in both towers."""
    entries: list[tuple[str, str, str, str]] = []
    for src, dst, kind in _VISION_TOP:
        entries.append((src, "clip_vision", dst, kind))
    for src, dst, kind in _TEXT_TOP:
        entries.append((src, "clip_text", dst, kind))
    towers = [("clip_vision", "vision_model", meta["vision_layers"]),
              ("clip_text", "text_model", meta["text_layers"])]
    for bucket, prefix, layers in towers:
        for i in range(layers):
            for src, dst, kind in _BLOCK:
                entries.append((f"transformer.resblocks.{i}.{src}", bucket,
                                f"{prefix}.encoder.layers.{i}.{dst}", kind))
    return entries


def _transform(tensor: torch.Tensor, kind: str) -> torch.Tensor:
    if kind == "copy":
        return tensor.clone()
    if kind == "transpose":
        return tensor.t().contiguous()
    if kind.startswith("qkv:"):
        third = tensor.shape[0] // 3
        i = int(kind.split(":")[1])
        return tensor[i * third:(i + 1) * third].clone()
    raise ValueError(f"unknown transform {kind!r}")


def convert(vision: Mapping[str, torch.Tensor],
            text: Mapping[str, torch.Tensor]) -> dict[str, Any]:
    """Convert prefix-stripped OpenAI-CLIP tower state dicts to HF layout.

    Returns ``clip_vision`` / ``clip_text`` state dicts plus the inferred ``meta``. Raises if
    either tower has keys this mapping does not know, or lacks keys it expects: a silently
    dropped tensor would surface much later as a quietly wrong retrieval score.
    """
    meta = infer_meta(vision, text)
    entries = _entries(meta)
    sources = {"clip_vision": dict(vision), "clip_text": dict(text)}

    for bucket, state in sources.items():
        expected = {src for src, b, _, _ in entries if b == bucket}
        if missing := sorted(expected - set(state)):
            raise ValueError(f"{bucket} is missing {len(missing)} expected key(s): {missing[:5]}")
        if unexpected := sorted(set(state) - expected):
            raise ValueError(f"{bucket} has {len(unexpected)} unrecognized key(s): {unexpected[:5]}")

    out: dict[str, Any] = {"clip_vision": {}, "clip_text": {}, "meta": meta}
    for src, bucket, dst, kind in entries:
        out[bucket][dst] = _transform(sources[bucket][src], kind)
    return out


def _split_prefixed(state: Mapping[str, torch.Tensor], prefix: str) -> dict[str, torch.Tensor]:
    return {k[len(prefix):]: v for k, v in state.items() if k.startswith(prefix)}


def load_mti_checkpoint(path: str | Path) -> tuple[dict[str, torch.Tensor],
                                                   dict[str, torch.Tensor]]:
    """Read the released MTI ``best.pth`` into (vision, text) tower state dicts.

    The file is a training checkpoint -- model weights, optimizer moments and a step counter --
    so only ``model_state_dict`` is read, and its empty ``compositor`` entry is asserted rather
    than assumed: a future checkpoint that did carry head weights must not convert silently.
    """
    raw = torch.load(path, map_location="cpu", weights_only=True)
    model = raw["model_state_dict"]
    if compositor := model.get("compositor"):
        raise ValueError(
            f"this checkpoint's compositor has {len(compositor)} tensor(s); MTI is expected to "
            "have none (composition is a parameter-free weighted sum), so mti_peft.py has no "
            "module to load them into")
    image_encoder = {k: v for k, v in model["image_encoder"].items()
                     if k not in DROPPED_VISION_KEYS}
    if leftover := sorted(k for k in image_encoder if not k.startswith(MTI_VISION_PREFIX)):
        raise ValueError(f"unexpected non-tower key(s) in image_encoder: {leftover}")
    return _split_prefixed(image_encoder, MTI_VISION_PREFIX), dict(model["text_encoder"])


def load_stock_checkpoint(name: str = "ViT-L/14") -> tuple[dict[str, torch.Tensor],
                                                           dict[str, torch.Tensor]]:
    """Read unmodified OpenAI CLIP into (vision, text) tower state dicts, for the control arm."""
    import clip

    model, _ = clip.load(name, device="cpu", jit=False)
    state = {k: v.float() for k, v in model.state_dict().items()}
    vision = _split_prefixed(state, STOCK_VISION_PREFIX)
    text = {k: v for k, v in state.items()
            if not k.startswith(STOCK_VISION_PREFIX) and k != "logit_scale"}
    return vision, text


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    root = ROOT
    default_dir = root / "resources" / "pretrained" / "mti"
    parser.add_argument("--checkpoint", default=str(default_dir / "best.pth"),
                        help="released MTI best.pth (ignored with --stock)")
    parser.add_argument("--stock", action="store_true",
                        help="convert unmodified OpenAI CLIP ViT-L/14 instead: the control arm "
                             "that isolates what masked tuning bought")
    parser.add_argument("--output", default=None, help="destination .pt")
    args = parser.parse_args()

    if args.stock:
        print("reading stock OpenAI CLIP ViT-L/14")
        vision, text = load_stock_checkpoint()
        output = Path(args.output or default_dir / "mti_clip_large_stock.pt")
    else:
        print(f"reading {args.checkpoint}")
        vision, text = load_mti_checkpoint(args.checkpoint)
        output = Path(args.output or default_dir / "mti_clip_large_torch.pt")

    converted = convert(vision, text)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(converted, output)
    counts = {k: len(v) for k, v in converted.items() if k != "meta"}
    print(f"wrote {output} ({counts})")
    for key, value in converted["meta"].items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
