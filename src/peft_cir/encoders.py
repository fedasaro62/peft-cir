"""CLIP tower construction and text encoding, shared by every architecture.

Two ways to get a tower pair: :func:`build_clip_towers` downloads stock OpenAI CLIP (what
SEARLE-XL and Pic2Word adapt, since both leave the backbone untouched upstream), and
:func:`towers_from_meta` builds empty towers shaped by a converted checkpoint (what MagicLens
and MTI need, since both ship their own fine-tuned weights). Either way the result is a
HuggingFace module pair, which is what makes :func:`peft_cir.adapters.attach_peft` apply
unchanged across all four.
"""


import torch
from transformers import (
    CLIPImageProcessor,
    CLIPTextConfig,
    CLIPTextModelWithProjection,
    CLIPVisionConfig,
    CLIPVisionModelWithProjection,
)

CLIP_IMAGE_MEAN = [0.48145466, 0.4578275, 0.40821073]
CLIP_IMAGE_STD = [0.26862954, 0.26130258, 0.27577711]
CLIP_LAYER_NORM_EPS = 1e-5  # scenic's CLIP LayerNorm matches PyTorch's default, not flax's

CLIP_MODELS = {
    "large": "openai/clip-vit-large-patch14",
    "huge": "laion/CLIP-ViT-H-14-laion2B-s32B-b79K",
    "giga": "Geonmo/CLIP-Giga-config-fixed",
}


def build_preprocess(image_size: int = 224) -> CLIPImageProcessor:
    """CLIP preprocessing at ``image_size``, shared by every arch.

    Deliberately not each paper's own transform (MagicLens rescales by the per-image maximum
    and resizes bilinearly; MTI uses CLIP4Cir's targetpad): every arch in the benchmark should
    see identical pixels, or a preprocessing difference confounds the cross-arch comparison.
    """
    return CLIPImageProcessor(crop_size={"height": image_size, "width": image_size},
                              do_center_crop=True, do_convert_rgb=True, do_normalize=True,
                              do_rescale=True, do_resize=True, image_mean=CLIP_IMAGE_MEAN,
                              image_std=CLIP_IMAGE_STD, resample=3,
                              size={"shortest_edge": image_size})


def build_clip_towers(clip_model_name: str, cache_dir: str | None = None,
                      fp16: bool = False) -> tuple[CLIPVisionModelWithProjection,
                                                   CLIPImageProcessor,
                                                   CLIPTextModelWithProjection]:
    """Stock pretrained CLIP towers plus the shared preprocessing, from HuggingFace."""
    if clip_model_name not in CLIP_MODELS:
        raise ValueError(f"unknown CLIP backbone {clip_model_name!r} (want one of {list(CLIP_MODELS)})")
    name = CLIP_MODELS[clip_model_name]
    dtype = torch.float16 if fp16 else torch.float32
    vision = CLIPVisionModelWithProjection.from_pretrained(name, torch_dtype=dtype, cache_dir=cache_dir)
    text = CLIPTextModelWithProjection.from_pretrained(name, torch_dtype=dtype, cache_dir=cache_dir)
    return vision, build_preprocess(), text


def towers_from_meta(meta) -> tuple[CLIPVisionModelWithProjection, CLIPTextModelWithProjection]:
    """Empty HF CLIP towers shaped by a converted checkpoint's ``meta``.

    Built from configs rather than ``from_pretrained``: MagicLens and MTI both trained CLIP
    end to end, so every weight comes from their own checkpoint and downloading stock CLIP
    would only be overwritten.
    """
    text_config = CLIPTextConfig(
        hidden_size=meta["text_width"], intermediate_size=meta["text_intermediate"],
        num_hidden_layers=meta["text_layers"], num_attention_heads=meta["text_heads"],
        vocab_size=meta["vocab_size"], max_position_embeddings=meta["context_length"],
        projection_dim=meta["embed_dim"], hidden_act="quick_gelu",
        layer_norm_eps=CLIP_LAYER_NORM_EPS)
    vision_config = CLIPVisionConfig(
        hidden_size=meta["vision_width"], intermediate_size=meta["vision_intermediate"],
        num_hidden_layers=meta["vision_layers"], num_attention_heads=meta["vision_heads"],
        image_size=meta["image_size"], patch_size=meta["patch_size"],
        projection_dim=meta["embed_dim"], hidden_act="quick_gelu",
        layer_norm_eps=CLIP_LAYER_NORM_EPS)
    return CLIPVisionModelWithProjection(vision_config), CLIPTextModelWithProjection(text_config)


def causal_mask(shape: torch.Size, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    """CLIP's causal self-attention mask.

    Copy-paste from transformers v4.34.0 ``modeling_clip._make_causal_mask``, which newer
    releases moved; the pin in requirements.txt is what keeps the rest of the text path valid.
    """
    bsz, tgt_len = shape
    mask = torch.full((tgt_len, tgt_len), torch.finfo(dtype).min, device=device)
    cond = torch.arange(mask.size(-1), device=device)
    mask.masked_fill_(cond < (cond + 1).view(mask.size(-1), 1), 0)
    return mask.to(dtype)[None, None, :, :].expand(bsz, 1, tgt_len, tgt_len)


def encode_text(text_model, tokens: torch.Tensor, pseudo_tokens: torch.Tensor | None = None,
                prefix_embeds: torch.Tensor | None = None,
                pseudo_token_id: int = 259) -> torch.Tensor:
    """CLIP text encoding, optionally splicing a pseudo-word and a soft prompt.

    Args:
        text_model: a ``CLIPTextModelWithProjection``.
        tokens: token ids, ``[B, L]``.
        pseudo_tokens: ``[B, D]`` embeddings replacing every ``pseudo_token_id`` position.
            ``None`` skips the substitution, leaving plain CLIP text encoding -- what MagicLens
            and MTI need, having no pseudo-word.
        prefix_embeds: soft prompt, ``[P, D]``, inserted right after BOS. Callers must tokenize
            to length ``77 - P`` so the total stays 77 and the pretrained position embeddings
            still apply.
        pseudo_token_id: the placeholder row; each arch splices at its own (259 for ``$``,
            265 for Pic2Word's ``*``).

    Returns:
        ``[B, D]`` projected text embeddings, unnormalized.
    """
    embeddings = text_model.text_model.embeddings
    x = embeddings.token_embedding(tokens).type(text_model.dtype)
    if pseudo_tokens is not None:
        x = torch.where(tokens.unsqueeze(-1) == pseudo_token_id,
                        pseudo_tokens.unsqueeze(1).type(x.dtype), x)

    eot_idx = tokens.argmax(dim=-1)
    if prefix_embeds is not None:
        prefix = prefix_embeds.unsqueeze(0).expand(x.shape[0], -1, -1).type(x.dtype)
        x = torch.cat([x[:, :1], prefix, x[:, 1:]], dim=1)  # insert after BOS
        eot_idx = eot_idx + prefix_embeds.shape[0]

    n = x.shape[1]
    x = x + embeddings.position_embedding(torch.arange(n, device=x.device).unsqueeze(0))
    x = text_model.text_model.encoder(
        inputs_embeds=x, attention_mask=None,
        causal_attention_mask=causal_mask((x.shape[0], n), x.dtype, x.device),
        return_dict=False)[0]
    x = text_model.text_model.final_layer_norm(x)
    x = x[torch.arange(x.shape[0], device=x.device), eot_idx]
    return text_model.text_projection(x) if hasattr(text_model, "text_projection") else x
