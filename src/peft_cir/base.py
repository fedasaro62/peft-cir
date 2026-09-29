"""The shared retriever wrapper every architecture subclasses.

All four retrievers freeze the same way, attach adapters the same way, checkpoint the same way
and index candidates the same way -- they differ only in how a query is composed. This class
holds everything in the first group; a subclass supplies its prompt, its mapping network and
its :meth:`PeftRetriever.encode_query`.
"""

import math

import clip
import torch
import torch.nn as nn
import torch.nn.functional as F

from peft_cir.adapters import METHODS, TARGETS, attach_peft


class PeftRetriever(nn.Module):
    """A CIR retriever wrapped with a selectable PEFT method.

    Subclasses set the class attributes below and implement :meth:`encode_query`. Parameter
    names are part of the checkpoint format -- ``trainable_state_dict`` keys are what
    ``best.pt`` stores -- so the mapping network is registered under :attr:`MAPPER_ATTR`
    verbatim.

    Attributes:
        PROMPT_TEMPLATE: ``str.format`` template with a ``{caption}`` field, or ``None`` for
            architectures trained on the bare caption (MagicLens, MTI).
        PSEUDO_TOKEN_ID: the placeholder row a pseudo-word is spliced onto.
        MAPPER_ATTR: attribute name for the always-trainable mapping network, or ``None`` for
            an architecture that has none (MTI).
    """

    PROMPT_TEMPLATE: str | None = None
    PSEUDO_TOKEN_ID: int = 259
    MAPPER_ATTR: str | None = None

    def __init__(self, vision_model, text_model, method: str, target: str,
                 mapper: nn.Module | None = None, rank: int = 16, alpha: int = 32,
                 dropout: float = 0.05, adapter_dim: int = 64, num_virtual_tokens: int = 10,
                 logit_scale_init: float = math.log(1.0 / 0.07),
                 context_length: int | None = None):
        super().__init__()
        assert method in METHODS, method
        assert target in TARGETS, target
        if method == "prompt" and target != "text":
            raise ValueError("prompt tuning is text-stream only; it has no vision variant")

        self.vision_model = vision_model
        self.text_model = text_model
        if self.MAPPER_ATTR is not None:
            setattr(self, self.MAPPER_ATTR, mapper)
        self.method = method
        self.target = target
        self.num_virtual_tokens = num_virtual_tokens
        self.context_length = context_length or text_model.config.max_position_embeddings
        self.logit_scale = nn.Parameter(torch.tensor(logit_scale_init))
        self.text_adapters = None
        self.vision_adapters = None
        self.soft_prompt = None

        # start fully frozen, then enable per method
        for p in self.parameters():
            p.requires_grad_(False)

        if method == "prompt":
            self.soft_prompt = nn.Parameter(
                torch.randn(num_virtual_tokens, text_model.config.hidden_size) * 0.02)
            self.context_length -= num_virtual_tokens
        else:
            self.text_adapters, self.vision_adapters = attach_peft(
                self.vision_model, self.text_model, method, target, rank=rank, alpha=alpha,
                dropout=dropout, adapter_dim=adapter_dim)

        # the mapping network and the temperature are always trainable; `frozen` is excluded
        # because it is eval-only -- it is each paper's own zero-shot model
        if method != "frozen":
            for p in (self.mapper_module.parameters() if self.mapper_module is not None else []):
                p.requires_grad_(True)
            self.logit_scale.requires_grad_(True)

        self.post_init()

    def post_init(self) -> None:
        """Hook for subclass setup that needs a fully built model (e.g. tokenizing)."""

    @property
    def mapper_module(self) -> nn.Module | None:
        return getattr(self, self.MAPPER_ATTR, None) if self.MAPPER_ATTR else None

    # --- encoding ---------------------------------------------------------------
    def tokenize(self, captions: list[str]) -> torch.Tensor:
        """Wrap captions in this architecture's prompt and tokenize to its context length."""
        if self.PROMPT_TEMPLATE is not None:
            captions = [self.PROMPT_TEMPLATE.format(caption=c) for c in captions]
        return clip.tokenize(captions, context_length=self.context_length, truncate=True)

    @property
    def prefix_embeds(self) -> torch.Tensor | None:
        return self.soft_prompt if self.method == "prompt" else None

    def image_features(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Raw CLIP image embeddings: the mapper's input, and the cacheable feature table."""
        return self.vision_model(pixel_values=pixel_values).image_embeds

    def candidates_from_features(self, image_features: torch.Tensor) -> torch.Tensor:
        """Finish a candidate vector from cached CLIP image features.

        For the pseudo-word architectures a candidate *is* the normalized image embedding, so
        this is the identity beyond normalization; MagicLens overrides it with its multimodal
        head. The trainer calls it instead of normalizing the cached table itself, so archs
        whose candidate path is trainable stay correct.
        """
        return F.normalize(image_features, dim=-1)

    def encode_candidates(self, pixel_values: torch.Tensor) -> torch.Tensor:
        return self.candidates_from_features(self.image_features(pixel_values))

    def encode_query(self, tokens: torch.Tensor, ref_image_features: torch.Tensor) -> torch.Tensor:
        """Compose one query into a unit vector. Every architecture defines this differently."""
        raise NotImplementedError

    # --- checkpoint (trainable subset only) ------------------------------------
    def trainable_parameters(self) -> list[nn.Parameter]:
        return [p for p in self.parameters() if p.requires_grad]

    def num_trainable(self) -> int:
        return sum(p.numel() for p in self.trainable_parameters())

    def trainable_state_dict(self) -> dict:
        return {n: p.detach().cpu() for n, p in self.named_parameters() if p.requires_grad}

    def load_trainable(self, state_dict: dict) -> None:
        own = dict(self.named_parameters())
        for n, v in state_dict.items():
            own[n].data.copy_(v.to(own[n].device))
