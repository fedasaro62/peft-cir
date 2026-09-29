"""The PEFT methods themselves, wired identically for every architecture.

LoRA / DoRA / (IA)^3 / AdaLoRA / VeRA come from HF ``peft``; the bottleneck adapter and the
soft prompt are small custom modules. :func:`attach_peft` is the single place adapters are
injected, so the methods compared across the four retrievers differ in nothing but the model
they are attached to.
"""


import torch
import torch.nn as nn
from peft import AdaLoraConfig, IA3Config, LoraConfig, VeraConfig, get_peft_model

METHODS = ["frozen", "full", "lora", "dora", "ia3", "adapter", "prompt", "adalora", "vera"]
TARGETS = ["text", "vision", "text_vision"]
CLIP_ATTN_TARGETS = ["q_proj", "k_proj", "v_proj", "out_proj"]
# AdaLoRA needs the total optimiser-step count up front to schedule its rank budget; the
# trainer patches it once the dataloader length is known (see set_adalora_total_step).
ADALORA_PLACEHOLDER_STEPS = 1000
VERA_RANK = 256  # VeRA's shared projections are cheap, so its rank is far above LoRA's
PEFT_INJECTED = ("lora", "dora", "ia3", "adalora", "vera")


class Bottleneck(nn.Module):
    """Houlsby-style residual bottleneck adapter (init near-identity)."""

    def __init__(self, dim: int, bottleneck: int):
        super().__init__()
        self.down = nn.Linear(dim, bottleneck)
        self.act = nn.GELU()
        self.up = nn.Linear(bottleneck, dim)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return h + self.up(self.act(self.down(h)))


def _attach_bottlenecks(clip_module, adapter_dim: int) -> nn.ModuleList:
    """Insert a residual bottleneck after each transformer layer via a forward hook."""
    layers = clip_module.encoder.layers
    dim = layers[0].mlp.fc2.out_features
    adapters = nn.ModuleList([Bottleneck(dim, adapter_dim) for _ in layers])

    def make_hook(adapter):
        def hook(module, inputs, output):
            return (adapter(output[0]),) + tuple(output[1:])
        return hook

    for layer, adapter in zip(layers, adapters):
        layer.register_forward_hook(make_hook(adapter))
    return adapters


def adapter_config(method: str, rank: int, alpha: int, dropout: float):
    """A fresh peft config for one tower; peft mutates config state during injection."""
    if method == "ia3":
        return IA3Config(target_modules=CLIP_ATTN_TARGETS + ["fc2"], feedforward_modules=["fc2"])
    if method == "adalora":
        # init_r > target_r: AdaLoRA starts wide and prunes singular values down
        return AdaLoraConfig(init_r=rank + rank // 2, target_r=rank, lora_alpha=alpha,
                             lora_dropout=dropout, target_modules=CLIP_ATTN_TARGETS,
                             total_step=ADALORA_PLACEHOLDER_STEPS)
    if method == "vera":
        return VeraConfig(r=VERA_RANK, vera_dropout=dropout, target_modules=CLIP_ATTN_TARGETS,
                          d_initial=0.1)
    return LoraConfig(r=rank, lora_alpha=alpha, lora_dropout=dropout,
                      target_modules=CLIP_ATTN_TARGETS, use_dora=(method == "dora"))


def attach_peft(vision_model, text_model, method: str, target: str, rank: int = 16,
                alpha: int = 32, dropout: float = 0.05,
                adapter_dim: int = 64) -> tuple[nn.ModuleList | None, nn.ModuleList | None]:
    """Attach ``method``'s adapter to the towers named by ``target``, in place.

    Both towers must already be fully frozen. ``target`` of ``vision`` leaves the text encoder
    frozen, so the composed query is still shaped by the (always trainable) mapping network
    while the prompt encoding itself is untouched -- the point of the target ablation.

    Returns:
        The (text, vision) bottleneck module lists for the ``adapter`` method, which the caller
        must assign to attributes so they are registered, saved and optimised. Every other
        method mutates the towers in place and returns ``(None, None)``.
    """
    adapt_text = target in ("text", "text_vision")
    adapt_vision = target in ("vision", "text_vision")

    if method in PEFT_INJECTED:
        # get_peft_model injects adapters in place and freezes the base weights
        if adapt_text:
            get_peft_model(text_model, adapter_config(method, rank, alpha, dropout))
        if adapt_vision:
            get_peft_model(vision_model, adapter_config(method, rank, alpha, dropout))
    elif method == "adapter":
        return (_attach_bottlenecks(text_model.text_model, adapter_dim) if adapt_text else None,
                _attach_bottlenecks(vision_model.vision_model, adapter_dim) if adapt_vision else None)
    elif method == "full":
        for model, adapt in ((text_model, adapt_text), (vision_model, adapt_vision)):
            if adapt:
                for p in model.parameters():
                    p.requires_grad_(True)
    # method == "frozen": nothing trainable; the caller unfreezes its mapper and temperature
    return None, None


def set_adalora_total_step(model: nn.Module, total_step: int) -> None:
    """Tell AdaLoRA the true optimiser-step count once the dataloader length is known.

    The rank-allocation schedule (tinit / tfinal / deltaT) is derived from total_step, so a
    stale placeholder would prune on the wrong timetable. No-op for other methods.
    """
    if model.method != "adalora":
        return
    for tower in (model.text_model, model.vision_model):
        for cfg in getattr(tower, "peft_config", {}).values():
            cfg.total_step = total_step
            cfg.tinit = max(1, int(0.1 * total_step))
            cfg.tfinal = max(1, int(0.3 * total_step))
            cfg.deltaT = max(1, total_step // 100)


def adalora_step(model: nn.Module, global_step: int) -> None:
    """AdaLoRA rank reallocation; must run after backward, before the next forward."""
    if model.method != "adalora":
        return
    for tower in (model.text_model, model.vision_model):
        allocate = getattr(getattr(tower, "base_model", None), "update_and_allocate", None)
        if callable(allocate):
            allocate(global_step)


def enable_gradient_checkpointing(model: nn.Module) -> list[str]:
    """Turn on activation checkpointing in both CLIP towers; returns the towers it reached.

    Trades compute for memory inside the encoder stacks: activations are recomputed during
    backward instead of retained. That is what lets batch 32 fit a 40GB MIG slice, where the
    dominant cost is the vision tower's retained activations.

    Safe for an in-batch contrastive objective. Checkpointing operates *within* a tower's
    forward; the [B, B] similarity matrix is built afterwards from the final [B, D] embeddings,
    which are never checkpointed. So all B embeddings still coexist and the negative pool is
    unchanged -- unlike gradient accumulation, which really would shrink it.

    ``enable_input_require_grads`` is the companion fix for the adapter methods: transformers
    4.34 uses *reentrant* checkpointing, which silently produces no gradient for a block whose
    inputs and parameters all have ``requires_grad=False`` -- exactly the case when a frozen
    CLIP tower carries LoRA. Forcing the embedding output to require grad restores the path.
    """
    reached = []
    for name in ("text_model", "vision_model"):
        tower = getattr(model, name, None)
        if tower is None:
            continue
        base = getattr(tower, "base_model", None)
        for obj in (tower, base, getattr(base, "model", None)):
            fn = getattr(obj, "gradient_checkpointing_enable", None)
            if callable(fn):
                fn()
                inp = getattr(obj, "enable_input_require_grads", None)
                if callable(inp):
                    inp()
                reached.append(name)
                break
    return reached
