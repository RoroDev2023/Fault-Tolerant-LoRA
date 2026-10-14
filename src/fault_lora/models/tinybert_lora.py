"""Rank-r LoRA on every nn.Linear, including layers narrower than r.

The existing `attach_lora` rejects a rank larger than either matrix dimension.
TinyBERT's SST-2 classifier is 312 → 2, so a declared rank of 4 on all Linear
layers cannot be attached there. `CappedRankLoRALayer` subclasses the unchanged
`LoRALayer` and skips only that check: it keeps the declared rank-r factors
(B: out × r, A: r × in), but their product has rank at most min(r, in, out).
Every layer where r fits uses the original `LoRALayer`. Because the subclass is
a `LoRALayer`, `adapter_state_dict`, `load_adapter_state`, and
`base_state_dict` apply unchanged.
"""

import math

from torch import nn

from fault_lora.models.lora import LoRALayer


class CappedRankLoRALayer(LoRALayer):
    def __init__(self, base, rank, alpha):
        nn.Module.__init__(self)
        if not isinstance(base, nn.Linear):
            raise TypeError("CappedRankLoRALayer supports nn.Linear only")
        if isinstance(rank, bool) or not isinstance(rank, int) or rank < 1:
            raise ValueError("rank must be a positive integer")
        if isinstance(alpha, bool) or not isinstance(alpha, (int, float)) or not math.isfinite(alpha) or alpha <= 0:
            raise ValueError("alpha must be finite and positive")
        self.base = base.requires_grad_(False)
        self.rank, self.alpha, self.scale = rank, float(alpha), float(alpha) / rank
        options = {"device": base.weight.device, "dtype": base.weight.dtype, "bias": False}
        self.lora_A = nn.Linear(base.in_features, rank, **options)
        self.lora_B = nn.Linear(rank, base.out_features, **options)
        # Same initialization as LoRALayer: Kaiming-uniform A, zero B (zero output at start).
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)


def attach_linear_lora(model, rank=4, alpha=4.0):
    """Freeze the whole base and wrap every named nn.Linear, in module order."""
    if any(isinstance(module, LoRALayer) for module in model.modules()):
        raise ValueError("Adapters are already attached")
    names = [name for name, module in model.named_modules() if name and isinstance(module, nn.Linear)]
    if not names:
        raise ValueError("Model has no nn.Linear layers")
    modules = {name: model.get_submodule(name) for name in names}
    # Construct in module order so the seeded Kaiming draws match attach_lora's order.
    wrappers = {}
    for name in names:
        base = modules[name]
        capped = rank > min(base.in_features, base.out_features)
        wrappers[name] = (CappedRankLoRALayer if capped else LoRALayer)(base, rank, alpha)
    model.requires_grad_(False)
    for name, wrapper in wrappers.items():
        parent_name, _, child_name = name.rpartition(".")
        parent = model.get_submodule(parent_name) if parent_name else model
        setattr(parent, child_name, wrapper)
        wrapper.lora_A.requires_grad_(True)
        wrapper.lora_B.requires_grad_(True)
    model.eval()
    return [{"name": name, "kind": "Linear", "adapter_class": type(wrapper).__name__, "rank": rank, "alpha": float(alpha), "scale": wrapper.scale,
             "effective_rank_bound": min(rank, wrapper.base.in_features, wrapper.base.out_features),
             "base_weight_shape": list(wrapper.base.weight.shape), "A_shape": list(wrapper.lora_A.weight.shape), "B_shape": list(wrapper.lora_B.weight.shape),
             "adapter_parameters": wrapper.lora_A.weight.numel() + wrapper.lora_B.weight.numel()} for name, wrapper in wrappers.items()]
