"""Unmerged low-rank residuals for ordinary Conv2d and Linear layers.

For Conv2d, B is a 1x1 convolution after A with the base kernel geometry.
The flattened effective update has rank at most r; base weights stay frozen.
"""

import math

import torch
from torch import nn


def _validate_layer(base, rank, alpha):
    if not isinstance(base, (nn.Conv2d, nn.Linear)):
        raise TypeError("LoRA supports Conv2d and Linear")
    if isinstance(base, nn.Conv2d) and base.groups != 1:
        raise ValueError("Grouped convolutions require a separate adapter design")
    input_width = base.in_features if isinstance(base, nn.Linear) else base.in_channels * math.prod(base.kernel_size)
    output_width = base.out_features if isinstance(base, nn.Linear) else base.out_channels
    if isinstance(rank, bool) or not isinstance(rank, int) or not 1 <= rank <= min(input_width, output_width):
        raise ValueError("rank must be a positive integer no greater than the flattened matrix dimensions")
    if isinstance(alpha, bool) or not isinstance(alpha, (int, float)) or not math.isfinite(alpha) or alpha <= 0:
        raise ValueError("alpha must be finite and positive")


class LoRALayer(nn.Module):
    def __init__(self, base, rank, alpha):
        super().__init__()
        _validate_layer(base, rank, alpha)
        self.base = base.requires_grad_(False)
        self.rank, self.alpha, self.scale = rank, float(alpha), float(alpha) / rank
        options = {"device": base.weight.device, "dtype": base.weight.dtype, "bias": False}
        if isinstance(base, nn.Linear):
            self.lora_A = nn.Linear(base.in_features, rank, **options)
            self.lora_B = nn.Linear(rank, base.out_features, **options)
        else:
            self.lora_A = nn.Conv2d(base.in_channels, rank, base.kernel_size, stride=base.stride,
                                    padding=base.padding, dilation=base.dilation, padding_mode=base.padding_mode, **options)
            self.lora_B = nn.Conv2d(rank, base.out_channels, 1, **options)
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, inputs):
        return self.base(inputs) + self.scale * self.lora_B(self.lora_A(inputs))


def attach_lora(model, rank=4, alpha=4.0, target_names=None):
    """Freeze the entire base and attach adapters to an explicit layer list.

    None selects every named Conv2d/Linear. All choices are validated before
    mutation. BatchNorm buffers are held fixed by the recovery training loop.
    """
    if any(isinstance(module, LoRALayer) for module in model.modules()):
        raise ValueError("Adapters are already attached")
    candidates = {name: module for name, module in model.named_modules() if name and isinstance(module, (nn.Conv2d, nn.Linear))}
    names = list(candidates) if target_names is None else list(target_names)
    if not names or len(set(names)) != len(names) or any(name not in candidates for name in names):
        raise ValueError("Targets must be distinct existing named Conv2d/Linear layers")
    for name in names:
        _validate_layer(candidates[name], rank, alpha)
    wrappers = {name: LoRALayer(candidates[name], rank, alpha) for name in names}
    model.requires_grad_(False)
    for name, wrapper in wrappers.items():
        parent_name, _, child_name = name.rpartition(".")
        parent = model.get_submodule(parent_name) if parent_name else model
        setattr(parent, child_name, wrapper)
        wrapper.lora_A.requires_grad_(True)
        wrapper.lora_B.requires_grad_(True)
    model.eval()
    return [{"name": name, "kind": type(wrapper.base).__name__, "rank": rank, "alpha": float(alpha),
             "scale": wrapper.scale, "base_weight_shape": list(wrapper.base.weight.shape),
             "A_shape": list(wrapper.lora_A.weight.shape), "B_shape": list(wrapper.lora_B.weight.shape),
             "adapter_parameters": wrapper.lora_A.weight.numel() + wrapper.lora_B.weight.numel()}
            for name, wrapper in wrappers.items()]


def adapter_state_dict(model):
    return {f"{name}.{part}.weight": getattr(module, part).weight.detach().cpu().clone()
            for name, module in model.named_modules() if isinstance(module, LoRALayer)
            for part in ("lora_A", "lora_B")}


def load_adapter_state(model, state):
    expected = adapter_state_dict(model)
    if set(state) != set(expected) or any(state[name].shape != value.shape or state[name].dtype != value.dtype
                                         or not torch.isfinite(state[name]).all() for name, value in expected.items()):
        raise ValueError("Adapter state has unexpected names, shapes, dtypes, or nonfinite values")
    with torch.no_grad():
        for name, module in model.named_modules():
            if isinstance(module, LoRALayer):
                for part in ("lora_A", "lora_B"):
                    getattr(module, part).weight.copy_(state[f"{name}.{part}.weight"])


def base_state_dict(model):
    """Return frozen parameters AND buffers under their pre-adapter names."""
    state = dict(model.state_dict())
    for name, module in model.named_modules():
        if isinstance(module, LoRALayer):
            for part in ("lora_A", "lora_B"):
                del state[f"{name}.{part}.weight"]
            for key in module.base.state_dict():
                state[f"{name}.{key}"] = state.pop(f"{name}.base.{key}")
    return {name: value.detach().cpu().clone() for name, value in state.items()}
