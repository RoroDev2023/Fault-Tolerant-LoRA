"""Low-rank residual adapters for nn.Embedding, plus attachment to Linear and Embedding layers.

For an embedding table W (num × dim), the adapter adds scale · B(A[ids]): A is a
trainable num × r table and B an r → dim bias-free linear map, so the correction
to the table, A Bᵀ, has rank at most r (at most min(r, num, dim)). B starts at
zero, so the initial output is exactly the base embedding. The layer subclasses
the unchanged LoRALayer, so adapter_state_dict, load_adapter_state, and
base_state_dict apply unchanged. Linear layers keep attach_linear_lora's adapters
and seeded initialization order; embedding adapters are constructed after them.
"""

import math

import torch
from torch import nn

from fault_lora.models.lora import LoRALayer
from fault_lora.models.tinybert_lora import attach_linear_lora


class EmbeddingLoRALayer(LoRALayer):
    def __init__(self, base, rank, alpha):
        nn.Module.__init__(self)
        if not isinstance(base, nn.Embedding):
            raise TypeError("EmbeddingLoRALayer supports nn.Embedding only")
        if isinstance(rank, bool) or not isinstance(rank, int) or rank < 1:
            raise ValueError("rank must be a positive integer")
        if isinstance(alpha, bool) or not isinstance(alpha, (int, float)) or not math.isfinite(alpha) or alpha <= 0:
            raise ValueError("alpha must be finite and positive")
        if base.max_norm is not None or base.sparse:
            raise ValueError("max_norm and sparse embeddings are not supported")
        self.base = base.requires_grad_(False)
        self.rank, self.alpha, self.scale = rank, float(alpha), float(alpha) / rank
        options = {"device": base.weight.device, "dtype": base.weight.dtype}
        # A is a lookup table (A[ids]); B maps r → dim. Same convention as LoRALayer: random A, zero B.
        self.lora_A = nn.Embedding(base.num_embeddings, rank, **options)
        self.lora_B = nn.Linear(rank, base.embedding_dim, bias=False, **options)
        nn.init.normal_(self.lora_A.weight)
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, inputs):
        return self.base(inputs) + self.scale * self.lora_B(self.lora_A(inputs))

    def delta_weight(self):
        """The full-table correction scale · A Bᵀ (num × dim)."""
        return self.scale * self.lora_A.weight @ self.lora_B.weight.T


def attach_linear_embedding_lora(model, rank=4, alpha=4.0, *, include_embeddings=True):
    """Linear adapters exactly as attach_linear_lora, then (optionally) every named nn.Embedding."""
    embeddings = [name for name, module in model.named_modules() if name and isinstance(module, nn.Embedding)] if include_embeddings else []
    layout = attach_linear_lora(model, rank, alpha)
    for name in embeddings:
        base = model.get_submodule(name)
        wrapper = EmbeddingLoRALayer(base, rank, alpha)
        parent_name, _, child_name = name.rpartition(".")
        setattr(model.get_submodule(parent_name) if parent_name else model, child_name, wrapper)
        wrapper.lora_A.requires_grad_(True)
        wrapper.lora_B.requires_grad_(True)
        layout.append({"name": name, "kind": "Embedding", "adapter_class": type(wrapper).__name__, "rank": rank, "alpha": float(alpha), "scale": wrapper.scale,
                       "effective_rank_bound": min(rank, base.num_embeddings, base.embedding_dim), "base_weight_shape": list(base.weight.shape),
                       "A_shape": list(wrapper.lora_A.weight.shape), "B_shape": list(wrapper.lora_B.weight.shape),
                       "adapter_parameters": wrapper.lora_A.weight.numel() + wrapper.lora_B.weight.numel()})
    model.eval()
    return layout
