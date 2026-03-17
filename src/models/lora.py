"""
Low-Rank Adaptation (LoRA) module for efficient fine-tuning.

LoRA injects trainable low-rank matrices into frozen pretrained weights,
enabling parameter-efficient adaptation while preventing catastrophic forgetting.

Reference: https://arxiv.org/abs/2106.09685
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional


class LoRALinear(nn.Module):
    """
    Linear layer with LoRA adaptation.

    Computes: output = Wx + (BA)x * scaling
    Where W is frozen, and B, A are low-rank trainable matrices.

    Args:
        in_features: Input dimension
        out_features: Output dimension
        rank: LoRA rank (lower = fewer parameters, higher = more capacity)
        alpha: LoRA scaling factor
        dropout: Dropout on LoRA path
        merge_weights: If True, merge LoRA into base weights for inference
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.0,
        bias: bool = True,
        merge_weights: bool = False,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        self.merge_weights = merge_weights
        self.merged = False

        # Base linear layer (frozen)
        self.linear = nn.Linear(in_features, out_features, bias=bias)

        # LoRA matrices
        self.lora_A = nn.Parameter(torch.zeros(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))

        # Dropout
        self.lora_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # Initialize
        self.reset_lora_parameters()

    def reset_lora_parameters(self):
        """Initialize LoRA matrices."""
        # A uses Kaiming uniform, B uses zeros
        # This ensures the LoRA contribution starts at zero
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def freeze_base(self):
        """Freeze the base linear weights."""
        self.linear.weight.requires_grad = False
        if self.linear.bias is not None:
            self.linear.bias.requires_grad = False

    def unfreeze_base(self):
        """Unfreeze the base linear weights."""
        self.linear.weight.requires_grad = True
        if self.linear.bias is not None:
            self.linear.bias.requires_grad = True

    def merge(self):
        """Merge LoRA weights into base weights for efficient inference."""
        if not self.merged:
            self.linear.weight.data += (self.lora_B @ self.lora_A) * self.scaling
            self.merged = True

    def unmerge(self):
        """Unmerge LoRA weights from base weights."""
        if self.merged:
            self.linear.weight.data -= (self.lora_B @ self.lora_A) * self.scaling
            self.merged = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.merged:
            return self.linear(x)

        # Base output
        base_out = self.linear(x)

        # LoRA output: x @ A^T @ B^T * scaling
        lora_out = self.lora_dropout(x)
        lora_out = F.linear(lora_out, self.lora_A)  # (*, rank)
        lora_out = F.linear(lora_out, self.lora_B)  # (*, out_features)
        lora_out = lora_out * self.scaling

        return base_out + lora_out

    @classmethod
    def from_linear(
        cls,
        linear: nn.Linear,
        rank: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.0,
    ) -> "LoRALinear":
        """Create LoRALinear from existing Linear layer."""
        lora_linear = cls(
            in_features=linear.in_features,
            out_features=linear.out_features,
            rank=rank,
            alpha=alpha,
            dropout=dropout,
            bias=linear.bias is not None,
        )
        # Copy weights
        lora_linear.linear.weight.data.copy_(linear.weight.data)
        if linear.bias is not None:
            lora_linear.linear.bias.data.copy_(linear.bias.data)
        # Freeze base
        lora_linear.freeze_base()
        return lora_linear


def apply_lora_to_model(
    model: nn.Module,
    target_modules: list[str],
    rank: int = 8,
    alpha: float = 16.0,
    dropout: float = 0.0,
) -> nn.Module:
    """
    Apply LoRA to specified modules in a model.

    Args:
        model: The model to modify
        target_modules: List of module name patterns to apply LoRA to
                       (e.g., ['qkv', 'proj', 'fc1', 'fc2'])
        rank: LoRA rank
        alpha: LoRA scaling
        dropout: LoRA dropout

    Returns:
        Modified model with LoRA layers
    """
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            # Check if this module should have LoRA
            should_apply = any(target in name for target in target_modules)
            if should_apply:
                # Get parent module and attribute name
                parts = name.rsplit('.', 1)
                if len(parts) == 2:
                    parent_name, attr_name = parts
                    parent = model.get_submodule(parent_name)
                else:
                    parent = model
                    attr_name = name

                # Replace with LoRA version
                lora_module = LoRALinear.from_linear(
                    module, rank=rank, alpha=alpha, dropout=dropout
                )
                setattr(parent, attr_name, lora_module)

    return model


def get_lora_parameters(model: nn.Module) -> list[nn.Parameter]:
    """Get only the LoRA parameters (for optimizer)."""
    lora_params = []
    for name, param in model.named_parameters():
        if 'lora_' in name:
            lora_params.append(param)
    return lora_params


def count_lora_parameters(model: nn.Module) -> tuple[int, int]:
    """Count total and LoRA parameters."""
    total = sum(p.numel() for p in model.parameters())
    lora = sum(p.numel() for n, p in model.named_parameters() if 'lora_' in n)
    return total, lora


def freeze_non_lora(model: nn.Module) -> None:
    """Freeze all parameters except LoRA."""
    for name, param in model.named_parameters():
        if 'lora_' not in name:
            param.requires_grad = False
        else:
            param.requires_grad = True
