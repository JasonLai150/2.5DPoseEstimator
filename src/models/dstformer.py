"""
Dual-Stream Spatio-Temporal Transformer (DSTformer) for 3D pose estimation.

Adapted from MotionBERT with LoRA support for efficient fine-tuning.
Reference: https://github.com/Walter0807/MotionBERT

Architecture:
- Dual parallel streams: spatial→temporal (ST) and temporal→spatial (TS)
- Learned attention fusion between streams
- Spatial attention: cross-joint within frame
- Temporal attention: cross-frame within joint
"""

import torch
import torch.nn as nn
import math
from functools import partial
from typing import Any, Optional
from collections import OrderedDict

from .base import PoseEstimatorBase
from .lora import LoRALinear, apply_lora_to_model, freeze_non_lora, count_lora_parameters


def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    """Truncated normal initialization."""
    with torch.no_grad():
        def norm_cdf(x):
            return (1. + math.erf(x / math.sqrt(2.))) / 2.

        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)
        tensor.uniform_(2 * l - 1, 2 * u - 1)
        tensor.erfinv_()
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)
        tensor.clamp_(min=a, max=b)
        return tensor


class DropPath(nn.Module):
    """Stochastic depth (drop path) for regularization."""

    def __init__(self, drop_prob: float = 0.):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        return x.div(keep_prob) * random_tensor


class MLP(nn.Module):
    """MLP block with GELU activation."""

    def __init__(
        self,
        in_features: int,
        hidden_features: int = None,
        out_features: int = None,
        drop: float = 0.,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class SpatialAttention(nn.Module):
    """
    Spatial attention: attends across joints within each frame.

    Input: (B*T, J, C) - batch*time flattened, joints, channels
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = True,
        attn_drop: float = 0.,
        proj_drop: float = 0.,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class TemporalAttention(nn.Module):
    """
    Temporal attention: attends across frames for each joint.

    Input: (B*T, J, C) with seqlen T to reshape properly
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = True,
        attn_drop: float = 0.,
        proj_drop: float = 0.,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, seqlen: int):
        BT, J, C = x.shape
        B = BT // seqlen
        T = seqlen

        # Reshape to (B, T, J, C) then (B, J, T, C) for temporal attention
        qkv = self.qkv(x).reshape(B, T, J, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(3, 0, 2, 4, 1, 5)  # (3, B, J, H, T, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]  # Each: (B, J, H, T, head_dim)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v)  # (B, J, H, T, head_dim)
        x = x.permute(0, 3, 1, 2, 4).reshape(BT, J, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class DualStreamBlock(nn.Module):
    """
    Dual-stream transformer block with spatial and temporal attention.

    Two modes:
    - 'st': Spatial attention first, then temporal
    - 'ts': Temporal attention first, then spatial
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        mlp_ratio: float = 4.,
        qkv_bias: bool = True,
        drop: float = 0.,
        attn_drop: float = 0.,
        drop_path: float = 0.,
        mode: str = 'st',
    ):
        super().__init__()
        self.mode = mode

        # Spatial path
        self.norm1_s = nn.LayerNorm(dim)
        self.attn_s = SpatialAttention(dim, num_heads, qkv_bias, attn_drop, drop)
        self.norm2_s = nn.LayerNorm(dim)
        self.mlp_s = MLP(dim, int(dim * mlp_ratio), drop=drop)

        # Temporal path
        self.norm1_t = nn.LayerNorm(dim)
        self.attn_t = TemporalAttention(dim, num_heads, qkv_bias, attn_drop, drop)
        self.norm2_t = nn.LayerNorm(dim)
        self.mlp_t = MLP(dim, int(dim * mlp_ratio), drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x, seqlen: int):
        if self.mode == 'st':
            # Spatial first
            x = x + self.drop_path(self.attn_s(self.norm1_s(x)))
            x = x + self.drop_path(self.mlp_s(self.norm2_s(x)))
            # Then temporal
            x = x + self.drop_path(self.attn_t(self.norm1_t(x), seqlen))
            x = x + self.drop_path(self.mlp_t(self.norm2_t(x)))
        else:  # 'ts'
            # Temporal first
            x = x + self.drop_path(self.attn_t(self.norm1_t(x), seqlen))
            x = x + self.drop_path(self.mlp_t(self.norm2_t(x)))
            # Then spatial
            x = x + self.drop_path(self.attn_s(self.norm1_s(x)))
            x = x + self.drop_path(self.mlp_s(self.norm2_s(x)))
        return x


class DSTformer(PoseEstimatorBase):
    """
    Dual-Stream Spatio-Temporal Transformer for 3D pose lifting.

    Architecture:
    - Input embedding: Linear projection + positional embeddings
    - Dual parallel streams: ST blocks and TS blocks
    - Learned attention fusion between streams
    - Output head: Representation layer + linear projection

    Args:
        cfg: Configuration object with model parameters
        pretrained_path: Path to pretrained MotionBERT weights
    """

    def __init__(
        self,
        cfg: Any,
        pretrained_path: Optional[str] = None,
    ):
        super().__init__(cfg)

        # Architecture params
        dim_in = cfg.model.get('input_dim', 2) + 1  # +1 for confidence
        dim_out = cfg.model.get('output_dim', 3)
        dim_feat = cfg.model.get('embed_dim', 256)
        dim_rep = cfg.model.get('dim_rep', 512)
        depth = cfg.model.get('depth', 5)
        num_heads = cfg.model.get('num_heads', 8)
        mlp_ratio = cfg.model.get('mlp_ratio', 4.)
        drop_rate = cfg.model.get('drop_rate', 0.0)
        attn_drop_rate = cfg.model.get('attn_drop_rate', 0.0)
        drop_path_rate = cfg.model.get('drop_path_rate', 0.0)

        self.dim_feat = dim_feat
        self.depth = depth

        # Input embedding
        self.joints_embed = nn.Linear(dim_in, dim_feat)
        self.pos_drop = nn.Dropout(p=drop_rate)

        # Positional embeddings
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_joints, dim_feat))
        self.temp_embed = nn.Parameter(torch.zeros(1, self.seq_len, 1, dim_feat))
        trunc_normal_(self.pos_embed, std=.02)
        trunc_normal_(self.temp_embed, std=.02)

        # Stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]

        # Dual-stream blocks
        self.blocks_st = nn.ModuleList([
            DualStreamBlock(
                dim=dim_feat, num_heads=num_heads, mlp_ratio=mlp_ratio,
                qkv_bias=True, drop=drop_rate, attn_drop=attn_drop_rate,
                drop_path=dpr[i], mode='st'
            ) for i in range(depth)
        ])
        self.blocks_ts = nn.ModuleList([
            DualStreamBlock(
                dim=dim_feat, num_heads=num_heads, mlp_ratio=mlp_ratio,
                qkv_bias=True, drop=drop_rate, attn_drop=attn_drop_rate,
                drop_path=dpr[i], mode='ts'
            ) for i in range(depth)
        ])

        # Stream fusion attention
        self.stream_fusion = nn.ModuleList([
            nn.Linear(dim_feat * 2, 2) for _ in range(depth)
        ])
        # Initialize fusion to 0.5/0.5
        for fusion in self.stream_fusion:
            nn.init.zeros_(fusion.weight)
            nn.init.constant_(fusion.bias, 0.5)

        # Output layers
        self.norm = nn.LayerNorm(dim_feat)
        self.pre_logits = nn.Sequential(OrderedDict([
            ('fc', nn.Linear(dim_feat, dim_rep)),
            ('act', nn.Tanh())
        ]))
        self.head = nn.Linear(dim_rep, dim_out)

        # Initialize weights
        self.apply(self._init_weights)

        # Load pretrained weights if provided
        if pretrained_path:
            self.load_pretrained(pretrained_path)

        # Apply LoRA if configured
        if cfg.model.get('lora', {}).get('enabled', False):
            self._apply_lora(cfg.model.lora)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def _apply_lora(self, lora_cfg):
        """Apply LoRA to attention layers."""
        rank = lora_cfg.get('rank', 8)
        alpha = lora_cfg.get('alpha', 16)
        dropout = lora_cfg.get('dropout', 0.0)
        target_modules = lora_cfg.get('target_modules', ['qkv', 'proj'])

        apply_lora_to_model(self, target_modules, rank, alpha, dropout)
        freeze_non_lora(self)

        total, lora = count_lora_parameters(self)
        print(f"LoRA enabled: {lora:,} trainable params ({100*lora/total:.2f}% of {total:,})")

    def load_pretrained(self, path: str):
        """Load pretrained MotionBERT weights."""
        checkpoint = torch.load(path, map_location='cpu')

        # Handle different checkpoint formats
        if 'model' in checkpoint:
            state_dict = checkpoint['model']
        elif 'model_pos' in checkpoint:
            state_dict = checkpoint['model_pos']
        else:
            state_dict = checkpoint

        # Map MotionBERT keys to our model
        new_state_dict = {}
        for k, v in state_dict.items():
            # Remove 'module.' prefix if present
            if k.startswith('module.'):
                k = k[7:]

            # Map block names
            new_key = k
            # Add more mappings as needed

            new_state_dict[new_key] = v

        # Load with strict=False to handle mismatches
        missing, unexpected = self.load_state_dict(new_state_dict, strict=False)
        if missing:
            print(f"Missing keys: {len(missing)}")
        if unexpected:
            print(f"Unexpected keys: {len(unexpected)}")
        print(f"Loaded pretrained weights from {path}")

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: 2D keypoints (B, T, J, 2)
            mask: Visibility mask (B, T, J)

        Returns:
            3D poses (B, T, J, 3)
        """
        B, T, J, C = x.shape

        # Add confidence channel
        if C == 2:
            if mask is not None:
                conf = mask.unsqueeze(-1).float()
            else:
                conf = torch.ones(B, T, J, 1, device=x.device, dtype=x.dtype)
            x = torch.cat([x, conf], dim=-1)

        # Flatten batch and time
        x = x.reshape(B * T, J, -1)

        # Input embedding
        x = self.joints_embed(x)
        x = x + self.pos_embed

        # Add temporal embedding
        x = x.reshape(B, T, J, -1) + self.temp_embed[:, :T, :, :]
        x = x.reshape(B * T, J, -1)
        x = self.pos_drop(x)

        # Process through dual-stream blocks
        for idx, (blk_st, blk_ts, fusion) in enumerate(
            zip(self.blocks_st, self.blocks_ts, self.stream_fusion)
        ):
            x_st = blk_st(x, T)
            x_ts = blk_ts(x, T)

            # Fuse streams with learned attention
            alpha = torch.cat([x_st, x_ts], dim=-1)
            alpha = fusion(alpha).softmax(dim=-1)
            x = x_st * alpha[..., 0:1] + x_ts * alpha[..., 1:2]

        # Output
        x = self.norm(x)
        x = x.reshape(B, T, J, -1)
        x = self.pre_logits(x)
        x = self.head(x)

        return x

    def get_representation(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Get intermediate representation (before output head)."""
        B, T, J, C = x.shape

        if C == 2:
            if mask is not None:
                conf = mask.unsqueeze(-1).float()
            else:
                conf = torch.ones(B, T, J, 1, device=x.device, dtype=x.dtype)
            x = torch.cat([x, conf], dim=-1)

        x = x.reshape(B * T, J, -1)
        x = self.joints_embed(x)
        x = x + self.pos_embed
        x = x.reshape(B, T, J, -1) + self.temp_embed[:, :T, :, :]
        x = x.reshape(B * T, J, -1)
        x = self.pos_drop(x)

        for idx, (blk_st, blk_ts, fusion) in enumerate(
            zip(self.blocks_st, self.blocks_ts, self.stream_fusion)
        ):
            x_st = blk_st(x, T)
            x_ts = blk_ts(x, T)
            alpha = torch.cat([x_st, x_ts], dim=-1)
            alpha = fusion(alpha).softmax(dim=-1)
            x = x_st * alpha[..., 0:1] + x_ts * alpha[..., 1:2]

        x = self.norm(x)
        x = x.reshape(B, T, J, -1)
        x = self.pre_logits(x)

        return x
