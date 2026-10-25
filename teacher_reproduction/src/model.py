"""Paper-aligned compact Vision Transformer used as the teacher.

Based on the architecture described in:
A. Ouamane et al., "Optimized Vision Transformers for Superior Plant Disease
Detection," IEEE Access, 2025, DOI: 10.1109/ACCESS.2025.3547416.

Paper-explicit choices implemented here:
- 224x224 input, patch size 16, embedding dimension 512, depth 6,
  8 attention heads, MLP dimension 1024
- sinusoidal positional encoding
- learnable [CLS] token
- ReLU feed-forward activation
- post-normalization around each residual sublayer

The paper does not specify all implementation details (for example dropout,
initialization, and the exact positional encoding treatment for [CLS]); those
are documented in README.md as implementation choices, not verified details.
"""

import math

import torch
from torch import nn


class SinusoidalPositionalEncoding(nn.Module):
    """Fixed sinusoidal positional encoding (not trainable)."""

    def __init__(self, num_positions: int, dim: int) -> None:
        super().__init__()
        pe = torch.zeros(num_positions, dim, dtype=torch.float32)
        position = torch.arange(num_positions, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, dim, 2, dtype=torch.float32) * (-math.log(10000.0) / dim)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)].to(dtype=x.dtype)


class PatchEmbedding(nn.Module):
    def __init__(self, image_size: int, patch_size: int, in_channels: int, dim: int) -> None:
        super().__init__()
        if image_size % patch_size:
            raise ValueError("image_size must be divisible by patch_size")
        self.num_patches = (image_size // patch_size) ** 2
        self.proj = nn.Conv2d(
            in_channels, dim, kernel_size=patch_size, stride=patch_size
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        return x.flatten(2).transpose(1, 2)


class EncoderBlock(nn.Module):
    """Post-norm encoder block following Equations (8)-(9) in the paper."""

    def __init__(self, dim: int, heads: int, mlp_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm1 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.ReLU(inplace=False),
            nn.Linear(mlp_dim, dim),
        )
        self.norm2 = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out, _ = self.attn(x, x, x, need_weights=False)
        x = self.norm1(x + self.dropout(attn_out))
        x = self.norm2(x + self.dropout(self.mlp(x)))
        return x


class VisionTransformer(nn.Module):
    def __init__(
        self,
        num_classes: int = 39,
        image_size: int = 224,
        patch_size: int = 16,
        dim: int = 512,
        depth: int = 6,
        heads: int = 8,
        mlp_dim: int = 1024,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.config = {
            "num_classes": num_classes,
            "image_size": image_size,
            "patch_size": patch_size,
            "dim": dim,
            "depth": depth,
            "heads": heads,
            "mlp_dim": mlp_dim,
            "dropout": dropout,
            "positional_encoding": "sinusoidal_fixed",
            "normalization": "post_norm",
            "activation": "ReLU",
        }
        self.patch_embed = PatchEmbedding(image_size, patch_size, 3, dim)
        num_tokens = self.patch_embed.num_patches + 1
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.pos_encoding = SinusoidalPositionalEncoding(num_tokens, dim)
        self.pos_drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            [EncoderBlock(dim, heads, mlp_dim, dropout) for _ in range(depth)]
        )
        self.final_norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        x = self.patch_embed(x)
        cls = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls, x), dim=1)
        x = self.pos_drop(self.pos_encoding(x))
        for block in self.blocks:
            x = block(x)
        x = self.final_norm(x)
        return self.head(x[:, 0])  # raw logits; CrossEntropyLoss applies log-softmax


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def build_teacher(num_classes: int = 39) -> VisionTransformer:
    return VisionTransformer(
        num_classes=num_classes,
        image_size=224,
        patch_size=16,
        dim=512,
        depth=6,
        heads=8,
        mlp_dim=1024,
        # The reference does not report a dropout probability; keep dropout off
        # rather than silently claiming 0.1 is a paper-specified value.
        dropout=0.0,
    )
