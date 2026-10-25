"""
Vision Transformer matching the optimal configuration reported in:

    A. Ouamane, A. Chouchane, Y. Himeur, S. Miniaoui, S. Atalla, W. Mansoor,
    and H. Al-Ahmad, "Optimized Vision Transformers for Superior Plant Disease
    Detection," IEEE Access, vol. 13, pp. 48552-48570, 2025.

Default hyperparameters = the paper's reported optimum:
    image_size=224, patch_size=16, dim=512, depth=6, heads=8, mlp_dim=1024
    -> 13.02M trainable parameters, 50.89 MB storage in the paper.

Each transformer block can optionally store its attention weights, which
visualize.py uses to draw attention-rollout heatmaps (what the model "looks at").
"""

import math
import torch
import torch.nn as nn


class SinusoidalPositionalEncoding(nn.Module):
    """Fixed (non-learnable) sinusoidal position encoding."""

    def __init__(self, num_positions: int, dim: int):
        super().__init__()
        pe = torch.zeros(num_positions, dim)
        position = torch.arange(0, num_positions, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, dim, 2, dtype=torch.float32) * (-math.log(10000.0) / dim)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class PatchEmbedding(nn.Module):
    """Non-overlapping patches -> linear projection (one strided conv)."""

    def __init__(self, image_size: int, patch_size: int, in_channels: int, dim: int):
        super().__init__()
        assert image_size % patch_size == 0, "image_size must be divisible by patch_size"
        self.num_patches = (image_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_channels, dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)                    # (B, dim, H/P, W/P)
        return x.flatten(2).transpose(1, 2)  # (B, num_patches, dim)


class EncoderBlock(nn.Module):
    """Pre-norm transformer block: LN -> MHSA -> residual, LN -> MLP -> residual."""

    def __init__(self, dim: int, heads: int, mlp_dim: int, dropout: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(mlp_dim, dim), nn.Dropout(dropout),
        )
        self.drop = nn.Dropout(dropout)
        self.save_attn = False      # switched on only for visualization
        self.attn_weights = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        if self.save_attn:
            out, w = self.attn(h, h, h, need_weights=True, average_attn_weights=True)
            self.attn_weights = w.detach()
        else:
            out, _ = self.attn(h, h, h, need_weights=False)
        x = x + self.drop(out)
        x = x + self.mlp(self.norm2(x))
        return x


class VisionTransformer(nn.Module):
    def __init__(
        self,
        image_size: int = 224,
        patch_size: int = 16,
        num_classes: int = 38,        # PlantVillage has 38 classes
        dim: int = 512,
        depth: int = 6,
        heads: int = 8,
        mlp_dim: int = 1024,
        in_channels: int = 3,
        dropout: float = 0.1,
        positional_encoding: str = "sinusoidal",  # "sinusoidal" or "learned"
    ):
        super().__init__()
        self.patch_embed = PatchEmbedding(image_size, patch_size, in_channels, dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        self.pos_encoding_type = positional_encoding
        if positional_encoding == "sinusoidal":
            self.pos_embed = SinusoidalPositionalEncoding(num_patches + 1, dim)
        elif positional_encoding == "learned":
            self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, dim))
            nn.init.trunc_normal_(self.pos_embed, std=0.02)
        else:
            raise ValueError("positional_encoding must be 'sinusoidal' or 'learned'")

        self.pos_drop = nn.Dropout(dropout)
        self.encoder_blocks = nn.ModuleList(
            [EncoderBlock(dim, heads, mlp_dim, dropout) for _ in range(depth)]
        )
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        x = self.patch_embed(x)
        x = torch.cat((self.cls_token.expand(B, -1, -1), x), dim=1)

        if self.pos_encoding_type == "sinusoidal":
            x = self.pos_embed(x)
        else:
            x = x + self.pos_embed
        x = self.pos_drop(x)

        for block in self.encoder_blocks:
            x = block(x)

        x = self.norm(x)
        return self.head(x[:, 0])


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def model_size_mb(model: nn.Module) -> float:
    """Approximate float32 size in MB (how the paper reports storage)."""
    return count_parameters(model) * 4 / (1024 ** 2)


def build_baseline_model(num_classes: int = 38) -> VisionTransformer:
    """The base paper's exact optimal configuration."""
    return VisionTransformer(
        image_size=224, patch_size=16, num_classes=num_classes,
        dim=512, depth=6, heads=8, mlp_dim=1024,
    )
