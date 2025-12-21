import torch
import torch.nn as nn
import torch.nn.functional as F


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class CrossAttentionDiTBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, audio_dim, mlp_ratio=4.0, dropout=0.1):
        super().__init__()

        self.norm1 = nn.LayerNorm(hidden_size)
        self.attn1 = nn.MultiheadAttention(hidden_size, num_heads, dropout=dropout, batch_first=True)

        # Cross Attention (Audio Conditioning)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.attn2 = nn.MultiheadAttention(hidden_size, num_heads, dropout=dropout, batch_first=True)
        self.audio_proj = nn.Linear(audio_dim, hidden_size) # Adapt audio dim to model dim

        self.norm3 = nn.LayerNorm(hidden_size)
        self.ff = FeedForward(hidden_size, int(hidden_size * mlp_ratio), dropout=dropout)

    def forward(self, x, audio_context):
        """
        x: (Batch, Seq_Len, Hidden_Size) -> Piano Latents (with Timestep info already added)
        audio_context: (Batch, Audio_Len, Audio_Dim) -> Pop Audio Features
        """

        # multihead self-attention
        residual = x
        x = self.norm1(x)
        x, _ = self.attn1(x, x, x)
        x = residual + x

        # multihead cross-attention
        residual = x
        x = self.norm2(x)

        # Project audio context for correct input
        # to the model
        # k, v from audio condition, q from piano latents x
        context = self.audio_proj(audio_context)

        x, _ = self.attn2(query=x, key=context, value=context)
        x = residual + x

        # feed forward network of transformer block
        residual = x
        x = self.norm3(x)
        x = self.ff(x)
        x = residual + x

        return x


def get_time_embedding(time_steps, temb_dim):
    assert temb_dim % 2 == 0, "time embedding dimension must be divisible by 2"

    # factor = 10000^(2i/d_model)
    factor = 10000 ** ((torch.arange(
        start=0, end=temb_dim // 2, dtype=torch.float32, device=time_steps.device) / (temb_dim // 2))
    )

    # pos / factor
    t_emb = time_steps[:, None].repeat(1, temb_dim // 2) / factor
    t_emb = torch.cat([torch.sin(t_emb), torch.cos(t_emb)], dim=-1)
    return t_emb


class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.frequency_embedding_size = frequency_embedding_size

    def forward(self, t):
        t_freq = get_time_embedding(t, self.frequency_embedding_size)
        # learnable MLP
        t_emb = self.mlp(t_freq)
        return t_emb


class ResBlock(nn.Module):
    """
    Simple Residual Block to keep gradients flowing in deep networks.
    """
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
            nn.SiLU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
        )

    def forward(self, x):
        return x + self.block(x)


class GroupNorm32(nn.GroupNorm):
    """
    Fixed GroupNorm with 32 groups (Standard in Stable Diffusion)
    """
    def forward(self, x):
        return super().forward(x)


def normalization(channels):
    """
    GroupNorm is superior to BatchNorm for Generative VAEs
    because it is independent of batch size.
    """
    return GroupNorm32(32, channels)


class ResnetBlock(nn.Module):
    def __init__(self, in_channels, out_channels=None, dropout=0.1):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels

        self.norm1 = normalization(in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.norm2 = normalization(out_channels)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)

        # Skip connection adjustment
        if self.in_channels != self.out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        h = x
        h = self.norm1(h)
        h = F.silu(h)
        h = self.conv1(h)

        h = self.norm2(h)
        h = F.silu(h)
        h = self.dropout(h)
        h = self.conv2(h)

        return h + self.shortcut(x)


class AttnBlock(nn.Module):
    """
    Self-Attention Block.
    """
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels
        self.norm = normalization(in_channels)
        self.q = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.k = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.v = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.proj_out = nn.Conv2d(in_channels, in_channels, kernel_size=1)

    def forward(self, x):
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        # Compute Attention
        b, c, h, w = q.shape
        q = q.reshape(b, c, h*w)
        q = q.permute(0, 2, 1)   # b,hw,c
        k = k.reshape(b, c, h*w) # b,c,hw

        w_ = torch.bmm(q, k)     # b,hw,hw
        w_ = w_ * (int(c)**(-0.5))
        w_ = F.softmax(w_, dim=2)

        # Attend to values
        v = v.reshape(b, c, h*w)
        w_ = w_.permute(0, 2, 1)   # b,hw,hw
        h_ = torch.bmm(v, w_)      # b, c,hw
        h_ = h_.reshape(b, c, h, w)

        h_ = self.proj_out(h_)

        return x + h_


class Downsample(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=2, padding=0)

    def forward(self, x):
        pad = (0, 1, 0, 1)
        x = nn.functional.pad(x, pad, mode="constant", value=0)
        return self.conv(x)


class Upsample(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2.0, mode="nearest")
        return self.conv(x)
