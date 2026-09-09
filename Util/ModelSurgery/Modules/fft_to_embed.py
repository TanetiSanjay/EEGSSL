import torch
import torch.nn as nn
import torch.nn.functional as F 

from einops import rearrange


class FFTChannelAttention(nn.Module):
    def __init__(
        self,
        n_channels  : int   = 128,
        fft_dim     : int   = 128,
        embed_dim   : int   = 768,
        freq_dim    : int   = 256,
        num_heads   : int   = 8,
        dropout     : float = 0.0
    ):
        super().__init__()

        self.n_channels = n_channels
        self.fft_dim    = fft_dim
        self.embed_dim  = embed_dim
        self.freq_dim   = freq_dim
        self.num_heads  = num_heads
        self.dropout    = dropout

        self.freq_proj  = nn.Sequential(
            nn.Linear(fft_dim, freq_dim),
            nn.GELU(),
            nn.LayerNorm(freq_dim)
        )

        self.channel_attention  = nn.MultiheadAttention(
            embed_dim   = freq_dim,
            num_heads   = num_heads,
            dropout     = dropout,
            batch_first = True
        )

        self.norm               = nn.LayerNorm(freq_dim)

        self.proj               = nn.Sequential(
            nn.Linear(freq_dim, embed_dim),
            nn.GELU()
        )

    def forward(self, x):
        B, C, N, F_dim = x.shape


        if C        != self.n_channels  : raise ValueError("Wrong C")
        if F_dim    != self.fft_dim     : raise ValueError("Wrong F_dim")

        x = rearrange(x, "b c n f -> (b n) c f")

        x = self.freq_proj(x)
        attn_out, _ = self.channel_attention(x, x, x, need_weights = True)

        x = self.norm(x + attn_out)
        x = x.mean(dim = 1)
        x = self.proj(x)

        x = rearrange(x, "(b n) d -> b n d", b=B, n=N)

        return x


class FFTDecoderHead(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        n_channels: int,
        fft_dim: int,
    ) -> None:
        super().__init__()

        self.channel_embedding = nn.Embedding(
            num_embeddings=n_channels,
            embedding_dim=embed_dim,
        )

        self.projection = nn.Sequential(
            nn.Linear(2 * embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, fft_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, N, D]

        Returns:
            [B, C, N, F]
        """
        batch_size, n_patches, embed_dim = x.shape
        n_channels = self.channel_embedding.num_embeddings

        channel_ids = torch.arange(
            n_channels,
            device=x.device,
        )

        channel_embeddings = self.channel_embedding(channel_ids)
        # [C, D]

        x = x.unsqueeze(2).expand(
            -1,
            -1,
            n_channels,
            -1,
        )
        # [B, N, C, D]

        channel_embeddings = channel_embeddings.view(
            1,
            1,
            n_channels,
            embed_dim,
        )
        # [1, 1, C, D]

        channel_embeddings = channel_embeddings.expand(
            batch_size,
            n_patches,
            -1,
            -1,
        )
        # [B, N, C, D]

        x = torch.cat(
            (x, channel_embeddings),
            dim=-1,
        )
        # [B, N, C, 2D]

        x = self.projection(x)
        # [B, N, C, F]

        return rearrange(
            x,
            "b n c f -> b c n f",
        )



if __name__ == '__main__':
    surgery_patch_embed = FFTDecoderHead(128, 128, 128)
    print(surgery_patch_embed(torch.randn(1, 128, 128)))

    count = 0
    for p in surgery_patch_embed.parameters(): count += p.numel()

    
    print(count)