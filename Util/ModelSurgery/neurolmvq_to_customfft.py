import os
import sys
import types

import torch
import torch.nn as nn
from einops import rearrange

sys.path.insert(0, os.environ.get("NEUROLM_PATH", "./NeuroLM"))

from model.model_vq import VQ  
from model.model_neural_transformer import NTConfig


class FFTChannelAttention(nn.Module):
    def __init__(
        self,
        fft_dim: int,
        embed_dim: int = 768,
        channel_dim: int = 256,
        num_heads: int = 8,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.fft_dim = fft_dim
        self.embed_dim = embed_dim
        self.channel_dim = channel_dim

        self.freq_proj = nn.Sequential(
            nn.Linear(fft_dim, channel_dim),
            nn.GELU(),
            nn.LayerNorm(channel_dim),
        )

        self.channel_attention = nn.MultiheadAttention(
            embed_dim=channel_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(channel_dim)

        self.proj = nn.Sequential(
            nn.Linear(channel_dim, embed_dim),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, N, F_dim = x.shape
        if F_dim != self.fft_dim:
            raise ValueError(
                f"FFTChannelAttention configured for fft_dim={self.fft_dim}, got F={F_dim}"
            )

        x = rearrange(x, "b c n f -> (b n) c f")
        x = self.freq_proj(x)
        attn_out, _ = self.channel_attention(x, x, x)
        x = self.norm(x + attn_out)
        x = x.mean(dim=1)
        x = self.proj(x)
        x = rearrange(x, "(b n) d -> b n d", b=B, n=N)
        return x


class FFTDecoderHead(nn.Module):
    def __init__(self, embed_dim: int, n_channels: int, fft_dim: int):
        super().__init__()
        self.embed_dim = embed_dim
        self.n_channels = n_channels
        self.fft_dim = fft_dim

        self.channel_embedding = nn.Embedding(n_channels, embed_dim)

        self.mlp = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, fft_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, D = x.shape
        C = self.n_channels

        channel_ids = torch.arange(C, device=x.device)
        channel_emb = self.channel_embedding(channel_ids)

        x = x.unsqueeze(2).expand(B, N, C, D)
        channel_emb = channel_emb.view(1, 1, C, D).expand(B, N, C, D)

        feat = torch.cat([x, channel_emb], dim=-1)
        out = self.mlp(feat)
        out = rearrange(out, "b n c f -> b c n f")
        return out


def _get_n_embd(transformer: nn.Module) -> int:
    return transformer.pos_embed.embedding_dim


def build_attention_mask(input_mask, seq_len: int):
    if input_mask is None:
        return None
<<<<<<< HEAD
    return input_mask.unsqueeze(1).repeat(1, seq_len, 1).unsqueeze(1)
=======
    return input_mask.unsqueeze(1).repeat(1, seq_len, 1).unsqueeze(1).bool()
>>>>>>> recovery


def _default_chans_and_time(batch_size, seq_len, device, input_chans, input_time):
    if input_chans is None:
        input_chans = torch.zeros(batch_size, seq_len, dtype=torch.long, device=device)
    if input_time is None:
        input_time = (torch.arange(seq_len, device=device).long() % 64)
        input_time = input_time.unsqueeze(0).expand(batch_size, seq_len)
    return input_chans, input_time


def _patched_forward_features(self, x, input_chans=None, input_times=None, mask=None, return_all_tokens=False, **kwargs):
    x = self.patch_embed(x)
    x = x + self.pos_embed(input_chans)
    x = x + self.time_embed(input_times)
    x = self.pos_drop(x)

    for blk in self.blocks:
        x = blk(x, mask)

    x = self.norm(x)
    if self.fc_norm is not None:
        if return_all_tokens:
            return self.fc_norm(x)
        return self.fc_norm(x.mean(1))
    return x


def _fft_encode(self, x, input_chans=None, input_time=None, mask=None):
    input_chans, input_time = _default_chans_and_time(
        x.size(0), x.size(2), x.device, input_chans, input_time
    )
    encoder_features = self.encoder(
        x,
        input_chans,
        input_time,
        mask,
        return_all_tokens=True,
    )

    with torch.amp.autocast(device_type=x.device.type, enabled=False):
        to_quantizer_features = self.encode_task_layer(
            encoder_features.type_as(self.encode_task_layer[-1].weight)
        )

    quantize, loss, embed_ind = self.quantize(to_quantizer_features)

    return quantize, embed_ind, loss, encoder_features


def _fft_decode(self, quantize, input_chans=None, input_time=None, mask=None, **kwargs):
    input_chans, input_time = _default_chans_and_time(
        quantize.size(0), quantize.size(1), quantize.device, input_chans, input_time
    )
    decoder_features = self.decoder_freq(
        quantize,
        input_chans,
        input_time,
        mask,
        return_all_tokens=True,
    )
    rec_fft = self.fft_decoder_head(decoder_features)
    return rec_fft


def _fft_get_tokens(self, data, input_chans=None, input_times=None, mask=None, **kwargs):
    quantize, embed_ind, loss, _ = self.encode(data, input_chans, input_times, mask)
    return embed_ind.view(data.size(0), data.size(2))


def _fft_get_codebook_indices(self, x, input_chans=None, input_time=None, input_mask=None, **kwargs):
    mask = build_attention_mask(input_mask, x.size(2))
    return self.get_tokens(x, input_chans, input_time, mask, **kwargs)


def _fft_forward(self, x, y_fft, input_chans=None, input_time=None, input_mask=None, **kwargs):
    if x.dim() != 4:
        raise ValueError(f"Expected x of shape [B, C, N, F], got {tuple(x.shape)}")
    if x.shape != y_fft.shape:
        raise ValueError(
            f"Expected y_fft to have the same shape as x {tuple(x.shape)}, got {tuple(y_fft.shape)}"
        )

    mask = build_attention_mask(input_mask, x.size(2))

    quantize, embed_ind, emb_loss, encoder_features = self.encode(
        x, input_chans, input_time, mask
    )

    xrec_fft = self.decode(quantize, input_chans, input_time, mask)

    if input_mask is not None:
        loss_mask = input_mask[:, None, :, None]
        rec_fft_loss = self.calculate_rec_loss(xrec_fft * loss_mask, y_fft)
    else:
        rec_fft_loss = self.calculate_rec_loss(xrec_fft, y_fft)

    loss = emb_loss + rec_fft_loss

    split = "train" if self.training else "val"
    log = {
        f"{split}/quant_loss": emb_loss.item(),
        f"{split}/rec_fft_loss": rec_fft_loss.item(),
        f"{split}/total_loss": loss.item(),
    }

<<<<<<< HEAD
    return loss, encoder_features, log
=======
    return loss, encoder_features, log, xrec_fft
>>>>>>> recovery


def patch_vq(model: "VQ", fft_dim: int, n_channels: int) -> "VQ":
    encoder_embed_dim = _get_n_embd(model.encoder)
    decoder_embed_dim = _get_n_embd(model.decoder_freq)

    model.encoder.patch_embed = FFTChannelAttention(
        fft_dim=fft_dim,
        embed_dim=encoder_embed_dim,
    )

    model.fft_decoder_head = FFTDecoderHead(
        embed_dim=decoder_embed_dim,
        n_channels=n_channels,
        fft_dim=fft_dim,
    )

    model.encoder.forward_features = types.MethodType(_patched_forward_features, model.encoder)
    model.decoder_freq.forward_features = types.MethodType(_patched_forward_features, model.decoder_freq)

    model.encode = types.MethodType(_fft_encode, model)
    model.decode = types.MethodType(_fft_decode, model)
    model.get_tokens = types.MethodType(_fft_get_tokens, model)
    model.get_codebook_indices = types.MethodType(_fft_get_codebook_indices, model)
    model.forward = types.MethodType(_fft_forward, model)

    return model


if __name__ == '__main__':
    B, C, N, F_dim = 2, 128, 4, 200

    encoder_args = dict(n_layer=12, n_head=12, n_embd=768, block_size=1024,
                         bias=False, dropout=0., num_classes=0, in_chans=1, out_chans=16)
    decoder_args = dict(n_layer=4, n_head=12, n_embd=768, block_size=1024,
                         bias=False, dropout=0., num_classes=0, in_chans=128)

    encoder_config = NTConfig(**encoder_args)
    decoder_config = NTConfig(**decoder_args)

    model = VQ(encoder_config=encoder_config, decoder_config=decoder_config)
    model = patch_vq(model, fft_dim=F_dim, n_channels=C)
    model.eval()

    x = torch.randn(B, C, N, F_dim)
    y_fft = torch.randn(B, C, N, F_dim)
    input_mask = torch.ones(B, N)

    with torch.no_grad():
<<<<<<< HEAD
        loss, encoder_features, log = model(x, x, input_mask=input_mask)
=======
        loss, encoder_features, log, xrec_fft = model(x, x, input_mask=input_mask)
>>>>>>> recovery

    print("loss:", loss.item())
    print("encoder_features:", encoder_features.shape)
    print("log:", log)

    tokens = model.get_codebook_indices(x, input_mask=input_mask)
    print("tokens:", tokens.shape)