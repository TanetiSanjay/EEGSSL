import sys
sys.path.insert(0, "./NeuroLM")

from model.model_vq import VQ
from model.model_neural_transformer import NTConfig
from Util.ModelSurgery.neurolmvq_to_customfft import patch_vq
from Util.DataLoader.ExtractedMAT_FFT.loader import create_dataloaders

import torch
import numpy as np
import matplotlib.pyplot as plt

def _unpack_batch(batch, device = "cpu"):
    x           = batch["x"].to(device, non_blocking=True)
    input_mask  = batch["input_mask"].to(device, non_blocking=True)

    input_chans = batch.get("input_chans", None)
    input_time  = batch.get("input_time", None)

    if input_chans is not None: input_chans = input_chans.to(device, non_blocking=True)
    if input_time  is not None: input_time  = input_time.to(device, non_blocking=True)

    return x, input_chans, input_time, input_mask.bool()


vq_path = ".weights/NeuroLm/checkpoints/VQ.pt"
vq = torch.load(vq_path, map_location="cpu", weights_only=False)

encoder_config = NTConfig(**vq["encoder_args"])
decoder_config = NTConfig(**vq["decoder_args"])

model = VQ(
    encoder_config=encoder_config,
    decoder_config=decoder_config
)
state_dict = vq["model"]
clean_state_dict = {}

for k, v in state_dict.items():
    if k.startswith("_orig_mod.VQ."):
        new_k = k[len("_orig_mod.VQ."):]
        clean_state_dict[new_k] = v
        
model.load_state_dict(clean_state_dict)
model = patch_vq(model, fft_dim=100, n_channels=128)
model.load_state_dict(torch.load("/speech/sanjay/Projects/EEGSSL/runs/vq_fft/checkpoints/best_test.pt", map_location="cpu")["model"], strict = False)

count = 0
for p in model.parameters(): count += p.numel()
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Trainable params: {trainable_params}") # Should match ~147M
print(f"The model has around {count} params")

train_loader, valid_loader = create_dataloaders(
    root_dir="Data/MAT_CHUNK_EXTRACTED",
    val_split=0.1,
    batch_size=1,
    num_workers=0,
    seed=42
)

batch = next(iter(train_loader))

x, input_chans, input_time, input_mask = _unpack_batch(batch)

print(x.shape)

loss, encoder_features, log, recon_x = model(
    x,
    x,
    input_chans = input_chans,
    input_time  = input_time,
    input_mask  = input_mask,
)

x_plot = x.detach().cpu().squeeze(0)
recon_plot = recon_x.detach().cpu().squeeze(0)

print("x:", x_plot.shape)
print("recon_x:", recon_plot.shape)

print("x:", x_plot.shape)
print("recon_x:", recon_plot.shape)

fs = 250
N_fft = 200

freqs = np.fft.rfftfreq(N_fft, d=1/fs)

segment = 0

# [128, 4, 101] -> [128, 101]
original = x_plot[:, segment, :].numpy()
reconstruction = recon_plot[:, segment, :].numpy()

print("Original spectrum:", original.shape)
print("Reconstruction spectrum:", reconstruction.shape)

fig, axes = plt.subplots(1, 2, figsize=(18, 7))

# Original
im1 = axes[0].imshow(
    original,
    aspect="auto",
    origin="lower",
    extent=[freqs[0], freqs[-1], 0, 128],
)

axes[0].set_title(f"Original Spectrum — Segment {segment}")
axes[0].set_xlabel("Frequency (Hz)")
axes[0].set_ylabel("EEG Channel")

fig.colorbar(im1, ax=axes[0], label="Magnitude")


# Reconstruction
im2 = axes[1].imshow(
    reconstruction,
    aspect="auto",
    origin="lower",
    extent=[freqs[0], freqs[-1], 0, 128],
)

axes[1].set_title(f"Reconstructed Spectrum — Segment {segment}")
axes[1].set_xlabel("Frequency (Hz)")
axes[1].set_ylabel("EEG Channel")

fig.colorbar(im2, ax=axes[1], label="Magnitude")

plt.tight_layout()
plt.savefig(
    "spectrum_segment_0.png",
    dpi=200,
    bbox_inches="tight"
)
plt.close()