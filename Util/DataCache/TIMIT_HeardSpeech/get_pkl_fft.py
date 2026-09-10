import torch
import numpy as np
import pickle

from pathlib import Path
from tqdm    import tqdm

def patchify(
    data        : np.ndarray,
    patch_size  : int   = 200,
    hop_size    : int   = 50,
):
    n_channels, n_samples   = data.shape
    n_patches               = int(np.ceil(
        (n_samples - patch_size) / hop_size
    )) + 1

    padded_length           = (n_patches - 1) * hop_size + patch_size
    pad_len                 = padded_length - n_samples

    if pad_len > 0:
        data        = np.pad(data, ((0, 0), (0, pad_len)))

    patches = np.stack(
        [data[:, i:i + patch_size] for i in range(0, n_patches * hop_size, hop_size)],
        axis=1
    )

    return patches


def get_fft_data(
    root    : Path,
    out_dir : Path
):
    files = sorted(path for path in root.rglob("*.pkl") if path.is_file())

    for file in tqdm(files):
        with open(file, "rb") as f: data = pickle.load(f)

        for key, item in data.items():
            count, audio_id, speaker_id = key.split("_")
            data = patchify(item)
            fft  = np.log1p(np.abs(np.fft.rfft(data, axis=-1))[..., 1:])
            print(fft.shape)
    
        break

if __name__ == '__main__':
    get_fft_data(
        Path("Data/TIMIT_HEARONLY/TIMIT_EXTRACTED_25SYLL"),
        None
    )