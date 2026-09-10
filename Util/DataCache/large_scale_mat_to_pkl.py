import pickle
from math import gcd
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from scipy.io import loadmat
from scipy.signal import resample_poly
from tqdm import tqdm

MAX_N = 4
TARGET_FS = 200
PATCH_SIZE = 200


def find_mat_files(root: Path) -> list[Path]:
    """Recursively find all .mat files under root, sorted for determinism."""
    return sorted(path for path in root.rglob("*.mat") if path.is_file())


def extract_data(path: Path) -> Tuple[Optional[np.ndarray], Optional[float]]:
    """
    Load a .mat file and pull out the EEG data array (dropping the first
    of 129 rows, which is assumed to be a non-EEG reference channel) along
    with the recording's sampling rate, if present.
    """
    mat = loadmat(path)

    sampling_rate = None
    if "EEGSamplingRate" in mat:
        raw_rate = np.asarray(mat["EEGSamplingRate"]).squeeze()
        if raw_rate.size == 1:
            sampling_rate = float(raw_rate)

    for _, item in mat.items():
        if isinstance(item, np.ndarray) and item.ndim >= 2 and item.shape[0] == 129:
            return item[1:, :], sampling_rate

    return None, sampling_rate


def resample_eeg(
    data: np.ndarray,
    original_fs: Optional[float],
    target_fs: int = TARGET_FS,
) -> np.ndarray:
    """Resample EEG data (channels, samples) from original_fs to target_fs."""
    if original_fs is None:
        print("WARNING: EEGSamplingRate not found. Skipping resampling.")
        return data

    if original_fs == target_fs:
        return data

    print(f"Resampling EEG from {original_fs:g} Hz to {target_fs} Hz")

    common = gcd(int(round(original_fs)), int(target_fs))
    up = target_fs // common
    down = int(round(original_fs)) // common

    data = resample_poly(data, up=up, down=down, axis=1)
    return data.astype(np.float32)


def normalize_channels(data: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """
    Remove DC offset and normalize each EEG channel independently:
    subtract the per-channel mean, then divide by the per-channel std.

    data: [n_channels, n_samples]
    """
    mean = data.mean(axis=1, keepdims=True)
    std  = data.std(axis=1, keepdims=True) + eps

    data = (data - mean) / std
    return data.astype(np.float32)


def patchify(data: np.ndarray, patch_size: int) -> np.ndarray:
    """Split (channels, samples) into (channels, n_patches, patch_size)."""
    n_channels, n_samples = data.shape
    n_patches = n_samples // patch_size

    return data[:, : n_patches * patch_size].reshape(
        n_channels,
        n_patches,
        patch_size,
    )


def compute_fft(patches: np.ndarray) -> np.ndarray:
    """
    Remove any residual per-patch DC offset, then compute the log-magnitude
    FFT of each patch along the last axis.

    patches: [n_channels, n_patches, patch_size]
    returns: [n_channels, n_patches, patch_size // 2 + 1]
    """
    patches = patches - patches.mean(axis=-1, keepdims=True)

    fft = np.log1p(np.abs(np.fft.rfft(patches, axis=-1)))[..., 1:]
    return fft.astype(np.float32)


def process_file(file: Path, out_dir: Path, count: int, save_dict: dict) -> None:
    data, sampling_rate = extract_data(file)

    if data is None:
        print(f"WARNING: No EEG data found in {file}")
        return

    data = resample_eeg(data, sampling_rate, TARGET_FS)
    data = normalize_channels(data)

    fft = compute_fft(patchify(data, PATCH_SIZE))

    num_samples = fft.shape[1] // MAX_N
    fft = fft[:, : num_samples * MAX_N, :]

    fft = fft.reshape(
        fft.shape[0],   # n_channels
        num_samples,    # n_chunks
        MAX_N,          # patches per chunk
        fft.shape[-1],  # fft bins
    ).transpose(1, 0, 2, 3)

    for sample in range(fft.shape[0]):
        out_path = out_dir / f"{count:03d}_{sample:04d}.pkl"
        with open(out_path, "wb") as f:
            pickle.dump(fft[sample, :, :, :], f)

    save_dict[count] = file


def main() -> None:
    save_dict = {}

    out_dir = Path("Data/MAT_CHUNK_EXTRACTED")
    out_dir.mkdir(parents=True, exist_ok=True)

    mat_files = find_mat_files(Path("Data/MAT_DATA"))

    for count, file in tqdm(enumerate(mat_files), total=len(mat_files)):
        process_file(file, out_dir, count, save_dict)

    with open(out_dir / "lookup.pkl", "wb") as f:
        pickle.dump(save_dict, f)


if __name__ == "__main__":
    main()