import pickle 
import torch
import numpy as np

from scipy.io import loadmat
from scipy.signal import resample_poly
from tqdm import tqdm
from pathlib import Path
from math import gcd


MAX_N = 4
TARGET_FS = 250


def find_mat_files(
    root: Path
) -> list[Path]:
    return sorted(
        path for path in root.rglob("*.mat")
        if path.is_file()
    )


def extract_data(
    path: Path
):
    mat = loadmat(path)

    sampling_rate = None

    # Get sampling rate
    if "EEGSamplingRate" in mat:
        sampling_rate = np.asarray(mat["EEGSamplingRate"]).squeeze()

        if sampling_rate.size == 1:
            sampling_rate = float(sampling_rate)
        else:
            sampling_rate = None

    # Get EEG data
    for _, item in mat.items():
        if isinstance(item, np.ndarray):
            if item.ndim >= 2 and item.shape[0] == 129:
                return item[1:, :], sampling_rate

    return None, sampling_rate


def resample_eeg(
    data: np.ndarray,
    original_fs: float,
    target_fs: int = TARGET_FS,
):
    if original_fs is None:
        print("WARNING: EEGSamplingRate not found. Skipping resampling.")
        return data

    if original_fs == target_fs:
        return data

    print(
        f"Resampling EEG from {original_fs:g} Hz to {target_fs} Hz"
    )

    # Reduce the ratio first
    common = gcd(int(round(original_fs)), int(target_fs))
    up = target_fs // common
    down = int(round(original_fs)) // common

    # Resample along the time axis
    data = resample_poly(
        data,
        up=up,
        down=down,
        axis=1,
    )

    return data.astype(np.float32)


def patchify(
    data: np.ndarray,
    patch_size: int,
):
    n_channels, n_samples = data.shape
    n_patches = n_samples // patch_size

    return data[:, :n_patches * patch_size].reshape(
        n_channels,
        n_patches,
        patch_size,
    )


def compute_fft(
    patches: np.ndarray
):
    fft = np.log1p(
        np.abs(
            np.fft.rfft(
                patches,
                axis=-1,
            )
        )
    )

    return fft.astype(np.float32)


if __name__ == '__main__':

    save_dict = {}

    out_dir = Path("Data/MAT_CHUNK_EXTRACTED")
    out_dir.mkdir(parents=True, exist_ok=True)

    mat_files = find_mat_files(Path("Data/MAT_DATA"))

    for count, file in tqdm(
        enumerate(mat_files),
        total=len(mat_files),
    ):

        data, sampling_rate = extract_data(file)

        if data is None:
            print(f"WARNING: No EEG data found in {file}")
            continue

        # Resample BEFORE patchification
        data = resample_eeg(
            data,
            sampling_rate,
            TARGET_FS,
        )

        fft = compute_fft(
            patchify(
                data,
                200,
            )
        )

        num_samples = fft.shape[1] // MAX_N

        fft = fft[:, :num_samples * MAX_N, :]

        fft = fft.reshape(
            fft.shape[0],       # 128
            num_samples,        # number of chunks
            MAX_N,              # 4
            fft.shape[-1],      # 101
        ).transpose(1, 0, 2, 3)

        for sample in range(fft.shape[0]):
            with open(
                out_dir / f"{count:03d}_{sample:04d}.pkl",
                "wb",
            ) as f:
                pickle.dump(
                    fft[sample, :, :, :],
                    f,
                )

        save_dict[count] = file

    with open(
        out_dir / "lookup.pkl",
        "wb",
    ) as f:
        pickle.dump(
            save_dict,
            f,
        )