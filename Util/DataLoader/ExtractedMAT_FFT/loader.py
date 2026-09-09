import torch
import os
import torch.nn as nn 
import torch.nn.functional as F 
import pickle

from torch.utils.data import DataLoader, Dataset, random_split
from torch.utils.data.distributed import DistributedSampler
from pathlib import Path


class FFTDataLoader(Dataset):
    def __init__(
        self, 
        root_dir: Path | str,
    ):
        super().__init__()

        root = Path(root_dir) if isinstance(root_dir, str) else root_dir
        if not root.exists(): 
            raise FileNotFoundError(f"Directory {root} does not exist.")

        # Keep deterministic order so all DDP ranks see the exact same dataset ordering
        self.items = sorted([root/i for i in os.listdir(root) if i.endswith(".pkl") and "look" not in i])

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        file = self.items[idx]

        with open(file, "rb") as f: 
            data = pickle.load(f)

        tensor_data = torch.from_numpy(data) if not isinstance(data, torch.Tensor) else data

        return {
            "x": tensor_data,
            "input_mask": torch.ones(tensor_data.shape[-2], dtype=torch.long)
        }


def create_distributed_dataloaders(
    root_dir: str | Path,
    val_split: float = 0.2,
    batch_size: int = 32,
    num_workers: int = 4,
    seed: int = 42,
    pin_memory: bool = True
) -> tuple[DataLoader, DataLoader, DistributedSampler, DistributedSampler]:
    """
    Splits dataset into train/val subsets and returns distributed DataLoaders and Samplers.
    """
    dataset = FFTDataLoader(root_dir=root_dir)

    # 1. Calculate train/val lengths
    val_size = int(len(dataset) * val_split)
    train_size = len(dataset) - val_size

    # 2. Perform train/val split using a fixed-seed Generator (consistent across all DDP ranks)
    generator = torch.Generator().manual_seed(seed)
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size], generator=generator)

    # 3. Create Distributed Samplers for train and val
    train_sampler = DistributedSampler(
        train_dataset,
        shuffle=True,
        seed=seed,
        drop_last=True
    )

    val_sampler = DistributedSampler(
        val_dataset,
        shuffle=False,  # No need to shuffle validation set
        seed=seed,
        drop_last=False
    )

    # 4. Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        sampler=val_sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0
    )

    return train_loader, val_loader, train_sampler, val_sampler

def create_dataloaders(
    root_dir: str | Path,
    val_split: float = 0.2,
    batch_size: int = 32,
    num_workers: int = 4,
    seed: int = 42,
    pin_memory: bool = True,
) -> tuple[DataLoader, DataLoader]:

    dataset = FFTDataLoader(root_dir=root_dir)

    # Train/validation split
    val_size = int(len(dataset) * val_split)
    train_size = len(dataset) - val_size

    generator = torch.Generator().manual_seed(seed)

    train_dataset, val_dataset = random_split(
        dataset,
        [train_size, val_size],
        generator=generator
    )

    # Normal DataLoaders — no DistributedSampler
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
    )

    return train_loader, val_loader

if __name__ == '__main__':
    # Example usage inside your DDP script:
    train_loader, val_loader, train_sampler, val_sampler = create_distributed_dataloaders(
        root_dir="Data/MAT_CHUNK_EXTRACTED",
        val_split=0.2,
        batch_size=16,
        num_workers=2,
        seed=42
    )

    print(f"Train samples per GPU: {len(train_loader.dataset)} across {len(train_loader)} batches")
    print(f"Val samples per GPU: {len(val_loader.dataset)} across {len(val_loader)} batches")

    # In your training loop:
    # for epoch in range(epochs):
    #     train_sampler.set_epoch(epoch) # Re-shuffle training set per epoch
    #     ...