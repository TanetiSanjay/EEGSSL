import os
import gc
import sys
import torch
import torch.distributed as dist

sys.path.insert(0, os.environ.get("NEUROLM_PATH", "./NeuroLM"))

from model.model_vq import VQ  
from model.model_neural_transformer import NTConfig
from Util.ModelSurgery.neurolmvq_to_customfft import patch_vq
from Util.DataLoader.ExtractedMAT_FFT.loader import create_distributed_dataloaders
from Util.Trainer.recon_trainer import DistributedVQTrainer
from dataclasses import dataclass
from datetime import timedelta

@dataclass
class TrainConfig:
    # Model & Data Paths
    vq_path: str = ".weights/NeuroLm/checkpoints/VQ.pt"
    data_root: str = "Data/MAT_CHUNK_EXTRACTED"
    fft_dim: int = 101
    n_channels: int = 128

    # DataLoader & Data Split
    batch_size: int = 16
    num_workers: int = 2
    val_split: float = 0.1
    seed: int = 42
    distributed: bool = True

    # Optimization & Training Setup
    num_epochs: int = 40
    learning_rate: float = 3e-4
    weight_decay: float = 0.05
    warmup_epochs: int = 10
    min_lr: float = 1e-6
    grad_accum_steps: int = 1
    grad_clip_norm: float = 1.0
    valid_interval: int = 1
    amp: bool = True

    # Directories & Checkpoints
    output_dir: str = "runs/vq_fft"
    checkpoint_dir: str = "runs/vq_fft/checkpoints"
    log_dir: str = "runs/vq_fft/logs"
    resume_from: str | None = None

def main():
    config = TrainConfig()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    
    if config.distributed:
        dist.init_process_group(
            backend="nccl",
            device_id=torch.device(f"cuda:{local_rank}"),  
            timeout=timedelta(hours=1)
        )

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

    del clean_state_dict
    gc.collect()

    model = patch_vq(model, fft_dim=101, n_channels=128)

    train_lodaer, valid_loader, _, _ = create_distributed_dataloaders(
        root_dir        = "Data/MAT_CHUNK_EXTRACTED",
        val_split       = 0.1,
        batch_size      = config.batch_size,
        num_workers     = 2,
        seed            = 42
    )

    trainer = DistributedVQTrainer(
        config,
        model,
        train_lodaer,
        valid_loader
    )

    trainer.fit()


if __name__ == '__main__':
    main()