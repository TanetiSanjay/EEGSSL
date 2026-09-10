import logging
import os
import torch
import torch.distributed as dist
import torch.nn as nn

from pathlib import Path
from typing import Optional
from torch.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR


class TqdmLoggingHandler(logging.Handler):
    def emit(self, record):
        try:
            tqdm.write(self.format(record))

        except Exception:
            self.handleError(record)


def build_logger(name: str, log_dir: Optional[str], rank: int, is_main_process: bool) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    if logger.handlers:
        return logger

    fmt = logging.Formatter(
        fmt=f"%(asctime)s | rank{rank} | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console_handler = TqdmLoggingHandler()
    console_handler.setFormatter(fmt)
    console_handler.setLevel(logging.INFO if is_main_process else logging.WARNING)
    logger.addHandler(console_handler)

    if log_dir is not None and is_main_process:
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(Path(log_dir) / "train.log")
        file_handler.setFormatter(fmt)
        file_handler.setLevel(logging.INFO)
        logger.addHandler(file_handler)

    return logger


class DistributedVQTrainer:
    def __init__(
        self,
        config,
        model           : nn.Module,
        train_loader    : DataLoader,
        valid_loader    : DataLoader
    ):
        self.config         = config
        self.train_loader   = train_loader
        self.valid_loader   = valid_loader
        self.setup_distributed()

        self.logger = build_logger(
            "ddp_vq_trainer", config.log_dir, self.rank, self.is_main_process
        )

        self.model = model.to(self.device)

        self.optimizer = AdamW(
            params          = self.build_param_groups(),
            lr              = self.config.learning_rate,
            weight_decay    = self.config.weight_decay
        )

        self.grad_accum_steps = max(1, getattr(self.config, "grad_accum_steps", 1))
        self.steps_per_epoch  = max(len(self.train_loader) // self.grad_accum_steps, 1)

        total_steps     = self.config.num_epochs * self.steps_per_epoch
        warmup_epochs   = getattr(self.config, "warmup_epochs", 10)
        warmup_steps    = warmup_epochs * self.steps_per_epoch

        warmup_scheduler = LinearLR(
            self.optimizer,
            start_factor    = 1e-3,  # Ramps up from 0.1% of base LR
            end_factor      = 1.0,
            total_iters     = warmup_steps
        )

        cosine_scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=max(1, total_steps - warmup_steps),
            eta_min=getattr(self.config, "min_lr", 1e-6)
        )

        self.scheduler = SequentialLR(
            self.optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[warmup_steps]
        )

        self.scaler = GradScaler(self.device.type, enabled=self.config.amp)

        self.start_epoch     = 0
        self.best_valid_loss = float("inf")

        os.makedirs(self.config.output_dir, exist_ok=True)
        os.makedirs(self.config.checkpoint_dir, exist_ok=True)
        os.makedirs(self.config.log_dir, exist_ok=True)

        if self.config.resume_from is not None:
            self.load_checkpoint(self.config.resume_from)
            self.logger.info(f"[INFO] Loaded checkpoint from {self.config.resume_from} and will be trained.")


        if self.is_distributed:
            self.model = DDP(
                self.model,
                device_ids      = [self.local_rank] if self.device.type == "cuda" else None,
                output_device   = self.local_rank if self.device.type == "cuda" else None,
                find_unused_parameters=True,
            )


    def setup_distributed(self):
        self.rank       = int(os.environ.get("RANK", 0))
        self.world_size = int(os.environ.get("WORLD_SIZE", 1))
        self.local_rank = int(os.environ.get("LOCAL_RANK", 0))

        self.is_distributed = self.world_size > 1

        if self.is_distributed:
            torch.cuda.set_device(self.local_rank)
            self.device = torch.device(f"cuda:{self.local_rank}")

        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    def build_param_groups(self):
        decay, no_decay = [], []

        for name, param in self.model.named_parameters():
            if not param.requires_grad: continue

            if param.ndim <= 1 or "norm" in name.lower() or "bias" in name.lower(): no_decay.append(param)
            else: decay.append(param)

        return [
            {
                "params": decay,
                "weight_decay": self.config.weight_decay
            },
            {
                "params": no_decay,
                "weight_decay": 0.0
            }
        ]


    def load_checkpoint(self, path: str) -> None:
        checkpoint = torch.load(path, map_location=self.device)

        print("Checkpoint keys:", checkpoint.keys())
        print("First checkpoint key:", next(iter(checkpoint["model"])))
        print("First model key:", next(iter(self.model.state_dict())))

        self.model.load_state_dict(checkpoint["model"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.scaler.load_state_dict(checkpoint["scaler"])

        if "scheduler" in checkpoint and hasattr(self, "scheduler"):
            self.scheduler.load_state_dict(checkpoint["scheduler"])

        self.start_epoch     = checkpoint["epoch"] + 1
        self.best_valid_loss = checkpoint["best_valid_loss"]


    def save_checkpoint(
        self,
        filename    : str,
        epoch       : int,
    ) -> None:

        if not self.is_main_process: return

        checkpoint = {
            "model"             : self.raw_model.state_dict(),
            "optimizer"         : self.optimizer.state_dict(),
            "scheduler"         : self.scheduler.state_dict(),
            "scaler"            : self.scaler.state_dict(),
            "epoch"             : epoch,
            "best_valid_loss"   : self.best_valid_loss,
        }

        path    = Path(self.config.checkpoint_dir) / filename
        torch.save(checkpoint, path)

        self.logger.info(f"Saved checkpoint: {path}")

    @property
    def is_main_process(self): return self.rank == 0

    @property
    def raw_model(self): return self.model.module if self.is_distributed else self.model


    def reduce_mean(self, value : float) -> float:
        if not self.is_distributed: return value

        tensor = torch.tensor(value, device=self.device)
        dist.all_reduce(tensor, op = dist.ReduceOp.SUM)

        return (tensor / self.world_size).item()


    def reduce_sum(self, value: torch.Tensor) -> torch.Tensor:
        if not self.is_distributed: return value

        dist.all_reduce(value, op=dist.ReduceOp.SUM)

        return value


    def _unpack_batch(self, batch):
        x           = batch["x"].to(self.device, non_blocking=True)
        input_mask  = batch["input_mask"].to(self.device, non_blocking=True)

        input_chans = batch.get("input_chans", None)
        input_time  = batch.get("input_time", None)

        if input_chans is not None: input_chans = input_chans.to(self.device, non_blocking=True)
        if input_time  is not None: input_time  = input_time.to(self.device, non_blocking=True)

        return x, input_chans, input_time, input_mask.bool()


    def train_epoch(self, epoch : int) -> float:
        self.model.train()

        if self.is_distributed and hasattr(self.train_loader, "sampler"):
            sampler = self.train_loader.sampler
            if hasattr(sampler, "set_epoch"):
                sampler.set_epoch(epoch)

        total_loss, num_batches = 0.0, 0
        total_rec_loss          = 0.0

        self.optimizer.zero_grad(set_to_none=True)

        pbar = tqdm(
            enumerate(self.train_loader),
            total   = len(self.train_loader),
            desc    = f"Training [{epoch + 1} / {self.config.num_epochs}]",
            disable = not self.is_main_process
        )

        for step, batch in pbar:
            x, input_chans, input_time, input_mask = self._unpack_batch(batch)

            with autocast(self.device.type, enabled=self.config.amp):
                loss, _, log, _ = self.model(
                    x,
                    x,
                    input_chans = input_chans,
                    input_time  = input_time,
                    input_mask  = input_mask,
                )
                loss = loss / self.grad_accum_steps

            self.scaler.scale(loss).backward()

            is_accum_boundary = ((step + 1) % self.grad_accum_steps == 0) or ((step + 1) == len(self.train_loader))

            if is_accum_boundary:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip_norm)

                scale_before = self.scaler.get_scale()
                self.scaler.step(self.optimizer)
                self.scaler.update()
                scale_after  = self.scaler.get_scale()

                if scale_before <= scale_after: self.scheduler.step()

                self.optimizer.zero_grad(set_to_none=True)

            step_loss     = loss.item() * self.grad_accum_steps
            total_loss   += step_loss
            total_rec_loss += log.get(f"{'train' if self.model.training else 'val'}/rec_fft_loss", 0.0)
            num_batches  += 1

            if self.is_main_process:
                current_lr = self.optimizer.param_groups[0]["lr"]
                pbar.set_postfix(
                    loss        = f"{step_loss:.4f}",
                    avg         = f"{total_loss/num_batches:.4f}",
                    rec         = f"{total_rec_loss/num_batches:.4f}",
                    current_lr  = f"{current_lr:.2e}"
                )

        return self.reduce_mean(total_loss / max(num_batches, 1))


    @torch.no_grad()
    def valid_epoch(self, epoch: int) -> float:
        self.model.eval()

        if self.is_distributed and hasattr(self.valid_loader, "sampler"):
            sampler = self.valid_loader.sampler
            if hasattr(sampler, "set_epoch"):
                sampler.set_epoch(epoch)

        total_loss, total_rec_loss, num_batches = 0.0, 0.0, 0

        pbar = tqdm(
            self.valid_loader,
            desc    = f"Validation [{epoch + 1}/{self.config.num_epochs}]",
            disable = not self.is_main_process,
        )

        for batch in pbar:
            x, input_chans, input_time, input_mask = self._unpack_batch(batch)

            with autocast(self.device.type, enabled=self.config.amp):
                loss, encoder_features, log, _ = self.model(
                    x,
                    x,
                    input_chans = input_chans,
                    input_time  = input_time,
                    input_mask  = input_mask,
                )

            rec_fft_loss = log.get("val/rec_fft_loss", log.get("train/rec_fft_loss", loss.item()))

            total_loss     += loss.item()
            total_rec_loss += float(rec_fft_loss)
            num_batches    += 1

            if self.is_main_process:
                pbar.set_postfix(
                    loss = f"{loss.item():.4f}",
                    rec  = f"{total_rec_loss/num_batches:.4f}",
                )

            del x, input_mask, encoder_features

        return self.reduce_mean(total_rec_loss / max(num_batches, 1))


    def fit(self) -> None:
        if self.is_main_process:
            self.logger.info(f"Started training | epochs = {self.start_epoch}->{self.config.num_epochs}")

        if self.is_distributed: dist.barrier()

        for epoch in range(self.start_epoch, self.config.num_epochs):
            train_loss = 0.0
            train_loss = self.train_epoch(epoch)
            validated  = False

            if self.is_main_process:
                self.logger.info(f"EPOCH {epoch} | train_loss = {train_loss:.4f}")

            if epoch % self.config.valid_interval == 0:
                val_rec_loss = self.valid_epoch(epoch)

                if self.is_main_process:
                    self.logger.info(f"EPOCH {epoch} | val_rec_loss = {val_rec_loss:.4f}")

                    if val_rec_loss < self.best_valid_loss:
                        self.best_valid_loss = val_rec_loss
                        self.save_checkpoint("best_test.pt", epoch)
                        self.logger.info(f"[INFO] New best val_rec_loss = {val_rec_loss:.4f}")
                    validated = True

            self.save_checkpoint("latest.pt", epoch)

            if self.is_distributed: dist.barrier()

            if epoch == self.config.num_epochs - 1 and not validated:
                val_rec_loss = self.valid_epoch(epoch)

                if self.is_main_process:
                    self.logger.info(f"EPOCH {epoch} | val_rec_loss = {val_rec_loss:.4f}")

                    if val_rec_loss < self.best_valid_loss:
                        self.best_valid_loss = val_rec_loss
                        self.save_checkpoint("best_test.pt", epoch)
                        self.logger.info(f"[INFO] New best val_rec_loss = {val_rec_loss:.4f}")


        if self.is_distributed: dist.barrier()
        self.logger.info("[INFO] Training complete!!!")