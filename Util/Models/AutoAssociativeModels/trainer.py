import logging
import os
import torch
import torch.nn as nn 


from pathlib import Path
from typing import Optional
from torch.optim import Adam
from tqdm.auto import tqdm
from torch.utils.data import DataLoader

class TqdmLoggingHandler(logging.Handler):
    def emit(self, record):
        try:
            tqdm.write(self.format(record))

        except Exception as e:
            self.handleError(record)


def build_logger(
    name    : str,
    log_dir : Optional[str]
) -> logging.Logger:
    logger  = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    if logger.handlers:
        return logger

    fmt = logging.Formatter(
        fmt=f"%(asctime)s | rank:0 | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console_handler = TqdmLoggingHandler()
    console_handler.setFormatter(fmt)
    console_handler.setLevel(logging.INFO)
    logger.addHandler(console_handler)

    if log_dir is not None:
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(Path(log_dir) / "train.log")
        file_handler.setFormatter(fmt)
        file_handler.setLevel(logging.INFO)
        logger.addHandler(file_handler)

    return logger


class SingleGPUAutoAssociativeTrainer:
    def __init__(
        self,
        config,
        model           : nn.Module,
        train_loader    : DataLoader,
        valid_loader    : DataLoader,
    ):
        self.config         = config
        self.train_loader   = train_loader
        self.valid_loader   = valid_loader

        self.logger         = build_logger("single_gpu_autoassociative_trainer", config.log_dir)
        self.model          = model.to(config.device)

        self.optimizer      = Adam(
            params          = self.build_param_groups(),
            lr              = self.config.learning_rate,
            weight_decay    = self.config.weight_decay, 
        )

        os.makedirs(self.config.output_dir, exist_ok=True)
        os.makedirs(self.config.checkpoint_dir, exist_ok=True)
        os.makedirs(self.config.log_dir, exist_ok=True)

        self.criterion      = nn.MSELoss()

        if self.config.resume_from is not None:
            self.load_checkpoint(self.config.resume_from)
            self.logger.info(f"[INFO] Loaded checkpoint from {self.config.resume_from} and will be trained.")


    def load_checkpoint(self, path: str) -> None:
        checkpoint = torch.load(path, map_location=self.device)

        print("Checkpoint keys:", checkpoint.keys())
        print("First checkpoint key:", next(iter(checkpoint["model"])))
        print("First model key:", next(iter(self.model.state_dict())))

        self.model.load_state_dict(checkpoint["model"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])

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
                "epoch"             : epoch,
                "best_valid_loss"   : self.best_valid_loss,
            }
    
            path    = Path(self.config.checkpoint_dir) / filename
            torch.save(checkpoint, path)
    
            self.logger.info(f"Saved checkpoint: {path}")


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


    def train_epoch(self, epoch : int) -> float:
        self.model.train()

        total_loss = 0.0

        self.optimizer.zero_grad(set_to_none=True)

        pbar    = tqdm(
            enumerate(self.train_loader),
            total   = len(self.train_loader),
            desc    = f"Training [{epoch + 1} / {self.config.num_epochs}]"
        )

        for step, batch in pbar:
            x, y        = self.unpack_batch(batch, device = self.config.device)
            y_rec       = self.model(x)
            loss        = self.criterion(y, y_rec)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

            pbar.set_postfix(
                loss        = f"{total_loss / (step+1):.04f}",
                step_loss   = f"{loss:.04f}",
                lr          = f"{self.optimizer.param_groups[0]["lr"]:.2e}"
            )


        return (total_loss / (step + 1))


    @torch.no_grad()
    def valid_epoch(self, epoch : int) -> float:
        self.model.eval()

        total_loss = 0.0        

        pbar       = tqdm(
            enumerate(self.valid_loader),
            total   = len(self.valid_loader),
            desc    = f"Validating [{epoch + 1}]"
        )

        for step, batch in pbar:
            x       = self.unpack_batch(batch, device = self.config.device)
            x_rec   = self.model(x)
            loss    = self.criterion(x, x_rec)

            total_loss += loss.item()

            pbar.set_postfix(
                loss        = f"{total_loss / (step+1):.04f}",
                step_loss   = f"{loss:.04f}",
            )

        return (total_loss / (step + 1))


    def fit(self) -> None:
        for epoch in range(self.start_epoch, self.config.num_epochs):
            train_loss = 0.0
            train_loss = self.train_epoch(epoch)
            validated  = False 

            self.logger.info(f"EPOCH {epoch} | train_loss = {train_loss:.4f}")

            if epoch % self.config.valid_interval == 0:
                val_rec_loss = self.valid_epoch(epoch)

                self.logger.info(f"EPOCH {epoch} | val_rec_loss = {val_rec_loss:.4f}")

                if val_rec_loss < self.best_valid_loss:
                    self.best_valid_loss    = val_rec_loss
                    self.save_checkpoint("best_test.pt", epoch)
                    self.logger.info(f"[INFO] New best val_rec_loss = {val_rec_loss:.4f}")

                validated = True

            self.save_checkpoint("latest.pt", epoch)


            if epoch == self.config.num_epochs - 1 and not validated:
                val_rec_loss = self.valid_epoch(epoch)

                if self.is_main_process:
                    self.logger.info(f"EPOCH {epoch} | val_rec_loss = {val_rec_loss:.4f}")

                    if val_rec_loss < self.best_valid_loss:
                        self.best_valid_loss = val_rec_loss
                        self.save_checkpoint("best_test.pt", epoch)
                        self.logger.info(f"[INFO] New best val_rec_loss = {val_rec_loss:.4f}")

        self.logger.info("[INFO] Training complete!!!")


if __name__ == '__main__':
    trainer = SingleGPUAutoAssociativeTrainer()
    trainer.fit()
