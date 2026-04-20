"""
Standard Supervised Fine-Tuning (SFT) trainer.

This is standard fine-tuning without self-distillation,
used as a baseline comparison for continual learning experiments.
"""

import os
import math
from typing import Dict, Optional, Any, List
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling

try:
    import torch_xla
    import torch_xla.core.xla_model as xm
    import torch_xla.distributed.xla_multiprocessing as xmp
    import torch_xla.distributed.parallel_loader as pl
    from torch_xla.amp import autocast, GradScaler
    HAS_XLA = True
except ImportError:
    HAS_XLA = False

from ..models.model_loader import LlamaModel
from .trainer_utils import (
    create_optimizer,
    create_scheduler,
    get_training_arguments,
    BiasEvaluationCallback,
    WandBLoggingCallback,
    save_checkpoint_tpu,
    tpu_synced_metrics
)


class SFTTrainer:
    """
    Standard Supervised Fine-Tuning Trainer.

    Implements basic next-token prediction loss without distillation.
    """

    def __init__(
        self,
        model: LlamaModel,
        config: Dict[str, Any],
        output_dir: str = "./experiments/exp_02_sft",
        use_wandb: bool = True
    ):
        self.model = model
        self.config = config
        self.output_dir = output_dir
        self.device_type = config.get("device", {}).get("type", "tpu")
        self.use_wandb = use_wandb

        os.makedirs(output_dir, exist_ok=True)

        # Training hyperparameters
        training_cfg = config.get("training", {})
        self.learning_rate = training_cfg.get("learning_rate", 2e-5)
        self.batch_size = training_cfg.get("batch_size", 4)
        self.gradient_accumulation_steps = training_cfg.get("gradient_accumulation_steps", 4)
        self.num_epochs = training_cfg.get("num_epochs", 3)
        self.warmup_ratio = training_cfg.get("warmup_ratio", 0.03)
        self.weight_decay = training_cfg.get("weight_decay", 0.01)
        self.max_grad_norm = training_cfg.get("max_grad_norm", 1.0)

        # Effective batch size
        self.effective_batch_size = (
            self.batch_size *
            self.gradient_accumulation_steps *
            (xm.xrt_world_size() if HAS_XLA and self.device_type == "tpu" else 1)
        )

        # Initialize WandB if enabled
        if use_wandb:
            self._init_wandb()

    def _init_wandb(self):
        """Initialize WandB logging."""
        try:
            import wandb
            wandb.init(
                project="llm-continual-alignment",
                name=f"sft_{self.output_dir.split('/')[-1]}",
                config=self.config
            )
        except ImportError:
            print("Warning: wandb not available")
            self.use_wandb = False

    def compute_loss(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute standard cross-entropy loss.

        Args:
            logits: Model output logits [batch, seq_len, vocab_size]
            labels: Target labels [batch, seq_len]

        Returns:
            Loss tensor
        """
        # Shift for next-token prediction
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        # Flatten for cross-entropy
        loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=-100
        )

        return loss

    def train(
        self,
        train_dataset,
        eval_dataset = None,
        resume_from_checkpoint: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Run standard fine-tuning training.

        Args:
            train_dataset: Training dataset
            eval_dataset: Evaluation dataset (optional)
            resume_from_checkpoint: Path to checkpoint to resume from

        Returns:
            Training metrics
        """
        print(f"\n{'='*60}")
        print(f"Starting Standard Fine-Tuning (SFT)")
        print(f"{'='*60}")

        # Setup device
        if self.device_type == "tpu" and HAS_XLA:
            device = xm.xla_device()
            world_size = xm.xrt_world_size()
            rank = xm.get_ordinal()
            print(f"TPU Device: {device}, Rank: {rank}/{world_size}")
        else:
            device = next(self.model.model.parameters()).device
            world_size = 1
            rank = 0
            print(f"Device: {device}")

        # Prepare model
        self.model.model.train()
        self.model.model.to(device)

        # Create optimizer and scheduler
        num_training_steps = (
            len(train_dataset) // self.effective_batch_size
        ) * self.num_epochs

        optimizer = create_optimizer(
            self.model.model,
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
            device_type=self.device_type
        )

        scheduler = create_scheduler(
            optimizer,
            num_training_steps=num_training_steps,
            warmup_ratio=self.warmup_ratio
        )

        # Load checkpoint if resuming
        start_epoch = 0
        start_step = 0
        if resume_from_checkpoint:
            checkpoint = torch.load(resume_from_checkpoint, map_location=device)
            self.model.model.load_state_dict(checkpoint["model_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            start_epoch = checkpoint.get("epoch", 0)
            start_step = checkpoint.get("step", 0)
            print(f"Resumed from checkpoint: {resume_from_checkpoint}")

        # Create dataloader
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=True
        )

        # Wrap for TPU if needed
        if self.device_type == "tpu" and HAS_XLA:
            train_loader = pl.ParallelLoader(
                train_loader,
                [device]
            ).per_device_loader(device)

        # Training loop
        global_step = start_step
        total_loss = 0.0

        for epoch in range(start_epoch, self.num_epochs):
            epoch_loss = 0.0
            num_batches = 0

            progress_bar = tqdm(
                train_loader,
                desc=f"Epoch {epoch+1}/{self.num_epochs}",
                disable=rank != 0
            )

            optimizer.zero_grad()

            for batch_idx, batch in enumerate(progress_bar):
                # Move to device
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch.get("attention_mask", None)
                if attention_mask is not None:
                    attention_mask = attention_mask.to(device)
                labels = batch["labels"].to(device)

                # Forward pass
                outputs = self.model.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=None  # We'll compute loss manually
                )

                # Compute loss
                loss = self.compute_loss(outputs.logits, labels)

                # Scale loss for gradient accumulation
                loss = loss / self.gradient_accumulation_steps
                loss.backward()

                total_loss += loss.item() * self.gradient_accumulation_steps
                epoch_loss += loss.item() * self.gradient_accumulation_steps

                # Update weights
                if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                    # Clip gradients
                    if self.max_grad_norm > 0:
                        torch.nn.utils.clip_grad_norm_(
                            self.model.model.parameters(),
                            self.max_grad_norm
                        )

                    # Optimizer step
                    if self.device_type == "tpu" and HAS_XLA:
                        xm.optimizer_step(optimizer)
                    else:
                        optimizer.step()

                    optimizer.zero_grad()
                    scheduler.step()
                    global_step += 1

                    # Logging
                    if rank == 0 and global_step % 10 == 0:
                        avg_loss = total_loss / 10
                        lr = scheduler.get_last_lr()[0]

                        log_dict = {
                            "epoch": epoch + 1,
                            "step": global_step,
                            "loss": avg_loss,
                            "learning_rate": lr
                        }

                        if self.use_wandb:
                            import wandb
                            wandb.log(log_dict)

                        progress_bar.set_postfix(log_dict)
                        total_loss = 0.0

                    # Save checkpoint
                    if global_step % 500 == 0 and rank == 0:
                        save_checkpoint_tpu(
                            self.model.model,
                            optimizer,
                            epoch,
                            global_step,
                            self.output_dir,
                            rank
                        )

            # End of epoch
            avg_epoch_loss = epoch_loss / len(train_loader)

            if rank == 0:
                print(f"\nEpoch {epoch+1} completed. Avg loss: {avg_epoch_loss:.4f}")

                # Save epoch checkpoint
                self.model.save(os.path.join(self.output_dir, f"epoch_{epoch+1}"))

            # Sync at epoch end for TPU
            if self.device_type == "tpu" and HAS_XLA:
                xm.rendezvous("epoch_end")

        # Final save
        if rank == 0:
            self.model.save(os.path.join(self.output_dir, "final"))
            print(f"\nTraining complete. Model saved to {self.output_dir}/final")

        if self.use_wandb and rank == 0:
            import wandb
            wandb.finish()

        return {
            "final_step": global_step,
            "num_epochs": self.num_epochs,
            "output_dir": self.output_dir
        }
