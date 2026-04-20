"""
Self-Distillation Fine-Tuning (SDFT) trainer.

Implements self-distillation regularization for continual learning,
using a frozen reference model to preserve prior capabilities while
learning new alignment objectives.
"""

import os
import math
from typing import Dict, Optional, Any, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import copy

try:
    import torch_xla
    import torch_xla.core.xla_model as xm
    import torch_xla.distributed.xla_multiprocessing as xmp
    import torch_xla.distributed.parallel_loader as pl
    HAS_XLA = True
except ImportError:
    HAS_XLA = False

from ..models.model_loader import LlamaModel, create_reference_model
from .trainer_utils import (
    create_optimizer,
    create_scheduler,
    save_checkpoint_tpu,
    compute_loss_with_distillation,
    tpu_synced_metrics
)


class SDFTTrainer:
    """
    Self-Distillation Fine-Tuning Trainer.

    Uses a frozen reference model initialized from the same checkpoint as
the training model to provide soft targets for distillation regularization.

    Loss = (1-alpha) * CE(student, labels) + alpha * KL(teacher_soft || student_soft)

    where teacher is the frozen reference model.
    """

    def __init__(
        self,
        model: LlamaModel,
        config: Dict[str, Any],
        output_dir: str = "./experiments/exp_03_sdft",
        use_wandb: bool = True
    ):
        self.model = model
        self.config = config
        self.output_dir = output_dir
        self.device_type = config.get("device", {}).get("type", "tpu")
        self.use_wandb = use_wandb

        os.makedirs(output_dir, exist_ok=True)

        # SDFT specific parameters
        sdft_cfg = config.get("training", {}).get("sdft", {})
        self.alpha = sdft_cfg.get("alpha", 0.5)  # Distillation weight
        self.temperature = sdft_cfg.get("temperature", 2.0)  # Softmax temperature
        self.use_reference_model = sdft_cfg.get("use_reference_model", True)
        self.reference_model_path = sdft_cfg.get("reference_model_path")

        # General training parameters
        training_cfg = config.get("training", {})
        self.learning_rate = training_cfg.get("learning_rate", 1e-5)
        self.batch_size = training_cfg.get("batch_size", 4)
        self.gradient_accumulation_steps = training_cfg.get("gradient_accumulation_steps", 4)
        self.num_epochs = training_cfg.get("num_epochs", 5)
        self.warmup_ratio = training_cfg.get("warmup_ratio", 0.05)
        self.weight_decay = training_cfg.get("weight_decay", 0.01)
        self.max_grad_norm = training_cfg.get("max_grad_norm", 1.0)

        # Reference model (will be initialized before training)
        self.reference_model: Optional[LlamaModel] = None

        # Effective batch size
        self.effective_batch_size = (
            self.batch_size *
            self.gradient_accumulation_steps *
            (xm.xrt_world_size() if HAS_XLA and self.device_type == "tpu" else 1)
        )

        if use_wandb:
            self._init_wandb()

    def _init_wandb(self):
        """Initialize WandB logging."""
        try:
            import wandb
            wandb.init(
                project="llm-continual-alignment",
                name=f"sdft_{self.output_dir.split('/')[-1]}",
                config=self.config
            )
        except ImportError:
            print("Warning: wandb not available")
            self.use_wandb = False

    def _init_reference_model(self):
        """
        Initialize the frozen reference model for self-distillation.

        The reference model is initialized from the same checkpoint as the
        training model but remains frozen throughout training.
        """
        if not self.use_reference_model:
            return

        print("Initializing reference model for self-distillation...")

        if self.reference_model_path:
            # Load from specified checkpoint
            from ..models.model_loader import ModelLoader, ModelConfig

            loader = ModelLoader()
            ref_config = ModelConfig(
                model_name=self.model.config.model_name,
                cache_dir=self.model.config.cache_dir,
                torch_dtype=self.model.config.torch_dtype,
                device_type=self.device_type,
                use_lora=False
            )
            self.reference_model = loader.load_from_checkpoint(
                self.reference_model_path,
                ref_config,
                for_training=False
            )
        else:
            # Create from current model
            self.reference_model = create_reference_model(self.model)

        # Move to device and freeze
        if self.device_type == "tpu" and HAS_XLA:
            device = xm.xla_device()
            self.reference_model.model = self.reference_model.model.to(device)

        self.reference_model.freeze()

        print("Reference model initialized and frozen.")

    def compute_distillation_loss(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        labels: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute SDFT loss combining standard CE and distillation.

        Args:
            student_logits: Logits from training model
            teacher_logits: Logits from frozen reference model
            labels: Ground truth labels

        Returns:
            Total loss and dict of component losses
        """
        import torch.nn.functional as F

        # Standard cross-entropy loss (hard labels)
        shift_student_logits = student_logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        ce_loss = F.cross_entropy(
            shift_student_logits.view(-1, shift_student_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=-100
        )

        # Distillation loss (KL divergence with temperature scaling)
        shift_teacher_logits = teacher_logits[..., :-1, :].contiguous()

        # Apply temperature scaling
        student_soft = F.log_softmax(shift_student_logits / self.temperature, dim=-1)
        teacher_soft = F.softmax(shift_teacher_logits / self.temperature, dim=-1)

        kl_loss = F.kl_div(
            student_soft.view(-1, student_logits.size(-1)),
            teacher_soft.view(-1, teacher_logits.size(-1)),
            reduction="batchmean"
        ) * (self.temperature ** 2)

        # Combined loss
        total_loss = (1 - self.alpha) * ce_loss + self.alpha * kl_loss

        loss_dict = {
            "total_loss": total_loss.item(),
            "ce_loss": ce_loss.item(),
            "kl_loss": kl_loss.item(),
            "distillation_weight": self.alpha
        }

        return total_loss, loss_dict

    def train(
        self,
        train_dataset,
        eval_dataset=None,
        resume_from_checkpoint: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Run Self-Distillation Fine-Tuning.

        Args:
            train_dataset: Training dataset (typically StereoSet for alignment)
            eval_dataset: Evaluation dataset (optional)
            resume_from_checkpoint: Path to checkpoint to resume from

        Returns:
            Training metrics
        """
        print(f"\n{'='*60}")
        print(f"Starting Self-Distillation Fine-Tuning (SDFT)")
        print(f"Alpha (distillation weight): {self.alpha}")
        print(f"Temperature: {self.temperature}")
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

        # Initialize reference model
        self._init_reference_model()

        # Prepare training model
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
        total_ce_loss = 0.0
        total_kl_loss = 0.0

        for epoch in range(start_epoch, self.num_epochs):
            epoch_metrics = {
                "loss": 0.0,
                "ce_loss": 0.0,
                "kl_loss": 0.0,
                "batches": 0
            }

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

                # Student forward pass
                student_outputs = self.model.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=None
                )

                # Teacher forward pass (no gradient)
                if self.reference_model is not None:
                    with torch.no_grad():
                        teacher_outputs = self.reference_model.model(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            labels=None
                        )

                    # Compute distillation loss
                    loss, loss_dict = self.compute_distillation_loss(
                        student_outputs.logits,
                        teacher_outputs.logits,
                        labels
                    )
                else:
                    # Fallback to standard CE if no reference model
                    shift_logits = student_outputs.logits[..., :-1, :].contiguous()
                    shift_labels = labels[..., 1:].contiguous()
                    loss = F.cross_entropy(
                        shift_logits.view(-1, shift_logits.size(-1)),
                        shift_labels.view(-1),
                        ignore_index=-100
                    )
                    loss_dict = {
                        "total_loss": loss.item(),
                        "ce_loss": loss.item(),
                        "kl_loss": 0.0
                    }

                # Scale for gradient accumulation
                loss = loss / self.gradient_accumulation_steps
                loss.backward()

                # Track metrics
                total_loss += loss_dict["total_loss"]
                total_ce_loss += loss_dict["ce_loss"]
                total_kl_loss += loss_dict["kl_loss"]

                for k in ["loss", "ce_loss", "kl_loss"]:
                    epoch_metrics[k] += loss_dict.get(k, loss_dict.get("total_loss", 0))
                epoch_metrics["batches"] += 1

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
                        log_dict = {
                            "epoch": epoch + 1,
                            "step": global_step,
                            "loss": total_loss / 10,
                            "ce_loss": total_ce_loss / 10,
                            "kl_loss": total_kl_loss / 10,
                            "learning_rate": scheduler.get_last_lr()[0]
                        }

                        if self.use_wandb:
                            import wandb
                            wandb.log(log_dict, step=global_step)

                        progress_bar.set_postfix({
                            "loss": f"{log_dict['loss']:.4f}",
                            "ce": f"{log_dict['ce_loss']:.4f}",
                            "kl": f"{log_dict['kl_loss']:.4f}",
                            "lr": f"{log_dict['learning_rate']:.2e}"
                        })

                        total_loss = 0.0
                        total_ce_loss = 0.0
                        total_kl_loss = 0.0

                    # Save checkpoint
                    if global_step % 200 == 0 and rank == 0:
                        save_checkpoint_tpu(
                            self.model.model,
                            optimizer,
                            epoch,
                            global_step,
                            self.output_dir,
                            rank
                        )

            # End of epoch
            if rank == 0:
                avg_loss = epoch_metrics["loss"] / epoch_metrics["batches"]
                avg_ce = epoch_metrics["ce_loss"] / epoch_metrics["batches"]
                avg_kl = epoch_metrics["kl_loss"] / epoch_metrics["batches"]

                print(f"\nEpoch {epoch+1} completed.")
                print(f"  Avg Loss: {avg_loss:.4f}")
                print(f"  CE Loss: {avg_ce:.4f}")
                print(f"  KL Loss: {avg_kl:.4f}")

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
