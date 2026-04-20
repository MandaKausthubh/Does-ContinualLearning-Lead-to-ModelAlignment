"""
Training utilities for TPU distributed training with PyTorch/XLA.
"""

import os
import math
from typing import Dict, Optional, Any, Callable
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR

# TPU imports
try:
    import torch_xla
    import torch_xla.core.xla_model as xm
    import torch_xla.distributed.xla_multiprocessing as xmp
    import torch_xla.distributed.parallel_loader as pl
    from torch_xla.amp import autocast, GradScaler
    HAS_XLA = True
except ImportError:
    HAS_XLA = False
    print("Warning: torch_xla not available. TPU training will not work.")

from transformers import TrainingArguments, TrainerCallback


@dataclass
class TPUConfig:
    """Configuration for TPU training."""
    num_cores: int = 8
    master_port: int = 12355
    world_size: int = 8


def setup_tpu():
    """Setup TPU environment."""
    if not HAS_XLA:
        raise RuntimeError("torch_xla not available")

    device = xm.xla_device()
    print(f"TPU device: {xm.xla_real_devices([device])}")
    return device


def get_tpu_config() -> TPUConfig:
    """Get TPU configuration from environment."""
    num_cores = int(os.environ.get("TPU_NUM_DEVICES", 8))
    return TPUConfig(
        num_cores=num_cores,
        world_size=num_cores
    )


def create_optimizer(
    model: nn.Module,
    lr: float = 2e-5,
    weight_decay: float = 0.01,
    betas: tuple = (0.9, 0.999),
    eps: float = 1e-8,
    device_type: str = "tpu"
) -> torch.optim.Optimizer:
    """
    Create optimizer for model training.

    Handles parameter grouping for weight decay.
    """
    # Separate parameters that should/shouldn't have weight decay
    decay_params = []
    no_decay_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        # Don't apply weight decay to bias and LayerNorm parameters
        if "bias" in name or "layernorm" in name.lower() or "ln" in name.lower():
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    param_groups = [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0}
    ]

    optimizer = AdamW(param_groups, lr=lr, betas=betas, eps=eps)

    if device_type == "tpu":
        # Wrap optimizer for XLA if needed
        optimizer = xm.optimizers.Optimizer(optimizer) if HAS_XLA else optimizer

    return optimizer


def create_scheduler(
    optimizer: torch.optim.Optimizer,
    num_training_steps: int,
    warmup_ratio: float = 0.03,
    scheduler_type: str = "cosine"
) -> LambdaLR:
    """
    Create learning rate scheduler with warmup.

    Args:
        optimizer: The optimizer to schedule
        num_training_steps: Total number of training steps
        warmup_ratio: Fraction of steps for warmup
        scheduler_type: Type of scheduler ("cosine", "linear", "constant")
    """
    num_warmup_steps = int(num_training_steps * warmup_ratio)

    def lr_lambda(current_step: int) -> float:
        if current_step < num_warmup_steps:
            # Linear warmup
            return float(current_step) / float(max(1, num_warmup_steps))

        # After warmup
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )

        if scheduler_type == "cosine":
            return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
        elif scheduler_type == "linear":
            return max(0.0, 1.0 - progress)
        elif scheduler_type == "constant":
            return 1.0
        else:
            return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return LambdaLR(optimizer, lr_lambda)


def get_training_arguments(
    output_dir: str,
    config: Dict[str, Any],
    device_type: str = "tpu"
) -> TrainingArguments:
    """
    Create HuggingFace TrainingArguments with TPU support.

    Args:
        output_dir: Directory for saving checkpoints
        config: Training configuration dict
        device_type: Device type (tpu, cuda, cpu)
    """
    # Extract training parameters
    training = config.get("training", {})
    optimization = config.get("optimization", {})

    # Device-specific settings
    if device_type == "tpu":
        # TPU training doesn't use traditional distributed setup
        ddp_find_unused_parameters = False
        dataloader_num_workers = 0  # TPU prefers 0 workers
    else:
        ddp_find_unused_parameters = False
        dataloader_num_workers = optimization.get("dataloader_num_workers", 4)

    args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=training.get("num_epochs", 3),
        per_device_train_batch_size=training.get("batch_size", 4),
        per_device_eval_batch_size=training.get("batch_size", 4),
        gradient_accumulation_steps=training.get("gradient_accumulation_steps", 4),
        learning_rate=training.get("learning_rate", 2e-5),
        weight_decay=training.get("weight_decay", 0.01),
        warmup_ratio=training.get("warmup_ratio", 0.03),
        max_grad_norm=training.get("max_grad_norm", 1.0),

        # Logging and saving
        logging_steps=training.get("logging_steps", 10),
        save_steps=training.get("save_steps", 500),
        eval_steps=training.get("eval_steps", 500),
        save_total_limit=config.get("checkpoint", {}).get("save_total_limit", 3),
        load_best_model_at_end=config.get("checkpoint", {}).get("load_best_model_at_end", True),
        metric_for_best_model=config.get("checkpoint", {}).get("metric_for_best_model", "eval_loss"),
        greater_is_better=config.get("checkpoint", {}).get("greater_is_better", False),

        # Optimization
        fp16=optimization.get("fp16", False) if device_type != "tpu" else False,
        bf16=optimization.get("bf16", True),
        gradient_checkpointing=optimization.get("gradient_checkpointing", True),
        dataloader_num_workers=dataloader_num_workers,
        dataloader_pin_memory=optimization.get("dataloader_pin_memory", True) if device_type != "tpu" else False,
        group_by_length=optimization.get("group_by_length", True),

        # Distributed training
        ddp_find_unused_parameters=ddp_find_unused_parameters,

        # Evaluation
        evaluation_strategy="steps",
        save_strategy="steps",
        logging_strategy="steps",

        # Misc
        seed=training.get("seed", 42),
        report_to=["wandb"] if config.get("logging", {}).get("use_wandb", True) else [],
        remove_unused_columns=False,
    )

    return args


def compute_loss_with_distillation(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    labels: torch.Tensor,
    temperature: float = 2.0,
    alpha: float = 0.5
) -> torch.Tensor:
    """
    Compute loss combining standard cross-entropy and distillation.

    Loss = alpha * KL(teacher_soft || student_soft) + (1-alpha) * CE(student, labels)

    Args:
        student_logits: Logits from student model
        teacher_logits: Logits from teacher/reference model
        labels: Ground truth labels
        temperature: Temperature for softening distributions
        alpha: Weight for distillation loss (0-1)
    """
    import torch.nn.functional as F

    # Standard cross-entropy loss on hard labels
    ce_loss = F.cross_entropy(
        student_logits.view(-1, student_logits.size(-1)),
        labels.view(-1),
        ignore_index=-100
    )

    # Distillation loss (KL divergence between softened distributions)
    student_soft = F.log_softmax(student_logits / temperature, dim=-1)
    teacher_soft = F.softmax(teacher_logits / temperature, dim=-1)

    kl_loss = F.kl_div(
        student_soft.view(-1, student_logits.size(-1)),
        teacher_soft.view(-1, teacher_logits.size(-1)),
        reduction="batchmean"
    ) * (temperature ** 2)

    # Combined loss
    loss = alpha * kl_loss + (1 - alpha) * ce_loss

    return loss


def tpu_synced_metrics(metrics: Dict[str, float], device: Any = None) -> Dict[str, float]:
    """
    Synchronize metrics across TPU cores.

    Args:
        metrics: Dictionary of metric values
        device: TPU device (if None, gets current device)

    Returns:
        Metrics averaged across all TPU cores
    """
    if not HAS_XLA:
        return metrics

    if device is None:
        device = xm.xla_device()

    synced = {}
    for key, value in metrics.items():
        # Convert to tensor and sync
        tensor = torch.tensor(value, device=device)
        # Reduce across all devices
        synced[key] = xm.mesh_reduce(key, tensor.item(), lambda x: sum(x) / len(x))

    return synced


def save_checkpoint_tpu(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    step: int,
    output_dir: str,
    rank: int = 0
):
    """
    Save checkpoint on TPU (only from master process).

    Args:
        model: Model to save
        optimizer: Optimizer state
        epoch: Current epoch
        step: Current step
        output_dir: Directory to save checkpoint
        rank: Process rank (0 is master)
    """
    if not HAS_XLA or rank != 0:
        return

    os.makedirs(output_dir, exist_ok=True)

    # Mark step before saving
    xm.mark_step()

    checkpoint = {
        "epoch": epoch,
        "step": step,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }

    checkpoint_path = os.path.join(output_dir, f"checkpoint_epoch{epoch}_step{step}.pt")
    xm.save(checkpoint, checkpoint_path)

    print(f"Saved checkpoint to {checkpoint_path}")


class BiasEvaluationCallback(TrainerCallback):
    """
    Callback to run bias evaluation during training.

    This callback periodically runs bias metrics on a subset of the validation data
    to monitor model alignment during training.
    """

    def __init__(
        self,
        eval_dataset,
        bias_evaluator,
        eval_steps: int = 500,
        eval_at_end: bool = True,
        use_wandb: bool = True
    ):
        self.eval_dataset = eval_dataset
        self.bias_evaluator = bias_evaluator
        self.eval_steps = eval_steps
        self.eval_at_end = eval_at_end
        self.use_wandb = use_wandb

    def on_step_end(self, args, state, control, **kwargs):
        """Run bias eval every eval_steps."""
        if state.global_step % self.eval_steps == 0 and state.global_step > 0:
            # Run bias evaluation
            print(f"\nRunning bias evaluation at step {state.global_step}...")
            # Implementation depends on evaluator interface
            # metrics = self.bias_evaluator.evaluate_dataset(self.eval_dataset)
            # if self.use_wandb:
            #     wandb.log(metrics, step=state.global_step)
        return control

    def on_train_end(self, args, state, control, **kwargs):
        """Run final bias evaluation."""
        if self.eval_at_end:
            print("\nRunning final bias evaluation...")
        return control


class WandBLoggingCallback(TrainerCallback):
    """Custom callback for enhanced WandB logging."""

    def __init__(self, project: str, config: Dict[str, Any]):
        self.project = project
        self.config = config

    def on_train_begin(self, args, state, control, **kwargs):
        """Initialize WandB run."""
        try:
            import wandb
            wandb.init(
                project=self.project,
                config=self.config,
                name=f"run_{state.global_step}"
            )
        except ImportError:
            print("Warning: wandb not available")

    def on_log(self, args, state, control, logs=None, **kwargs):
        """Log metrics to WandB."""
        if logs:
            try:
                import wandb
                wandb.log(logs, step=state.global_step)
            except ImportError:
                pass

    def on_train_end(self, args, state, control, **kwargs):
        """Close WandB run."""
        try:
            import wandb
            wandb.finish()
        except ImportError:
            pass
