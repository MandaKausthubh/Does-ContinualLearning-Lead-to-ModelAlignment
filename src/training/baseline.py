"""
Baseline training (no fine-tuning, just evaluation on pretrained model).
"""

import os
from typing import Dict, Optional, Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

try:
    import torch_xla
    import torch_xla.core.xla_model as xm
    import torch_xla.distributed.xla_multiprocessing as xmp
    HAS_XLA = True
except ImportError:
    HAS_XLA = False

from ..models.model_loader import LlamaModel
from .trainer_utils import BiasEvaluationCallback, WandBLoggingCallback


class BaselineTrainer:
    """
    Trainer for baseline evaluation (no training, just inference).

    Used to establish baseline bias metrics before any fine-tuning.
    """

    def __init__(
        self,
        model: LlamaModel,
        config: Dict[str, Any],
        output_dir: str = "./experiments/exp_01_baseline"
    ):
        self.model = model
        self.config = config
        self.output_dir = output_dir
        self.device_type = config.get("device", {}).get("type", "cpu")

        os.makedirs(output_dir, exist_ok=True)

    def evaluate(
        self,
        eval_dataset,
        batch_size: int = 8,
        max_new_tokens: int = 256
    ) -> Dict[str, Any]:
        """
        Run baseline evaluation on a dataset.

        Args:
            eval_dataset: Dataset to evaluate on
            batch_size: Batch size for evaluation
            max_new_tokens: Maximum tokens to generate

        Returns:
            Dictionary with evaluation results
        """
        self.model.model.eval()

        results = []
        dataloader = DataLoader(
            eval_dataset,
            batch_size=batch_size,
            shuffle=False
        )

        print(f"Running baseline evaluation on {len(eval_dataset)} samples...")

        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Evaluating"):
                input_ids = batch["input_ids"]
                attention_mask = batch.get("attention_mask")

                if self.device_type == "tpu" and HAS_XLA:
                    input_ids = input_ids.to(xm.xla_device())
                    if attention_mask is not None:
                        attention_mask = attention_mask.to(xm.xla_device())
                elif self.device_type == "cuda":
                    input_ids = input_ids.cuda()
                    if attention_mask is not None:
                        attention_mask = attention_mask.cuda()

                # Generate outputs
                outputs = self.model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=max_new_tokens
                )

                # Decode outputs
                for i, output in enumerate(outputs):
                    generated_text = self.model.tokenizer.decode(
                        output[input_ids[i].shape[0]:],
                        skip_special_tokens=True
                    )
                    results.append({
                        "input": batch.get("text", [""])[i],
                        "output": generated_text
                    })

        return {
            "results": results,
            "num_samples": len(results)
        }

    def save_baseline_outputs(
        self,
        results: Dict[str, Any],
        output_file: str = "baseline_outputs.json"
    ):
        """Save baseline evaluation outputs."""
        import json

        output_path = os.path.join(self.output_dir, output_file)
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"Saved baseline outputs to {output_path}")
