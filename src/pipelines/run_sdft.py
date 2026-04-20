"""
Pipeline for Self-Distillation Fine-Tuning (SDFT).

Implements self-distillation regularization for continual learning,
using a frozen reference model to preserve prior capabilities.
"""

import os
import sys
import json
import argparse
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.model_loader import ModelLoader, ModelConfig
from data.loader import DataLoader
from training.sdft import SDFTTrainer
from utils.logger import setup_logger
from utils.seed import set_seed
import yaml


def run_sdft(
    model_name: str = "llama-1b",
    phase1_dataset: str = "dolly",
    phase2_dataset: str = "stereoset",
    output_dir: str = "./experiments/exp_03_sdft",
    config_path: str = "./configs/training_config.yaml",
    alpha: float = 0.5,
    temperature: float = 2.0,
    skip_phase1: bool = False,
    skip_phase2: bool = False,
    device_type: str = "tpu",
    seed: int = 42
):
    """
    Run Self-Distillation Fine-Tuning.

    Phase 1: Fine-tune on general instruction dataset
    Phase 2: Fine-tune on alignment dataset with SDFT regularization

    Args:
        model_name: Base model to fine-tune
        phase1_dataset: General dataset for initial fine-tuning
        phase2_dataset: Alignment dataset for debiasing
        output_dir: Directory to save results
        config_path: Path to training config
        alpha: Distillation weight
        temperature: Distillation temperature
        skip_phase1: Skip phase 1 training
        skip_phase2: Skip phase 2 training
        device_type: Device type
        seed: Random seed
    """
    set_seed(seed)
    logger = setup_logger("sdft", output_dir)
    os.makedirs(output_dir, exist_ok=True)

    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Update config with SDFT parameters
    config['device'] = {'type': device_type}
    config['training']['phase1']['dataset'] = phase1_dataset
    config['training']['phase2']['dataset'] = phase2_dataset
    config['training']['sdft']['alpha'] = alpha
    config['training']['sdft']['temperature'] = temperature

    logger.info("="*60)
    logger.info("Starting Self-Distillation Fine-Tuning (SDFT)")
    logger.info(f"Model: {model_name}")
    logger.info(f"Phase 1: {phase1_dataset}")
    logger.info(f"Phase 2: {phase2_dataset}")
    logger.info(f"Distillation Alpha: {alpha}")
    logger.info(f"Distillation Temperature: {temperature}")
    logger.info(f"Device: {device_type}")
    logger.info("="*60)

    # Initialize model
    logger.info("Loading base model...")
    model_config = ModelConfig(
        model_name=model_name,
        cache_dir="./cache/models",
        torch_dtype=config.get('optimization', {}).get('torch_dtype', 'bfloat16'),
        device_type=device_type,
        use_lora=config.get('lora', {}).get('enabled', True),
        lora_r=config.get('lora', {}).get('r', 16),
        lora_alpha=config.get('lora', {}).get('alpha', 32),
        lora_dropout=config.get('lora', {}).get('dropout', 0.05)
    )

    loader = ModelLoader()
    model = loader.load(model_config, for_training=True)
    logger.info(f"Loaded {model_name}")

    # Data loader
    data_loader = DataLoader()

    # Phase 1: General fine-tuning (standard, no distillation)
    if not skip_phase1:
        logger.info("\n" + "="*60)
        logger.info("PHASE 1: General Fine-Tuning")
        logger.info("="*60)

        phase1_config = config['training']['phase1']
        phase1_dir = os.path.join(output_dir, "phase1")
        os.makedirs(phase1_dir, exist_ok=True)

        # Load dataset
        logger.info(f"Loading {phase1_dataset} dataset...")
        if "dolly" in phase1_dataset.lower():
            train_dataset = data_loader.load_dolly(
                model.tokenizer,
                max_seq_length=phase1_config['max_seq_length']
            )
        elif "oasst" in phase1_dataset.lower():
            train_dataset = data_loader.load_oasst1(
                model.tokenizer,
                max_seq_length=phase1_config['max_seq_length']
            )
        else:
            raise ValueError(f"Unknown dataset: {phase1_dataset}")

        logger.info(f"Loaded {len(train_dataset)} training samples")

        # Train with standard SFT (no distillation in phase 1)
        from training.sft import SFTTrainer
        trainer = SFTTrainer(
            model=model,
            config=config,
            output_dir=phase1_dir,
            use_wandb=True
        )

        phase1_results = trainer.train(
            train_dataset=train_dataset,
            eval_dataset=None
        )

        # Save phase 1 model
        model.save(os.path.join(phase1_dir, "final"))
        logger.info(f"Phase 1 complete. Model saved to {phase1_dir}/final")

    # Phase 2: Alignment fine-tuning with SDFT
    if not skip_phase2:
        logger.info("\n" + "="*60)
        logger.info("PHASE 2: Alignment Fine-Tuning with SDFT")
        logger.info("="*60)

        phase2_config = config['training']['phase2']
        phase2_dir = os.path.join(output_dir, "phase2")
        os.makedirs(phase2_dir, exist_ok=True)

        # Load alignment dataset
        logger.info(f"Loading {phase2_dataset} dataset...")
        train_dataset = data_loader.load_stereoset(
            model.tokenizer,
            max_seq_length=phase2_config['max_seq_length'],
            for_alignment=True
        )

        logger.info(f"Loaded {len(train_dataset)} training samples")

        # Train with SDFT
        trainer = SDFTTrainer(
            model=model,
            config=config,
            output_dir=phase2_dir,
            use_wandb=True
        )

        phase2_results = trainer.train(
            train_dataset=train_dataset,
            eval_dataset=None
        )

        # Save phase 2 model
        model.save(os.path.join(phase2_dir, "final"))
        logger.info(f"Phase 2 complete. Model saved to {phase2_dir}/final")

    # Save final config
    with open(os.path.join(output_dir, "config.json"), 'w') as f:
        json.dump(config, f, indent=2)

    logger.info(f"\nSDFT training complete. Results saved to {output_dir}")

    return {"output_dir": output_dir, "config": config}


def main():
    parser = argparse.ArgumentParser(description="Run SDFT training")
    parser.add_argument("--model", type=str, default="llama-1b",
                       help="Base model name")
    parser.add_argument("--phase1-dataset", type=str, default="dolly",
                       help="Phase 1 dataset")
    parser.add_argument("--phase2-dataset", type=str, default="stereoset",
                       help="Phase 2 dataset")
    parser.add_argument("--output-dir", type=str, default="./experiments/exp_03_sdft",
                       help="Output directory")
    parser.add_argument("--config", type=str, default="./configs/training_config.yaml",
                       help="Training config file")
    parser.add_argument("--alpha", type=float, default=0.5,
                       help="Distillation weight (0-1)")
    parser.add_argument("--temperature", type=float, default=2.0,
                       help="Distillation temperature")
    parser.add_argument("--skip-phase1", action="store_true",
                       help="Skip phase 1 training")
    parser.add_argument("--skip-phase2", action="store_true",
                       help="Skip phase 2 training")
    parser.add_argument("--device", type=str, default="tpu",
                       choices=["tpu", "cuda", "cpu"])
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    run_sdft(
        model_name=args.model,
        phase1_dataset=args.phase1_dataset,
        phase2_dataset=args.phase2_dataset,
        output_dir=args.output_dir,
        config_path=args.config,
        alpha=args.alpha,
        temperature=args.temperature,
        skip_phase1=args.skip_phase1,
        skip_phase2=args.skip_phase2,
        device_type=args.device,
        seed=args.seed
    )


if __name__ == "__main__":
    main()
