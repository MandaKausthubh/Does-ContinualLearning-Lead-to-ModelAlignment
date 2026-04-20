#!/usr/bin/env python3
"""
Main entry point for LLM Continual Learning for Model Alignment experiments.

This script provides a unified interface to run all experiments:
- Baseline evaluation
- Standard Fine-Tuning (SFT)
- Self-Distillation Fine-Tuning (SDFT)
- Evaluation and comparison

Usage:
    # Run baseline evaluation
    python main.py baseline --model llama-1b --device tpu

    # Run SFT training
    python main.py sft --model llama-1b --device tpu

    # Run SDFT training
    python main.py sdft --model llama-1b --alpha 0.5 --device tpu

    # Evaluate a checkpoint
    python main.py evaluate --model-path ./experiments/exp_02_sft/final

    # Compare experiments
    python main.py compare --experiment-dirs ./experiments/exp_01_baseline ./experiments/exp_02_sft
"""

import sys
import argparse
from pathlib import Path

# Ensure src is in path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.pipelines.run_baseline import run_baseline
from src.pipelines.run_sft import run_sft
from src.pipelines.run_sdft import run_sdft
from src.pipelines.evaluate_all import evaluate_model, compare_experiments


def main():
    parser = argparse.ArgumentParser(
        description="LLM Continual Learning for Model Alignment",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Baseline evaluation
  python main.py baseline --model llama-1b --num-samples 500

  # Standard Fine-Tuning
  python main.py sft --model llama-1b --phase1-dataset dolly --phase2-dataset stereoset

  # Self-Distillation Fine-Tuning
  python main.py sdft --model llama-1b --alpha 0.5 --temperature 2.0

  # Evaluate a checkpoint
  python main.py evaluate --model-path ./experiments/exp_02_sft/phase2/final

  # Compare all experiments
  python main.py compare --experiments exp_01_baseline exp_02_sft exp_03_sdft
        """
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Baseline command
    baseline_parser = subparsers.add_parser(
        "baseline",
        help="Run baseline evaluation on pretrained model"
    )
    baseline_parser.add_argument("--model", type=str, default="llama-1b",
                                help="Model name (llama-1b, llama-3b, llama-8b)")
    baseline_parser.add_argument("--datasets", nargs="+", default=["dolly", "oasst1"],
                                help="Datasets to evaluate")
    baseline_parser.add_argument("--output-dir", type=str, default="./experiments/exp_01_baseline",
                                help="Output directory")
    baseline_parser.add_argument("--num-samples", type=int, default=500,
                                help="Number of samples for bias evaluation")
    baseline_parser.add_argument("--device", type=str, default="tpu",
                                choices=["tpu", "cuda", "cpu"],
                                help="Device type")
    baseline_parser.add_argument("--seed", type=int, default=42,
                                help="Random seed")

    # SFT command
    sft_parser = subparsers.add_parser(
        "sft",
        help="Run Standard Fine-Tuning"
    )
    sft_parser.add_argument("--model", type=str, default="llama-1b",
                           help="Base model name")
    sft_parser.add_argument("--phase1-dataset", type=str, default="dolly",
                           help="Phase 1 dataset (dolly, oasst1)")
    sft_parser.add_argument("--phase2-dataset", type=str, default="stereoset",
                           help="Phase 2 alignment dataset")
    sft_parser.add_argument("--output-dir", type=str, default="./experiments/exp_02_sft",
                           help="Output directory")
    sft_parser.add_argument("--config", type=str, default="./configs/training_config.yaml",
                           help="Training configuration file")
    sft_parser.add_argument("--skip-phase1", action="store_true",
                           help="Skip phase 1 training")
    sft_parser.add_argument("--skip-phase2", action="store_true",
                           help="Skip phase 2 training")
    sft_parser.add_argument("--device", type=str, default="tpu",
                           choices=["tpu", "cuda", "cpu"])
    sft_parser.add_argument("--seed", type=int, default=42)

    # SDFT command
    sdft_parser = subparsers.add_parser(
        "sdft",
        help="Run Self-Distillation Fine-Tuning"
    )
    sdft_parser.add_argument("--model", type=str, default="llama-1b",
                            help="Base model name")
    sdft_parser.add_argument("--phase1-dataset", type=str, default="dolly",
                            help="Phase 1 dataset")
    sdft_parser.add_argument("--phase2-dataset", type=str, default="stereoset",
                            help="Phase 2 alignment dataset")
    sdft_parser.add_argument("--output-dir", type=str, default="./experiments/exp_03_sdft",
                            help="Output directory")
    sdft_parser.add_argument("--config", type=str, default="./configs/training_config.yaml",
                            help="Training configuration file")
    sdft_parser.add_argument("--alpha", type=float, default=0.5,
                            help="Distillation weight (0-1)")
    sdft_parser.add_argument("--temperature", type=float, default=2.0,
                            help="Distillation temperature")
    sdft_parser.add_argument("--skip-phase1", action="store_true",
                            help="Skip phase 1 training")
    sdft_parser.add_argument("--skip-phase2", action="store_true",
                            help="Skip phase 2 training")
    sdft_parser.add_argument("--device", type=str, default="tpu",
                            choices=["tpu", "cuda", "cpu"])
    sdft_parser.add_argument("--seed", type=int, default=42)

    # Evaluate command
    eval_parser = subparsers.add_parser(
        "evaluate",
        help="Evaluate a model checkpoint"
    )
    eval_parser.add_argument("--model-path", type=str, required=True,
                            help="Path to model checkpoint")
    eval_parser.add_argument("--model-name", type=str, default="llama-1b",
                              help="Base model name (for config)")
    eval_parser.add_argument("--datasets", nargs="+", default=["dolly", "oasst1"],
                            help="Datasets to evaluate")
    eval_parser.add_argument("--output", type=str, required=True,
                            help="Output path for results")
    eval_parser.add_argument("--num-samples", type=int, default=500,
                            help="Number of samples to evaluate")
    eval_parser.add_argument("--device", type=str, default="tpu",
                            choices=["tpu", "cuda", "cpu"])

    # Compare command
    compare_parser = subparsers.add_parser(
        "compare",
        help="Compare multiple experiments"
    )
    compare_parser.add_argument("--experiments", nargs="+", required=True,
                                 help="Experiment directories to compare")
    compare_parser.add_argument("--output", type=str, default="./results/comparison.json",
                                help="Output path for comparison")

    # Full pipeline command
    full_parser = subparsers.add_parser(
        "full",
        help="Run full pipeline: baseline + SFT + SDFT + evaluation"
    )
    full_parser.add_argument("--model", type=str, default="llama-1b",
                            help="Model name")
    full_parser.add_argument("--device", type=str, default="tpu",
                            choices=["tpu", "cuda", "cpu"])
    full_parser.add_argument("--num-samples", type=int, default=500,
                            help="Number of samples for bias evaluation")
    full_parser.add_argument("--alpha", type=float, default=0.5,
                            help="SDFT distillation alpha")
    full_parser.add_argument("--temperature", type=float, default=2.0,
                            help="SDFT distillation temperature")
    full_parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return

    # Execute command
    if args.command == "baseline":
        run_baseline(
            model_name=args.model,
            eval_datasets=args.datasets,
            output_dir=args.output_dir,
            num_bias_samples=args.num_samples,
            device_type=args.device,
            seed=args.seed
        )

    elif args.command == "sft":
        run_sft(
            model_name=args.model,
            phase1_dataset=args.phase1_dataset,
            phase2_dataset=args.phase2_dataset,
            output_dir=args.output_dir,
            config_path=args.config,
            skip_phase1=args.skip_phase1,
            skip_phase2=args.skip_phase2,
            device_type=args.device,
            seed=args.seed
        )

    elif args.command == "sdft":
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

    elif args.command == "evaluate":
        evaluate_model(
            model_path=args.model_path,
            eval_datasets=args.datasets,
            output_path=args.output,
            num_samples=args.num_samples,
            device_type=args.device,
            model_name=args.model_name
        )

    elif args.command == "compare":
        compare_experiments(
            experiment_dirs=args.experiments,
            output_path=args.output
        )

    elif args.command == "full":
        print("="*80)
        print("RUNNING FULL PIPELINE")
        print("="*80)

        # 1. Baseline
        print("\n[1/4] Running baseline evaluation...")
        run_baseline(
            model_name=args.model,
            output_dir="./experiments/exp_01_baseline",
            num_bias_samples=args.num_samples,
            device_type=args.device,
            seed=args.seed
        )

        # 2. SFT
        print("\n[2/4] Running SFT training...")
        run_sft(
            model_name=args.model,
            output_dir="./experiments/exp_02_sft",
            device_type=args.device,
            seed=args.seed
        )

        # 3. SDFT
        print("\n[3/4] Running SDFT training...")
        run_sdft(
            model_name=args.model,
            output_dir="./experiments/exp_03_sdft",
            alpha=args.alpha,
            temperature=args.temperature,
            device_type=args.device,
            seed=args.seed
        )

        # 4. Compare
        print("\n[4/4] Comparing experiments...")
        compare_experiments(
            experiment_dirs=[
                "./experiments/exp_01_baseline",
                "./experiments/exp_02_sft",
                "./experiments/exp_03_sdft"
            ],
            output_path="./results/comparison.json"
        )

        print("\nFull pipeline complete!")


if __name__ == "__main__":
    main()
