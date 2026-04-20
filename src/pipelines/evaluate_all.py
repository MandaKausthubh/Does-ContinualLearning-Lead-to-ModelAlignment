"""
Comprehensive evaluation pipeline for all experiments.

Evaluates models at different stages and compares results across experiments.
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, List, Optional
import glob

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.model_loader import ModelLoader, ModelConfig
from data.loader import DataLoader
from data.augmentation import GenderSwapper
from models.wrappers import InferenceWrapper
from evaluation.bias_metrics import BiasEvaluator
from evaluation.flip_rate import FlipRateCalculator
from utils.logger import setup_logger
from utils.seed import set_seed


def evaluate_model(
    model_path: str,
    eval_datasets: List[str] = None,
    output_path: str = None,
    num_samples: int = 500,
    device_type: str = "tpu",
    model_name: str = None
):
    """
    Evaluate a single model checkpoint.

    Args:
        model_path: Path to model checkpoint
        eval_datasets: List of datasets to evaluate on
        output_path: Path to save results
        num_samples: Number of samples to evaluate
        device_type: Device type
        model_name: Base model name (for loading config)

    Returns:
        Dictionary with evaluation results
    """
    logger = setup_logger("eval")

    eval_datasets = eval_datasets or ["dolly", "oasst1"]

    logger.info(f"Evaluating model: {model_path}")

    # Load model
    config = ModelConfig(
        model_name=model_name or "llama-1b",
        cache_dir="./cache/models",
        torch_dtype="bfloat16",
        device_type=device_type,
        use_lora=False
    )

    loader = ModelLoader()
    model = loader.load_from_checkpoint(model_path, config, for_training=False)

    # Create inference wrapper
    wrapper = InferenceWrapper(model, batch_size=8, cache_results=False)

    # Initialize evaluators
    swapper = GenderSwapper()
    bias_evaluator = BiasEvaluator(device="cuda" if device_type == "cuda" else "cpu")
    flip_calculator = FlipRateCalculator()

    # Results storage
    all_results = {}

    for dataset_name in eval_datasets:
        logger.info(f"\nEvaluating on {dataset_name}...")

        # Load dataset
        data_loader = DataLoader()
        if "dolly" in dataset_name.lower():
            dataset = data_loader.load_dolly(model.tokenizer, max_samples=num_samples * 2)
        elif "oasst" in dataset_name.lower():
            dataset = data_loader.load_oasst1(model.tokenizer, max_samples=num_samples * 2)
        else:
            continue

        # Filter for gendered content
        gendered_data = swapper.extract_gendered_samples(
            [dataset[i] for i in range(min(len(dataset), num_samples * 2))],
            min_swaps=1
        )

        if len(gendered_data) > num_samples:
            import random
            gendered_data = random.sample(gendered_data, num_samples)

        # Generate outputs
        prompts_male = []
        prompts_female = []

        for item in gendered_data:
            text = item.get("text", "")
            swap_pair = swapper.swap_gender(text)

            if swap_pair.swap_type == "male_to_female":
                prompts_male.append(text)
                prompts_female.append(swap_pair.swapped)
            else:
                prompts_male.append(swap_pair.swapped)
                prompts_female.append(text)

        gen_results = wrapper.generate_for_bias_testing(
            prompts_male, prompts_female, max_new_tokens=256
        )

        # Compute metrics
        bias_metrics = bias_evaluator.evaluate_outputs(
            gen_results["male_outputs"],
            gen_results["female_outputs"]
        )

        flip_result = flip_calculator.calculate_flip_rate(
            gen_results["male_outputs"],
            gen_results["female_outputs"],
            prompts_male, prompts_female
        )

        all_results[dataset_name] = {
            "bias_metrics": bias_metrics.to_dict(),
            "flip_rate": flip_result.flip_rate,
            "n_flips": flip_result.n_flips,
            "n_samples": flip_result.n_samples
        }

    # Save results
    if output_path:
        with open(output_path, 'w') as f:
            json.dump(all_results, f, indent=2)
        logger.info(f"Saved results to {output_path}")

    return all_results


def compare_experiments(
    experiment_dirs: List[str],
    output_path: str = "./results/comparison.json"
):
    """
    Compare results across multiple experiments.

    Args:
        experiment_dirs: List of experiment directories to compare
        output_path: Path to save comparison

    Returns:
        Dictionary with comparison results
    """
    logger = setup_logger("compare")

    all_experiments = {}

    for exp_dir in experiment_dirs:
        exp_name = os.path.basename(exp_dir)

        # Look for result files
        result_files = {
            "baseline": os.path.join(exp_dir, "baseline_results.json"),
            "phase1": os.path.join(exp_dir, "phase1", "final", "eval_results.json"),
            "phase2": os.path.join(exp_dir, "phase2", "final", "eval_results.json"),
        }

        exp_results = {}
        for stage, result_file in result_files.items():
            if os.path.exists(result_file):
                with open(result_file, 'r') as f:
                    exp_results[stage] = json.load(f)

        if exp_results:
            all_experiments[exp_name] = exp_results

    # Compute comparison metrics
    comparison = {
        "experiments": all_experiments,
        "summary": {}
    }

    # Create summary table
    for exp_name, exp_data in all_experiments.items():
        comparison["summary"][exp_name] = {}

        for stage, stage_data in exp_data.items():
            if "datasets" in stage_data:
                for dataset, metrics in stage_data["datasets"].items():
                    if dataset not in comparison["summary"][exp_name]:
                        comparison["summary"][exp_name][dataset] = {}

                    bias = metrics.get("bias_metrics", {})
                    comparison["summary"][exp_name][dataset][stage] = {
                        "sentiment_bias": bias.get("sentiment_bias", 0),
                        "toxicity_bias": bias.get("toxicity_bias", 0),
                        "flip_rate": metrics.get("flip_rate", 0),
                    }

    # Save comparison
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(comparison, f, indent=2)

    logger.info(f"Saved comparison to {output_path}")

    # Print summary
    print("\n" + "="*80)
    print("EXPERIMENT COMPARISON SUMMARY")
    print("="*80)

    for exp_name, exp_summary in comparison["summary"].items():
        print(f"\n{exp_name}:")
        for dataset, stages in exp_summary.items():
            print(f"  {dataset}:")
            for stage, metrics in stages.items():
                print(f"    {stage}:")
                print(f"      Sentiment Bias: {metrics['sentiment_bias']:.4f}")
                print(f"      Toxicity Bias: {metrics['toxicity_bias']:.4f}")
                print(f"      Flip Rate: {metrics['flip_rate']:.4f}")

    return comparison


def main():
    parser = argparse.ArgumentParser(description="Evaluate models")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Evaluate command
    eval_parser = subparsers.add_parser("evaluate", help="Evaluate a model")
    eval_parser.add_argument("--model-path", type=str, required=True,
                            help="Path to model checkpoint")
    eval_parser.add_argument("--model-name", type=str, default="llama-1b",
                            help="Base model name")
    eval_parser.add_argument("--datasets", nargs="+", default=["dolly", "oasst1"],
                            help="Datasets to evaluate")
    eval_parser.add_argument("--output", type=str, required=True,
                            help="Output path for results")
    eval_parser.add_argument("--num-samples", type=int, default=500,
                            help="Number of samples")
    eval_parser.add_argument("--device", type=str, default="tpu",
                            choices=["tpu", "cuda", "cpu"])

    # Compare command
    compare_parser = subparsers.add_parser("compare", help="Compare experiments")
    compare_parser.add_argument("--experiment-dirs", nargs="+", required=True,
                               help="Experiment directories to compare")
    compare_parser.add_argument("--output", type=str, default="./results/comparison.json",
                               help="Output path")

    args = parser.parse_args()

    if args.command == "evaluate":
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
            experiment_dirs=args.experiment_dirs,
            output_path=args.output
        )
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
