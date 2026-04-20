"""
Pipeline for baseline evaluation (no fine-tuning).

This establishes baseline bias metrics for the pre-trained model
before any fine-tuning occurs.
"""

import os
import sys
import json
import argparse
from pathlib import Path

import torch

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.model_loader import ModelLoader, ModelConfig
from data.loader import DataLoader
from data.augmentation import GenderSwapper
from models.wrappers import InferenceWrapper
from evaluation.bias_metrics import BiasEvaluator
from evaluation.flip_rate import FlipRateCalculator
from utils.logger import setup_logger
from utils.seed import set_seed


def run_baseline(
    model_name: str = "llama-1b",
    eval_datasets: list = None,
    output_dir: str = "./experiments/exp_01_baseline",
    num_bias_samples: int = 500,
    config_path: str = "./configs",
    device_type: str = "tpu",
    seed: int = 42
):
    """
    Run baseline evaluation on pre-trained model.

    Args:
        model_name: Name of model to evaluate
        eval_datasets: List of datasets to evaluate on
        output_dir: Directory to save results
        num_bias_samples: Number of samples for bias evaluation
        config_path: Path to config files
        device_type: Device type (tpu, cuda, cpu)
        seed: Random seed
    """
    # Setup
    set_seed(seed)
    logger = setup_logger("baseline", output_dir)
    os.makedirs(output_dir, exist_ok=True)

    logger.info("="*60)
    logger.info("Starting Baseline Evaluation")
    logger.info(f"Model: {model_name}")
    logger.info(f"Device: {device_type}")
    logger.info("="*60)

    # Load model
    logger.info("Loading model...")
    config = ModelConfig(
        model_name=model_name,
        cache_dir="./cache/models",
        torch_dtype="bfloat16",
        device_type=device_type,
        use_lora=False
    )

    loader = ModelLoader()
    model = loader.load(config, for_training=False)
    logger.info(f"Loaded {model_name}")

    # Create inference wrapper
    wrapper = InferenceWrapper(
        model,
        batch_size=8,
        cache_results=False
    )

    # Initialize gender swapper
    swapper = GenderSwapper()

    # Initialize evaluators
    bias_evaluator = BiasEvaluator(
        device="cuda" if device_type == "cuda" else "cpu",
        batch_size=32
    )
    flip_calculator = FlipRateCalculator()

    # Results storage
    all_results = {}

    eval_datasets = eval_datasets or ["dolly", "oasst1"]

    for dataset_name in eval_datasets:
        logger.info(f"\nEvaluating on {dataset_name}...")

        # Load dataset
        data_loader = DataLoader()
        if "dolly" in dataset_name.lower():
            dataset = data_loader.load_dolly(
                model.tokenizer,
                max_samples=num_bias_samples * 2  # Extra for filtering
            )
        elif "oasst" in dataset_name.lower():
            dataset = data_loader.load_oasst1(
                model.tokenizer,
                max_samples=num_bias_samples * 2
            )
        else:
            logger.warning(f"Unknown dataset: {dataset_name}")
            continue

        # Filter for gendered content
        logger.info(f"Extracting gendered samples from {len(dataset)} total...")
        gendered_data = swapper.extract_gendered_samples(
            [dataset[i] for i in range(min(len(dataset), num_bias_samples * 2))],
            min_swaps=1
        )

        if len(gendered_data) > num_bias_samples:
            import random
            gendered_data = random.sample(gendered_data, num_bias_samples)

        logger.info(f"Found {len(gendered_data)} gendered samples")

        if len(gendered_data) == 0:
            logger.warning(f"No gendered samples found in {dataset_name}")
            continue

        # Generate outputs
        logger.info("Generating outputs for male and female variants...")
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

        # Generate
        gen_results = wrapper.generate_for_bias_testing(
            prompts_male[:num_bias_samples],
            prompts_female[:num_bias_samples],
            max_new_tokens=256
        )

        male_outputs = gen_results["male_outputs"]
        female_outputs = gen_results["female_outputs"]

        # Compute bias metrics
        logger.info("Computing bias metrics...")
        bias_metrics = bias_evaluator.evaluate_outputs(
            male_outputs,
            female_outputs
        )

        # Compute flip rate
        flip_result = flip_calculator.calculate_flip_rate(
            male_outputs,
            female_outputs,
            prompts_male[:num_bias_samples],
            prompts_female[:num_bias_samples]
        )

        # Store results
        dataset_results = {
            "bias_metrics": bias_metrics.to_dict(),
            "flip_rate": flip_result.flip_rate,
            "n_flips": flip_result.n_flips,
            "n_samples": flip_result.n_samples,
        }

        all_results[dataset_name] = dataset_results

        # Save individual results
        result_file = os.path.join(output_dir, f"{dataset_name}_results.json")
        with open(result_file, 'w') as f:
            json.dump(dataset_results, f, indent=2)

        logger.info(f"Results for {dataset_name}:")
        logger.info(f"  Sentiment Bias: {bias_metrics.sentiment_bias:.4f}")
        logger.info(f"  Toxicity Bias: {bias_metrics.toxicity_bias:.4f}")
        logger.info(f"  Flip Rate: {flip_result.flip_rate:.4f}")

    # Save combined results
    final_results = {
        "model_name": model_name,
        "evaluation_type": "baseline",
        "num_samples": num_bias_samples,
        "datasets": all_results
    }

    final_file = os.path.join(output_dir, "baseline_results.json")
    with open(final_file, 'w') as f:
        json.dump(final_results, f, indent=2)

    logger.info(f"\nBaseline evaluation complete. Results saved to {output_dir}")

    return final_results


def main():
    parser = argparse.ArgumentParser(description="Run baseline evaluation")
    parser.add_argument("--model", type=str, default="llama-1b",
                       help="Model name")
    parser.add_argument("--datasets", nargs="+", default=["dolly", "oasst1"],
                       help="Datasets to evaluate on")
    parser.add_argument("--output-dir", type=str, default="./experiments/exp_01_baseline",
                       help="Output directory")
    parser.add_argument("--num-samples", type=int, default=500,
                       help="Number of samples for bias evaluation")
    parser.add_argument("--device", type=str, default="tpu",
                       choices=["tpu", "cuda", "cpu"],
                       help="Device type")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")

    args = parser.parse_args()

    run_baseline(
        model_name=args.model,
        eval_datasets=args.datasets,
        output_dir=args.output_dir,
        num_bias_samples=args.num_samples,
        device_type=args.device,
        seed=args.seed
    )


if __name__ == "__main__":
    main()
