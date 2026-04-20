"""
Bias metrics calculation and statistical testing.

Implements the bias measurement framework from the project:
- Sentiment Bias (B_sent)
- Toxicity Bias (B_tox)
- Statistical significance testing
"""

import json
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass, asdict
from pathlib import Path

from scipy import stats
from scipy.stats import ttest_rel, wilcoxon

from .sentiment import SentimentAnalyzer
from .toxicity import ToxicityAnalyzer


@dataclass
class BiasMetrics:
    """Container for all bias metrics."""
    # Core bias metrics
    sentiment_bias: float
    toxicity_bias: float

    # Component scores
    sentiment_male_mean: float
    sentiment_female_mean: float
    sentiment_male_std: float
    sentiment_female_std: float

    toxicity_male_mean: float
    toxicity_female_mean: float
    toxicity_male_std: float
    toxicity_female_std: float

    # Statistical tests
    sentiment_p_value: float
    toxicity_p_value: float
    sentiment_effect_size: float
    toxicity_effect_size: float

    # Sample info
    n_samples: int

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=indent)


class BiasEvaluator:
    """
    Evaluator for computing bias metrics on model outputs.

    Uses gender-swapped prompt pairs to measure:
    - Sentiment bias: B_sent = E[S(male_output) - S(female_output)]
    - Toxicity bias: B_tox = E[T(male_output) - T(female_output)]
    """

    def __init__(
        self,
        sentiment_model: Optional[str] = None,
        toxicity_model: Optional[str] = None,
        device: str = "auto",
        batch_size: int = 32,
        cache_dir: Optional[str] = None
    ):
        """
        Initialize bias evaluator.

        Args:
            sentiment_model: Sentiment model name (None for default)
            toxicity_model: Toxicity model name (None for default)
            device: Device to run on
            batch_size: Batch size for evaluation
            cache_dir: Directory for caching evaluation results
        """
        self.batch_size = batch_size
        self.cache_dir = cache_dir
        if cache_dir:
            Path(cache_dir).mkdir(parents=True, exist_ok=True)

        # Initialize analyzers
        print("Initializing bias evaluators...")

        if sentiment_model:
            self.sentiment_analyzer = SentimentAnalyzer(
                model_name=sentiment_model,
                device=device,
                batch_size=batch_size
            )
        else:
            self.sentiment_analyzer = SentimentAnalyzer(
                device=device,
                batch_size=batch_size
            )

        if toxicity_model:
            self.toxicity_analyzer = ToxicityAnalyzer(
                model_name=toxicity_model,
                device=device,
                batch_size=batch_size
            )
        else:
            self.toxicity_analyzer = ToxicityAnalyzer(
                device=device,
                batch_size=batch_size
            )

    def evaluate_outputs(
        self,
        male_outputs: List[str],
        female_outputs: List[str],
        confidence_level: float = 0.95
    ) -> BiasMetrics:
        """
        Evaluate bias between male and female output pairs.

        Args:
            male_outputs: Model outputs for male-coded prompts
            female_outputs: Model outputs for female-coded prompts
            confidence_level: Confidence level for statistical tests

        Returns:
            BiasMetrics object with all computed metrics
        """
        assert len(male_outputs) == len(female_outputs), \
            "Output lists must have same length"

        n = len(male_outputs)

        # Get sentiment scores
        print("Computing sentiment scores...")
        male_sentiment = self.sentiment_analyzer.analyze_batch(
            male_outputs, return_scores_only=True
        )
        female_sentiment = self.sentiment_analyzer.analyze_batch(
            female_outputs, return_scores_only=True
        )

        # Get toxicity scores
        print("Computing toxicity scores...")
        male_toxicity = self.toxicity_analyzer.analyze_batch(
            male_outputs, return_scores_only=True
        )
        female_toxicity = self.toxicity_analyzer.analyze_batch(
            female_outputs, return_scores_only=True
        )

        # Compute bias metrics
        sentiment_diff = male_sentiment - female_sentiment
        toxicity_diff = male_toxicity - female_toxicity

        sentiment_bias = float(np.mean(sentiment_diff))
        toxicity_bias = float(np.mean(toxicity_diff))

        # Statistical significance tests
        # Paired t-test
        _, sentiment_p = ttest_rel(male_sentiment, female_sentiment)
        _, toxicity_p = ttest_rel(male_toxicity, female_toxicity)

        # Effect sizes (Cohen's d)
        sentiment_effect = self._cohens_d(male_sentiment, female_sentiment)
        toxicity_effect = self._cohens_d(male_toxicity, female_toxicity)

        return BiasMetrics(
            sentiment_bias=sentiment_bias,
            toxicity_bias=toxicity_bias,

            sentiment_male_mean=float(np.mean(male_sentiment)),
            sentiment_female_mean=float(np.mean(female_sentiment)),
            sentiment_male_std=float(np.std(male_sentiment)),
            sentiment_female_std=float(np.std(female_sentiment)),

            toxicity_male_mean=float(np.mean(male_toxicity)),
            toxicity_female_mean=float(np.mean(female_toxicity)),
            toxicity_male_std=float(np.std(male_toxicity)),
            toxicity_female_std=float(np.std(female_toxicity)),

            sentiment_p_value=float(sentiment_p),
            toxicity_p_value=float(toxicity_p),
            sentiment_effect_size=float(sentiment_effect),
            toxicity_effect_size=float(toxicity_effect),

            n_samples=n
        )

    def _cohens_d(self, x: np.ndarray, y: np.ndarray) -> float:
        """Compute Cohen's d effect size."""
        pooled_std = np.sqrt((np.var(x) + np.var(y)) / 2)
        if pooled_std == 0:
            return 0.0
        return (np.mean(x) - np.mean(y)) / pooled_std

    def evaluate_dataset(
        self,
        dataset,
        model_wrapper,
        gender_swapper,
        num_samples: Optional[int] = None,
        batch_size: int = 8
    ) -> Dict[str, BiasMetrics]:
        """
        Evaluate bias on a dataset by generating and comparing outputs.

        Args:
            dataset: Dataset to evaluate
            model_wrapper: Model wrapper for generation
            gender_swapper: GenderSwapper instance
            num_samples: Number of samples to evaluate (None for all)
            batch_size: Generation batch size

        Returns:
            Dictionary with bias metrics and generation results
        """
        from ..data.augmentation import SwapPair

        # Sample from dataset
        if num_samples and len(dataset) > num_samples:
            import random
            indices = random.sample(range(len(dataset)), num_samples)
            dataset = [dataset[i] for i in indices]

        print(f"Generating outputs for {len(dataset)} samples...")

        # Generate outputs
        male_outputs = []
        female_outputs = []
        prompts_male = []
        prompts_female = []

        for item in dataset:
            text = item.get("text", "")
            if not text:
                continue

            # Get gender-swapped versions
            swap_pair = gender_swapper.swap_gender(text)

            if swap_pair.swap_type == "male_to_female":
                male_prompt = text
                female_prompt = swap_pair.swapped
            else:
                male_prompt = swap_pair.swapped
                female_prompt = text

            prompts_male.append(male_prompt)
            prompts_female.append(female_prompt)

        # Generate outputs
        results = model_wrapper.generate_for_bias_testing(
            prompts_male,
            prompts_female,
            max_new_tokens=256
        )

        male_outputs = results["male_outputs"]
        female_outputs = results["female_outputs"]

        # Compute bias metrics
        metrics = self.evaluate_outputs(male_outputs, female_outputs)

        return {
            "bias_metrics": metrics,
            "male_outputs": male_outputs,
            "female_outputs": female_outputs,
            "male_prompts": prompts_male,
            "female_prompts": prompts_female
        }

    def save_results(
        self,
        metrics: BiasMetrics,
        output_path: str
    ):
        """Save metrics to JSON file."""
        with open(output_path, 'w') as f:
            f.write(metrics.to_json())
        print(f"Saved bias metrics to {output_path}")


def compute_bias_statistics(
    scores_male: np.ndarray,
    scores_female: np.ndarray,
    confidence_level: float = 0.95
) -> Dict[str, float]:
    """
    Compute bias statistics with confidence intervals.

    Args:
        scores_male: Scores for male-coded outputs
        scores_female: Scores for female-coded outputs
        confidence_level: Confidence level for intervals

    Returns:
        Dictionary with bias estimate, CI, and statistical tests
    """
    differences = scores_male - scores_female
    n = len(differences)

    # Point estimate
    bias_estimate = float(np.mean(differences))

    # Confidence interval
    alpha = 1 - confidence_level
    t_value = stats.t.ppf(1 - alpha/2, df=n-1)
    se = np.std(differences, ddof=1) / np.sqrt(n)
    ci_lower = bias_estimate - t_value * se
    ci_upper = bias_estimate + t_value * se

    # Statistical tests
    t_stat, p_value = ttest_rel(scores_male, scores_female)

    # Effect size
    cohens_d = float(np.mean(differences) / np.std(differences, ddof=1))

    return {
        "bias_estimate": bias_estimate,
        "confidence_interval_lower": float(ci_lower),
        "confidence_interval_upper": float(ci_upper),
        "confidence_level": confidence_level,
        "t_statistic": float(t_stat),
        "p_value": float(p_value),
        "cohens_d": cohens_d,
        "n_samples": n
    }
