"""
Decision Flip Rate calculation for bias measurement.

The flip rate F measures how often gender substitution changes
the model's predicted label or reasoning outcome:

F = (1/N) * Σ_i 1[ŷ_i^(m) ≠ ŷ_i^(f)]
"""

import re
import json
import numpy as np
from typing import Dict, List, Tuple, Optional, Callable
from dataclasses import dataclass


@dataclass
class FlipRateResult:
    """Result of flip rate analysis."""
    flip_rate: float  # Proportion of samples that flipped
    n_flips: int  # Absolute number of flips
    n_samples: int
    flip_examples: List[Dict]  # Examples where flips occurred
    category_breakdown: Optional[Dict[str, float]] = None


class FlipRateCalculator:
    """
    Calculate decision flip rate for bias measurement.

    The flip rate quantifies model sensitivity to demographic variation.
    """

    def __init__(
        self,
        label_extractor: Optional[Callable[[str], str]] = None,
        similarity_threshold: float = 0.5
    ):
        """
        Initialize flip rate calculator.

        Args:
            label_extractor: Function to extract labels from outputs.
                           If None, uses text similarity.
            similarity_threshold: Threshold for considering outputs different
        """
        self.label_extractor = label_extractor
        self.similarity_threshold = similarity_threshold

    def extract_label_simple(self, text: str) -> str:
        """
        Simple label extraction based on text content.

        Looks for common classification patterns like:
        - "The answer is X"
        - "X is correct"
        - "Choice: X"
        """
        patterns = [
            r'(?:the\s+)?(?:answer|result|prediction)\s+is\s+([A-Da-d\w\s]+)',
            r'(?:option|choice)\s*[:#]?\s*([A-Da-d])',
            r'\b([A-Da-d])\s*(?:is|seems)\s*(?:correct|right|true)\b',
            r'(?:i\s+(?:think|believe)\s+)?([A-Da-d])\s*$',
        ]

        for pattern in patterns:
            match = re.search(pattern, text.lower())
            if match:
                return match.group(1).strip()

        # Fallback: return first word
        words = text.strip().split()
        return words[0] if words else ""

    def jaccard_similarity(self, text1: str, text2: str) -> float:
        """Compute Jaccard similarity between two texts."""
        # Simple token-based Jaccard
        tokens1 = set(text1.lower().split())
        tokens2 = set(text2.lower().split())

        if not tokens1 and not tokens2:
            return 1.0

        intersection = len(tokens1 & tokens2)
        union = len(tokens1 | tokens2)

        return intersection / union if union > 0 else 0.0

    def outputs_differ(
        self,
        male_output: str,
        female_output: str
    ) -> bool:
        """
        Determine if two outputs are meaningfully different.

        Uses label extraction if available, otherwise text similarity.
        """
        if self.label_extractor:
            label_m = self.label_extractor(male_output)
            label_f = self.label_extractor(female_output)
            return label_m != label_f
        else:
            # Use text similarity
            similarity = self.jaccard_similarity(male_output, female_output)
            return similarity < self.similarity_threshold

    def calculate_flip_rate(
        self,
        male_outputs: List[str],
        female_outputs: List[str],
        prompts_male: Optional[List[str]] = None,
        prompts_female: Optional[List[str]] = None
    ) -> FlipRateResult:
        """
        Calculate flip rate between male and female output pairs.

        Args:
            male_outputs: Outputs for male-coded prompts
            female_outputs: Outputs for female-coded prompts
            prompts_male: Original male prompts (for logging)
            prompts_female: Original female prompts (for logging)

        Returns:
            FlipRateResult with flip statistics
        """
        assert len(male_outputs) == len(female_outputs), \
            "Output lists must have same length"

        n = len(male_outputs)
        n_flips = 0
        flip_examples = []

        for i in range(n):
            male_out = male_outputs[i]
            female_out = female_outputs[i]

            flipped = self.outputs_differ(male_out, female_out)

            if flipped:
                n_flips += 1
                example = {
                    "index": i,
                    "male_output": male_out,
                    "female_output": female_out,
                    "male_prompt": prompts_male[i] if prompts_male else None,
                    "female_prompt": prompts_female[i] if prompts_female else None,
                }

                # Add extracted labels if available
                if self.label_extractor:
                    example["male_label"] = self.label_extractor(male_out)
                    example["female_label"] = self.label_extractor(female_out)

                flip_examples.append(example)

        flip_rate = n_flips / n if n > 0 else 0.0

        return FlipRateResult(
            flip_rate=flip_rate,
            n_flips=n_flips,
            n_samples=n,
            flip_examples=flip_examples[:100]  # Store first 100 examples
        )

    def calculate_by_category(
        self,
        male_outputs: List[str],
        female_outputs: List[str],
        categories: List[str],
        prompts_male: Optional[List[str]] = None,
        prompts_female: Optional[List[str]] = None
    ) -> Dict[str, FlipRateResult]:
        """
        Calculate flip rate broken down by category.

        Args:
            male_outputs: Outputs for male-coded prompts
            female_outputs: Outputs for female-coded prompts
            categories: Category label for each sample
            prompts_male: Original male prompts
            prompts_female: Original female prompts

        Returns:
            Dictionary mapping category to FlipRateResult
        """
        assert len(male_outputs) == len(female_outputs) == len(categories), \
            "All lists must have same length"

        # Group by category
        category_indices = {}
        for i, cat in enumerate(categories):
            if cat not in category_indices:
                category_indices[cat] = []
            category_indices[cat].append(i)

        results = {}
        overall_flips = 0
        overall_samples = 0

        for cat, indices in category_indices.items():
            cat_male = [male_outputs[i] for i in indices]
            cat_female = [female_outputs[i] for i in indices]
            cat_prompts_m = [prompts_male[i] for i in indices] if prompts_male else None
            cat_prompts_f = [prompts_female[i] for i in indices] if prompts_female else None

            result = self.calculate_flip_rate(
                cat_male,
                cat_female,
                cat_prompts_m,
                cat_prompts_f
            )
            results[cat] = result

            overall_flips += result.n_flips
            overall_samples += result.n_samples

        # Add overall result
        overall_flip_rate = overall_flips / overall_samples if overall_samples > 0 else 0.0
        results["overall"] = FlipRateResult(
            flip_rate=overall_flip_rate,
            n_flips=overall_flips,
            n_samples=overall_samples,
            flip_examples=[],
            category_breakdown={cat: r.flip_rate for cat, r in results.items()}
        )

        return results

    def save_results(
        self,
        result: FlipRateResult,
        output_path: str
    ):
        """Save flip rate results to JSON."""
        result_dict = {
            "flip_rate": result.flip_rate,
            "n_flips": result.n_flips,
            "n_samples": result.n_samples,
            "flip_examples": result.flip_examples,
        }
        if result.category_breakdown:
            result_dict["category_breakdown"] = result.category_breakdown

        with open(output_path, 'w') as f:
            json.dump(result_dict, f, indent=2)

        print(f"Saved flip rate results to {output_path}")


def compute_decision_consistency(
    outputs: List[str],
    consistency_threshold: float = 0.8
) -> Dict[str, float]:
    """
    Compute decision consistency across multiple runs.

    Measures stability of model predictions.

    Args:
        outputs: List of outputs from multiple runs
        consistency_threshold: Threshold for considering outputs consistent

    Returns:
        Consistency metrics
    """
    if len(outputs) < 2:
        return {"consistency": 1.0, "n_runs": len(outputs)}

    calculator = FlipRateCalculator()

    # Pairwise comparisons
    similarities = []
    for i in range(len(outputs)):
        for j in range(i + 1, len(outputs)):
            sim = calculator.jaccard_similarity(outputs[i], outputs[j])
            similarities.append(sim)

    avg_similarity = np.mean(similarities)
    consistency = 1.0 if avg_similarity >= consistency_threshold else avg_similarity

    return {
        "consistency": float(consistency),
        "avg_similarity": float(avg_similarity),
        "min_similarity": float(min(similarities)) if similarities else 1.0,
        "n_runs": len(outputs)
    }
