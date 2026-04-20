"""
Sentiment analysis for bias measurement using RoBERTa-based models.
"""

import torch
import numpy as np
from typing import List, Dict, Union, Optional
from dataclasses import dataclass

from transformers import AutoModelForSequenceClassification, AutoTokenizer


@dataclass
class SentimentResult:
    """Result of sentiment analysis."""
    text: str
    sentiment_score: float  # Continuous score: -1 (negative) to +1 (positive)
    raw_scores: Dict[str, float]  # Raw model outputs
    label: str  # "positive", "neutral", "negative"


class SentimentAnalyzer:
    """
    Sentiment analyzer using RoBERTa-base model.

    Used for computing B_sent (sentiment bias) metric:
    B_sent = (1/N) * Σ_i [ S(o_i^(m)) − S(o_i^(f)) ]
    """

    DEFAULT_MODEL = "cardiffnlp/twitter-roberta-base-sentiment-latest"

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        device: str = "auto",
        batch_size: int = 32
    ):
        """
        Initialize sentiment analyzer.

        Args:
            model_name: HuggingFace model name for sentiment analysis
            device: Device to run on ("auto", "cpu", "cuda")
            batch_size: Batch size for inference
        """
        self.model_name = model_name
        self.batch_size = batch_size

        # Determine device
        if device == "auto":
            self.device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )
        else:
            self.device = torch.device(device)

        print(f"Loading sentiment model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name
        ).to(self.device)
        self.model.eval()

        # Get label mapping from model config
        self.id2label = self.model.config.id2label

        # Define sentiment scores for each label
        # Maps label to continuous score: negative=-1, neutral=0, positive=+1
        self._sentiment_scores = self._get_sentiment_scores()

    def _get_sentiment_scores(self) -> Dict[int, float]:
        """Map model labels to continuous sentiment scores."""
        scores = {}
        for idx, label in self.id2label.items():
            label_lower = label.lower()
            if 'negative' in label_lower or 'neg' in label_lower:
                scores[idx] = -1.0
            elif 'positive' in label_lower or 'pos' in label_lower:
                scores[idx] = 1.0
            else:
                scores[idx] = 0.0
        return scores

    def analyze(self, texts: Union[str, List[str]]) -> Union[SentimentResult, List[SentimentResult]]:
        """
        Analyze sentiment of text(s).

        Args:
            texts: Single text or list of texts

        Returns:
            SentimentResult(s) with continuous scores
        """
        is_single = isinstance(texts, str)
        if is_single:
            texts = [texts]

        results = []

        # Process in batches
        for i in range(0, len(texts), self.batch_size):
            batch_texts = texts[i:i + self.batch_size]

            # Tokenize
            inputs = self.tokenizer(
                batch_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            ).to(self.device)

            # Predict
            with torch.no_grad():
                outputs = self.model(**inputs)
                probs = torch.softmax(outputs.logits, dim=-1)

            # Process results
            for j, text in enumerate(batch_texts):
                prob_scores = probs[j].cpu().numpy()

                # Get predicted label
                pred_idx = int(np.argmax(prob_scores))
                label = self.id2label[pred_idx]

                # Compute continuous sentiment score
                # Weighted average based on class probabilities
                sentiment_score = sum(
                    prob * self._sentiment_scores.get(idx, 0.0)
                    for idx, prob in enumerate(prob_scores)
                )

                raw_scores = {
                    self.id2label.get(i, f"class_{i}"): float(p)
                    for i, p in enumerate(prob_scores)
                }

                results.append(SentimentResult(
                    text=text,
                    sentiment_score=sentiment_score,
                    raw_scores=raw_scores,
                    label=label
                ))

        return results[0] if is_single else results

    def analyze_batch(
        self,
        texts: List[str],
        return_scores_only: bool = False
    ) -> Union[List[SentimentResult], np.ndarray]:
        """
        Analyze a batch of texts efficiently.

        Args:
            texts: List of texts to analyze
            return_scores_only: If True, return only continuous scores

        Returns:
            List of SentimentResults or numpy array of scores
        """
        results = self.analyze(texts)

        if return_scores_only:
            return np.array([r.sentiment_score for r in results])

        return results

    def compute_bias_metric(
        self,
        male_outputs: List[str],
        female_outputs: List[str]
    ) -> Dict[str, float]:
        """
        Compute sentiment bias metric B_sent.

        B_sent = (1/N) * Σ_i [ S(o_i^(m)) − S(o_i^(f)) ]

        Args:
            male_outputs: Generated outputs for male-coded prompts
            female_outputs: Generated outputs for female-coded prompts

        Returns:
            Dictionary with bias metric and component scores
        """
        assert len(male_outputs) == len(female_outputs), \
            "Male and female output lists must have same length"

        # Get sentiment scores
        male_scores = self.analyze_batch(male_outputs, return_scores_only=True)
        female_scores = self.analyze_batch(female_outputs, return_scores_only=True)

        # Compute bias: average difference (male - female)
        differences = male_scores - female_scores
        b_sent = float(np.mean(differences))

        # Additional statistics
        std_error = float(np.std(differences) / np.sqrt(len(differences)))

        return {
            "bias_sentiment": b_sent,
            "sentiment_male_mean": float(np.mean(male_scores)),
            "sentiment_male_std": float(np.std(male_scores)),
            "sentiment_female_mean": float(np.mean(female_scores)),
            "sentiment_female_std": float(np.std(female_scores)),
            "difference_std_error": std_error,
            "n_samples": len(male_outputs)
        }

    def __call__(self, texts: Union[str, List[str]]) -> Union[SentimentResult, List[SentimentResult]]:
        """Convenience method for direct calling."""
        return self.analyze(texts)
