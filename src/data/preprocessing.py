"""
Data preprocessing utilities for text cleaning and formatting.
"""

import re
import random
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass


@dataclass
class TextExample:
    """A single text example with metadata."""
    text: str
    label: Optional[str] = None
    metadata: Optional[Dict] = None


class Preprocessor:
    """Text preprocessing for instruction datasets."""

    def __init__(self, max_length: int = 512):
        self.max_length = max_length

    def clean_text(self, text: str) -> str:
        """Basic text cleaning."""
        if not text:
            return ""

        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)

        # Remove special characters but keep punctuation
        text = re.sub(r'[^\w\s\.\,\;\:\-\?\!\'\"\(\)]', ' ', text)

        # Trim
        text = text.strip()

        return text

    def format_instruction(self, instruction: str, input_text: str = "", response: str = "") -> str:
        """Format instruction-following example."""
        if input_text:
            prompt = f"### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:\n{response}"
        else:
            prompt = f"### Instruction:\n{instruction}\n\n### Response:\n{response}"
        return prompt

    def truncate(self, text: str, max_tokens: Optional[int] = None) -> str:
        """Rough truncation by word count (approximate token count)."""
        if max_tokens is None:
            max_tokens = self.max_length

        # Rough estimate: 0.75 tokens per word
        max_words = int(max_tokens * 0.75)
        words = text.split()

        if len(words) <= max_words:
            return text

        return ' '.join(words[:max_words])

    def batch_preprocess(self, examples: List[Dict]) -> List[Dict]:
        """Preprocess a batch of examples."""
        processed = []
        for ex in examples:
            processed_ex = {
                "instruction": self.clean_text(ex.get("instruction", "")),
                "input": self.clean_text(ex.get("input", "")),
                "response": self.clean_text(ex.get("response", "")),
                "context": self.clean_text(ex.get("context", "")),
            }
            # Create formatted text
            processed_ex["text"] = self.format_instruction(
                processed_ex["instruction"],
                processed_ex["input"] or processed_ex["context"],
                processed_ex["response"]
            )
            processed_ex["text"] = self.truncate(processed_ex["text"])
            processed.append(processed_ex)
        return processed


class StereoSetProcessor:
    """Process StereoSet format data."""

    def __init__(self):
        self.bias_categories = {
            "gender",
            "profession",
            "race",
            "religion"
        }

    def extract_sentence_pairs(self, data: Dict) -> List[Dict]:
        """Extract stereotypical/anti-stereotypical sentence pairs."""
        pairs = []

        for item in data.get("data", {}).get("intrasentence", []):
            target = item.get("target", "")
            bias_type = item.get("bias_type", "")

            for sentence in item.get("sentences", []):
                text = sentence.get("sentence", "")
                gold_label = sentence.get("gold_label", "")

                pair = {
                    "target": target,
                    "bias_type": bias_type,
                    "text": text,
                    "gold_label": gold_label,
                }

                if gold_label == "stereotypical":
                    pair["stereotypical"] = text
                elif gold_label == "anti-stereotypical":
                    pair["anti_stereotypical"] = text

                pairs.append(pair)

        return pairs
