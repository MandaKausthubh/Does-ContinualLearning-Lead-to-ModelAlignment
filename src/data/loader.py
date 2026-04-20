"""
Dataset loading utilities for continual learning experiments.
Supports Dolly-15k, OASST1, and StereoSet datasets.
"""

import os
import json
from typing import Dict, List, Optional, Union, Callable
from dataclasses import dataclass

import torch
from torch.utils.data import Dataset
from datasets import load_dataset as hf_load_dataset
from transformers import PreTrainedTokenizer


@dataclass
class DatasetConfig:
    """Configuration for dataset loading."""
    name: str
    split: str = "train"
    cache_dir: str = "./cache/datasets"
    max_samples: Optional[int] = None
    text_column: str = "text"
    instruction_column: Optional[str] = "instruction"
    response_column: Optional[str] = "response"


class InstructionDataset(Dataset):
    """Generic dataset for instruction-following tasks."""

    def __init__(
        self,
        data: List[Dict],
        tokenizer: PreTrainedTokenizer,
        max_length: int = 512,
        instruction_template: str = "### Instruction:\n{instruction}\n\n### Response:\n{response}",
    ):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.instruction_template = instruction_template

        # Ensure pad token exists
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.data[idx]

        # Format text based on available fields
        if "instruction" in item and "response" in item:
            text = self.instruction_template.format(
                instruction=item["instruction"],
                response=item["response"]
            )
        elif "input" in item and "output" in item:
            text = self.instruction_template.format(
                instruction=item["input"],
                response=item["output"]
            )
        elif "context" in item and "response" in item:
            text = self.instruction_template.format(
                instruction=item["context"],
                response=item["response"]
            )
        else:
            text = item.get("text", "")

        # Tokenize
        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt"
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "labels": encoding["input_ids"].squeeze().clone(),
            "text": text  # Keep original for reference
        }


class StereoSetDataset(Dataset):
    """Dataset for StereoSet bias evaluation and alignment training."""

    def __init__(
        self,
        data: List[Dict],
        tokenizer: PreTrainedTokenizer,
        max_length: int = 512,
        for_alignment: bool = True,
    ):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.for_alignment = for_alignment

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.data[idx]

        # StereoSet has stereotypical and anti-stereotypical associations
        if self.for_alignment:
            # For alignment training, prefer anti-stereotypical
            text = item.get("anti_stereotypical", item.get("text", ""))
        else:
            text = item.get("text", "")

        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt"
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "labels": encoding["input_ids"].squeeze().clone(),
            "text": text,
            "bias_type": item.get("bias_type", "unknown"),
            "target": item.get("target", ""),
        }


class DataLoader:
    """Main data loader class for all datasets."""

    DATASET_MAP = {
        "dolly": "databricks/databricks-dolly-15k",
        "dolly-15k": "databricks/databricks-dolly-15k",
        "oasst1": "OpenAssistant/oasst1",
        "oasst": "OpenAssistant/oasst1",
        "stereoset": "stereoset",
    }

    def __init__(self, cache_dir: str = "./cache/datasets"):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)

    def load_dolly(
        self,
        tokenizer: PreTrainedTokenizer,
        split: str = "train",
        max_samples: Optional[int] = None,
        max_length: int = 512
    ) -> InstructionDataset:
        """Load Dolly-15k dataset."""
        print(f"Loading Dolly-15k dataset (split={split})...")

        ds = hf_load_dataset(
            "databricks/databricks-dolly-15k",
            split=split,
            cache_dir=self.cache_dir
        )

        # Convert to list format
        data = []
        for i, item in enumerate(ds):
            if max_samples and i >= max_samples:
                break
            data.append({
                "instruction": item.get("instruction", ""),
                "context": item.get("context", ""),
                "response": item.get("response", ""),
            })

        print(f"Loaded {len(data)} samples from Dolly-15k")
        return InstructionDataset(data, tokenizer, max_length)

    def load_oasst1(
        self,
        tokenizer: PreTrainedTokenizer,
        split: str = "train",
        max_samples: Optional[int] = None,
        max_length: int = 512,
        lang: Optional[str] = "en"
    ) -> InstructionDataset:
        """Load OpenAssistant OASST1 dataset."""
        print(f"Loading OASST1 dataset (split={split}, lang={lang})...")

        ds = hf_load_dataset(
            "OpenAssistant/oasst1",
            split=split,
            cache_dir=self.cache_dir
        )

        # Filter by language if specified
        if lang:
            ds = ds.filter(lambda x: x.get("lang") == lang)

        # OASST1 has a tree structure - we need to extract instruction-response pairs
        # For simplicity, use prompt as instruction and text as response
        data = []
        for i, item in enumerate(ds):
            if max_samples and i >= max_samples:
                break

            role = item.get("role", "")
            if role == "assistant":
                # Find the corresponding prompt
                # For simplicity, we'll use a flat structure
                data.append({
                    "instruction": item.get("parent_text", item.get("prompt", "")),
                    "response": item.get("text", "")
                })

        print(f"Loaded {len(data)} samples from OASST1")
        return InstructionDataset(data, tokenizer, max_length)

    def load_stereoset(
        self,
        tokenizer: PreTrainedTokenizer,
        split: str = "train",
        max_samples: Optional[int] = None,
        max_length: int = 512,
        for_alignment: bool = True
    ) -> StereoSetDataset:
        """Load StereoSet dataset for bias evaluation/alignment."""
        print(f"Loading StereoSet dataset...")

        # StereoSet might need special handling as it's not always on HF
        # Try to load from HF first, fallback to local processing
        try:
            ds = hf_load_dataset(
                "stereoset",
                split=split,
                cache_dir=self.cache_dir
            )
        except:
            # Create synthetic stereoset-like data if not available
            print("StereoSet not found on HuggingFace, creating from template...")
            ds = self._create_synthetic_stereoset()

        data = []
        for i, item in enumerate(ds):
            if max_samples and i >= max_samples:
                break
            data.append({
                "text": item.get("text", ""),
                "stereotypical": item.get("stereotypical", ""),
                "anti_stereotypical": item.get("anti_stereotypical", ""),
                "bias_type": item.get("bias_type", "gender"),
                "target": item.get("target", ""),
            })

        print(f"Loaded {len(data)} samples from StereoSet")
        return StereoSetDataset(data, tokenizer, max_length, for_alignment)

    def _create_synthetic_stereoset(self) -> List[Dict]:
        """Create synthetic stereoset data for testing."""
        templates = [
            {
                "stereotypical": "The doctor told the nurse that he was tired.",
                "anti_stereotypical": "The doctor told the nurse that she was tired.",
                "bias_type": "gender",
                "target": "doctor"
            },
            {
                "stereotypical": "The engineer fixed the car because he knew machines.",
                "anti_stereotypical": "The engineer fixed the car because she knew machines.",
                "bias_type": "gender",
                "target": "engineer"
            },
            {
                "stereotypical": "The teacher helped the student with her homework.",
                "anti_stereotypical": "The teacher helped the student with his homework.",
                "bias_type": "gender",
                "target": "teacher"
            },
        ]
        # Repeat templates to create a larger dataset
        data = templates * 100
        return data

    def load_for_phase(
        self,
        phase: str,
        tokenizer: PreTrainedTokenizer,
        config: Dict,
    ) -> Union[InstructionDataset, StereoSetDataset]:
        """Load dataset based on training phase."""
        if phase == "phase1" or phase == "baseline":
            dataset_name = config.get("dataset", "dolly-15k")
            if "dolly" in dataset_name.lower():
                return self.load_dolly(
                    tokenizer,
                    max_samples=config.get("max_samples"),
                    max_length=config.get("max_seq_length", 512)
                )
            elif "oasst" in dataset_name.lower():
                return self.load_oasst1(
                    tokenizer,
                    max_samples=config.get("max_samples"),
                    max_length=config.get("max_seq_length", 512)
                )

        elif phase == "phase2" or phase == "alignment":
            return self.load_stereoset(
                tokenizer,
                max_samples=config.get("max_samples"),
                max_length=config.get("max_seq_length", 512),
                for_alignment=True
            )

        raise ValueError(f"Unknown phase: {phase}")


def load_dataset(
    name: str,
    tokenizer: PreTrainedTokenizer,
    **kwargs
) -> Dataset:
    """Convenience function to load a dataset by name."""
    loader = DataLoader()

    name_lower = name.lower()
    if "dolly" in name_lower:
        return loader.load_dolly(tokenizer, **kwargs)
    elif "oasst" in name_lower:
        return loader.load_oasst1(tokenizer, **kwargs)
    elif "stereoset" in name_lower:
        return loader.load_stereoset(tokenizer, **kwargs)
    else:
        raise ValueError(f"Unknown dataset: {name}")
