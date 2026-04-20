"""
Model loading and management for LLaMA models with LoRA support.
"""

import os
import torch
from typing import Dict, Optional, Any
from dataclasses import dataclass

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
    BitsAndBytesConfig
)
from peft import (
    LoraConfig,
    get_peft_model,
    PeftModel,
    prepare_model_for_kbit_training
)

# TPU imports
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.distributed.xla_multiprocessing as xmp
import torch_xla.distributed.parallel_loader as pl


@dataclass
class ModelConfig:
    """Configuration for model loading."""
    model_name: str = "meta-llama/Llama-3.2-1B"
    cache_dir: str = "./cache/models"
    torch_dtype: str = "bfloat16"

    # LoRA config
    use_lora: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: Optional[list] = None

    # Device
    device_type: str = "tpu"  # tpu, cuda, cpu

    def __post_init__(self):
        if self.lora_target_modules is None:
            self.lora_target_modules = ["q_proj", "v_proj", "k_proj", "o_proj"]


class LlamaModel:
    """Wrapper for LLaMA model with tokenizer."""

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        config: ModelConfig,
        is_reference: bool = False
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.is_reference = is_reference

    def get_device(self):
        """Get the device the model is on."""
        if self.config.device_type == "tpu":
            return xm.xla_device()
        return next(self.model.parameters()).device

    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        do_sample: bool = True,
        **kwargs
    ) -> torch.Tensor:
        """Generate text using the model."""
        device = self.get_device()
        input_ids = input_ids.to(device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)

        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                do_sample=do_sample,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                **kwargs
            )

        return outputs

    def save(self, path: str):
        """Save model and tokenizer."""
        os.makedirs(path, exist_ok=True)
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)

    def load_weights(self, path: str):
        """Load model weights from path."""
        if self.config.use_lora and os.path.exists(os.path.join(path, "adapter_config.json")):
            self.model = PeftModel.from_pretrained(self.model, path)
        else:
            self.model = AutoModelForCausalLM.from_pretrained(path)

    def freeze(self):
        """Freeze all parameters (for reference model)."""
        for param in self.model.parameters():
            param.requires_grad = False
        self.model.eval()

    def unfreeze(self):
        """Unfreeze parameters for training."""
        for param in self.model.parameters():
            param.requires_grad = True
        self.model.train()


class ModelLoader:
    """Factory class for loading models with various configurations."""

    SUPPORTED_MODELS = {
        "llama-1b": "meta-llama/Llama-3.2-1B",
        "llama-3b": "meta-llama/Llama-3.2-3B",
        "llama-8b": "meta-llama/Llama-3.2-8B",
        "llama-3.2-1b": "meta-llama/Llama-3.2-1B",
        "llama-3.2-3b": "meta-llama/Llama-3.2-3B",
        "llama-3.2-8b": "meta-llama/Llama-3.2-8B",
    }

    def __init__(self, cache_dir: str = "./cache/models"):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)

    def _resolve_model_name(self, name: str) -> str:
        """Resolve short names to full HuggingFace model names."""
        name_lower = name.lower()
        if name_lower in self.SUPPORTED_MODELS:
            return self.SUPPORTED_MODELS[name_lower]
        return name

    def _get_dtype(self, dtype_str: str):
        """Convert string dtype to torch dtype."""
        dtype_map = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        return dtype_map.get(dtype_str.lower(), torch.bfloat16)

    def load_tokenizer(self, model_name: str) -> PreTrainedTokenizer:
        """Load tokenizer for the model."""
        resolved_name = self._resolve_model_name(model_name)

        tokenizer = AutoTokenizer.from_pretrained(
            resolved_name,
            cache_dir=self.cache_dir,
            trust_remote_code=True
        )

        # Set pad token if not defined
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id

        return tokenizer

    def load_base_model(
        self,
        config: ModelConfig,
        for_training: bool = True
    ) -> PreTrainedModel:
        """Load base model without LoRA."""
        model_name = self._resolve_model_name(config.model_name)
        dtype = self._get_dtype(config.torch_dtype)

        print(f"Loading base model: {model_name}")

        if config.device_type == "tpu":
            # TPU specific loading
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                cache_dir=config.cache_dir,
                torch_dtype=dtype,
                trust_remote_code=True
            )
            # Move to TPU device
            device = xm.xla_device()
            model = model.to(device)
        else:
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                cache_dir=config.cache_dir,
                torch_dtype=dtype,
                device_map="auto" if config.device_type == "cuda" else None,
                trust_remote_code=True
            )

        # Prepare for training if needed
        if for_training and config.use_lora:
            model = prepare_model_for_kbit_training(model)

        return model

    def apply_lora(self, model: PreTrainedModel, config: ModelConfig) -> PeftModel:
        """Apply LoRA to a base model."""
        if not config.use_lora:
            return model

        lora_config = LoraConfig(
            r=config.lora_r,
            lora_alpha=config.lora_alpha,
            target_modules=config.lora_target_modules,
            lora_dropout=config.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )

        model = get_peft_model(model, lora_config)
        print(f"Applied LoRA (r={config.lora_r}, alpha={config.lora_alpha})")
        model.print_trainable_parameters()

        return model

    def load(
        self,
        config: ModelConfig,
        for_training: bool = True
    ) -> LlamaModel:
        """Load complete model with tokenizer and optional LoRA."""
        tokenizer = self.load_tokenizer(config.model_name)
        base_model = self.load_base_model(config, for_training)

        if config.use_lora and for_training:
            model = self.apply_lora(base_model, config)
        else:
            model = base_model

        return LlamaModel(
            model=model,
            tokenizer=tokenizer,
            config=config,
            is_reference=False
        )

    def load_from_checkpoint(
        self,
        checkpoint_path: str,
        config: ModelConfig,
        for_training: bool = False
    ) -> LlamaModel:
        """Load model from a saved checkpoint."""
        tokenizer = self.load_tokenizer(config.model_name)

        if config.use_lora and os.path.exists(os.path.join(checkpoint_path, "adapter_config.json")):
            # Load LoRA adapter
            base_model = self.load_base_model(config, for_training=False)
            model = PeftModel.from_pretrained(base_model, checkpoint_path)
        else:
            model = AutoModelForCausalLM.from_pretrained(
                checkpoint_path,
                torch_dtype=self._get_dtype(config.torch_dtype),
                trust_remote_code=True
            )

        if config.device_type == "tpu":
            device = xm.xla_device()
            model = model.to(device)

        return LlamaModel(
            model=model,
            tokenizer=tokenizer,
            config=config,
            is_reference=not for_training
        )


def create_reference_model(model: LlamaModel) -> LlamaModel:
    """
    Create a frozen copy of a model for self-distillation.

    This creates a reference model that:
    1. Is initialized from the same weights as the training model
    2. Is frozen (no gradients)
    3. Runs in eval mode
    """
    reference_config = ModelConfig(
        model_name=model.config.model_name,
        cache_dir=model.config.cache_dir,
        torch_dtype=model.config.torch_dtype,
        device_type=model.config.device_type,
        use_lora=False  # Reference model doesn't need LoRA
    )

    # Load fresh copy of base model (without LoRA adapters)
    loader = ModelLoader()
    base_model = loader.load_base_model(reference_config, for_training=False)

    # Copy weights from current model
    # If model has LoRA, we need to merge and unload first
    if hasattr(model.model, 'merge_and_unload'):
        merged = model.model.merge_and_unload()
        base_model.load_state_dict(merged.state_dict())
    else:
        base_model.load_state_dict(model.model.state_dict())

    # Create wrapper and freeze
    reference = LlamaModel(
        model=base_model,
        tokenizer=model.tokenizer,  # Share tokenizer
        config=reference_config,
        is_reference=True
    )
    reference.freeze()

    return reference
