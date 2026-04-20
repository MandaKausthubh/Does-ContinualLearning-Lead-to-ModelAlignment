"""
Inference utilities and wrappers for model interaction.
"""

import torch
from typing import List, Dict, Optional, Union, Any
from dataclasses import dataclass

from transformers import PreTrainedTokenizer


@dataclass
class GenerationResult:
    """Result of text generation."""
    text: str
    input_ids: torch.Tensor
    output_ids: torch.Tensor
    prompt: str
    metadata: Dict[str, Any]


class InferenceWrapper:
    """Wrapper for model inference with batching and caching."""

    def __init__(
        self,
        model: Any,  # LlamaModel
        batch_size: int = 8,
        max_length: int = 512,
        cache_results: bool = True
    ):
        self.model = model
        self.tokenizer = model.tokenizer
        self.batch_size = batch_size
        self.max_length = max_length
        self.cache_results = cache_results
        self._cache: Dict[str, GenerationResult] = {}

    def generate(
        self,
        prompts: Union[str, List[str]],
        max_new_tokens: Optional[int] = None,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        return_full_text: bool = False,
        use_cache: bool = True,
        **kwargs
    ) -> Union[str, List[str], GenerationResult, List[GenerationResult]]:
        """
        Generate text from prompt(s).

        Args:
            prompts: Single prompt or list of prompts
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            top_k: Top-k sampling parameter
            return_full_text: Whether to return full text or just new tokens
            use_cache: Whether to use result caching

        Returns:
            Generated text(s) or GenerationResult(s)
        """
        is_single = isinstance(prompts, str)
        if is_single:
            prompts = [prompts]

        # Check cache
        if use_cache and self.cache_results:
            cached_results = []
            uncached_prompts = []
            for p in prompts:
                if p in self._cache:
                    cached_results.append((p, self._cache[p]))
                else:
                    uncached_prompts.append(p)
                    cached_results.append((p, None))
        else:
            uncached_prompts = prompts
            cached_results = [(p, None) for p in prompts]

        # Generate for uncached prompts
        if uncached_prompts:
            new_results = self._batch_generate(
                uncached_prompts,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                return_full_text=return_full_text,
                **kwargs
            )

            # Update cache
            if use_cache and self.cache_results:
                for prompt, result in zip(uncached_prompts, new_results):
                    self._cache[prompt] = result

            # Fill in cached results
            result_iter = iter(new_results)
            final_results = []
            for p, cached in cached_results:
                if cached is not None:
                    final_results.append(cached)
                else:
                    final_results.append(next(result_iter))
        else:
            final_results = [r for _, r in cached_results]

        if is_single:
            return final_results[0]
        return final_results

    def _batch_generate(
        self,
        prompts: List[str],
        max_new_tokens: Optional[int] = None,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        return_full_text: bool = False,
        **kwargs
    ) -> List[GenerationResult]:
        """Generate for a batch of prompts."""
        if max_new_tokens is None:
            max_new_tokens = 256

        results = []

        # Process in batches
        for i in range(0, len(prompts), self.batch_size):
            batch_prompts = prompts[i:i + self.batch_size]

            # Tokenize
            inputs = self.tokenizer(
                batch_prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_length
            )

            # Generate
            outputs = self.model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                **kwargs
            )

            # Decode
            for j, prompt in enumerate(batch_prompts):
                input_ids = inputs["input_ids"][j]
                output_ids = outputs[j]

                if return_full_text:
                    text = self.tokenizer.decode(output_ids, skip_special_tokens=True)
                else:
                    # Only return generated part
                    prompt_len = len(input_ids)
                    text = self.tokenizer.decode(
                        output_ids[prompt_len:],
                        skip_special_tokens=True
                    )

                results.append(GenerationResult(
                    text=text,
                    input_ids=input_ids,
                    output_ids=output_ids,
                    prompt=prompt,
                    metadata={
                        "temperature": temperature,
                        "top_p": top_p,
                        "max_new_tokens": max_new_tokens
                    }
                ))

        return results

    def generate_for_bias_testing(
        self,
        prompts_male: List[str],
        prompts_female: List[str],
        max_new_tokens: Optional[int] = None,
        **generation_kwargs
    ) -> Dict[str, List[str]]:
        """
        Generate outputs for paired male/female prompts for bias testing.

        Returns:
            Dictionary with 'male_outputs' and 'female_outputs'
        """
        male_results = self.generate(
            prompts_male,
            max_new_tokens=max_new_tokens,
            use_cache=False,
            **generation_kwargs
        )
        female_results = self.generate(
            prompts_female,
            max_new_tokens=max_new_tokens,
            use_cache=False,
            **generation_kwargs
        )

        return {
            "male_outputs": [r.text if isinstance(r, GenerationResult) else r for r in male_results],
            "female_outputs": [r.text if isinstance(r, GenerationResult) else r for r in female_results],
            "male_prompts": prompts_male,
            "female_prompts": prompts_female
        }

    def clear_cache(self):
        """Clear the generation cache."""
        self._cache.clear()

    def get_cache_stats(self) -> Dict[str, int]:
        """Get cache statistics."""
        return {
            "size": len(self._cache),
            "max_size": getattr(self, '_max_cache_size', 'unlimited')
        }


class DistributedInferenceWrapper(InferenceWrapper):
    """Inference wrapper with distributed (TPU) support."""

    def __init__(
        self,
        model: Any,
        batch_size: int = 8,
        max_length: int = 512,
        cache_results: bool = True,
        world_size: int = 8
    ):
        super().__init__(model, batch_size, max_length, cache_results)
        self.world_size = world_size

        # TPU imports
        import torch_xla.core.xla_model as xm
        self.xm = xm

    def generate_distributed(
        self,
        prompts: List[str],
        **kwargs
    ) -> List[GenerationResult]:
        """Generate using distributed processing across TPU cores."""
        # Split prompts across devices
        device = self.xm.xla_device()
        rank = self.xm.get_ordinal()

        # Calculate partition
        total = len(prompts)
        per_device = total // self.world_size
        start = rank * per_device
        end = start + per_device if rank < self.world_size - 1 else total

        local_prompts = prompts[start:end]

        # Generate locally
        local_results = self._batch_generate(local_prompts, **kwargs)

        # Gather results from all devices
        all_results = [None] * self.world_size
        self.xm.mesh_reduce("results", local_results, lambda x: all_results.__setitem__(rank, x))

        # Flatten results
        if rank == 0:
            flat_results = []
            for r in all_results:
                if r:
                    flat_results.extend(r)
            return flat_results

        return local_results
