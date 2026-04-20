"""
Random seed management for reproducibility.
"""

import os
import random
import numpy as np
import torch

try:
    import torch_xla.core.xla_model as xm
    HAS_XLA = True
except ImportError:
    HAS_XLA = False


def set_seed(seed: int = 42):
    """
    Set random seeds for reproducibility.

    Sets seeds for:
    - Python random
    - NumPy
    - PyTorch (CPU and CUDA)
    - PyTorch/XLA (if available)

    Args:
        seed: Random seed value
    """
    # Python random
    random.seed(seed)

    # NumPy
    np.random.seed(seed)

    # PyTorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # CuDNN determinism (may impact performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # TPU (if available)
    if HAS_XLA:
        import torch_xla.core.xla_model as xm
        # TPU seeds are handled through the device
        pass

    # Set environment variable
    os.environ["PYTHONHASHSEED"] = str(seed)

    print(f"Set random seed to {seed}")


def get_worker_init_fn(seed: int = 42):
    """
    Get worker initialization function for DataLoader.

    Ensures each worker has a unique but deterministic seed.

    Args:
        seed: Base seed

    Returns:
        Worker initialization function
    """
    def worker_init_fn(worker_id):
        worker_seed = seed + worker_id
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    return worker_init_fn


class SeedContext:
    """
    Context manager for temporarily setting a seed.

    Example:
        with SeedContext(42):
            # Code with seed 42
            pass
        # Back to previous seed
    """

    def __init__(self, seed: int = 42):
        self.seed = seed
        self.prev_state = None

    def __enter__(self):
        # Save current random state
        self.prev_state = {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        }

        # Set new seed
        set_seed(self.seed)

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Restore previous random state
        if self.prev_state:
            random.setstate(self.prev_state["python"])
            np.random.set_state(self.prev_state["numpy"])
            torch.set_rng_state(self.prev_state["torch"])
            if self.prev_state["torch_cuda"] is not None:
                torch.cuda.set_rng_state_all(self.prev_state["torch_cuda"])
