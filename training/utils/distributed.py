"""Multi-GPU distributed training utilities.

Supports 8x H100 GPU distributed training with:
- PyTorch DDP (Distributed Data Parallel)
- JAX pmap for world model
- Gradient accumulation
- Mixed precision training

Author: Billy Chern (Shichen)
License: MIT
"""

import os
from typing import Optional

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

try:
    import jax
    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False
    print("Warning: JAX not available. World model distributed training disabled.")


class DistributedConfig:
    """Configuration for distributed training."""

    def __init__(
        self,
        backend: str = "nccl",  # nccl for GPU
        init_method: str = "env://",
        world_size: int = 8,  # 8x H100
        rank: Optional[int] = None,
        local_rank: Optional[int] = None,
        use_amp: bool = True,  # Automatic mixed precision
        gradient_accumulation_steps: int = 1,
    ):
        """Initialize distributed config.

        Args:
            backend: Distributed backend (nccl, gloo)
            init_method: Initialization method
            world_size: Total number of processes
            rank: Global rank (set by environment)
            local_rank: Local rank on current node
            use_amp: Use automatic mixed precision
            gradient_accumulation_steps: Gradient accumulation steps
        """
        self.backend = backend
        self.init_method = init_method
        self.world_size = world_size
        self.use_amp = use_amp
        self.gradient_accumulation_steps = gradient_accumulation_steps

        # Get rank from environment if not provided
        self.rank = rank if rank is not None else int(os.environ.get("RANK", 0))
        self.local_rank = local_rank if local_rank is not None else int(
            os.environ.get("LOCAL_RANK", 0)
        )


def setup_distributed(config: DistributedConfig) -> bool:
    """Initialize distributed training.

    Args:
        config: Distributed configuration

    Returns:
        True if distributed training is enabled
    """
    # Check if distributed training is available
    if not dist.is_available():
        print("Distributed training not available")
        return False

    # Initialize process group
    if not dist.is_initialized():
        dist.init_process_group(
            backend=config.backend,
            init_method=config.init_method,
            world_size=config.world_size,
            rank=config.rank,
        )

    # Set device
    torch.cuda.set_device(config.local_rank)

    print(f"✓ Distributed training initialized")
    print(f"  Rank: {config.rank}/{config.world_size}")
    print(f"  Local rank: {config.local_rank}")
    print(f"  Device: cuda:{config.local_rank}")

    return True


def cleanup_distributed():
    """Cleanup distributed training."""
    if dist.is_initialized():
        dist.destroy_process_group()
        print("✓ Distributed training cleaned up")


def wrap_model_ddp(
    model: torch.nn.Module,
    device_id: int,
    find_unused_parameters: bool = False,
) -> DDP:
    """Wrap model with DistributedDataParallel.

    Args:
        model: Model to wrap
        device_id: GPU device ID
        find_unused_parameters: Find unused parameters

    Returns:
        DDP-wrapped model
    """
    model = model.to(device_id)
    model = DDP(
        model,
        device_ids=[device_id],
        output_device=device_id,
        find_unused_parameters=find_unused_parameters,
    )

    print(f"✓ Model wrapped with DDP (device: cuda:{device_id})")
    return model


def setup_jax_distributed(num_devices: int = 8) -> None:
    """Setup JAX for distributed training.

    Args:
        num_devices: Number of GPUs
    """
    if not JAX_AVAILABLE:
        print("JAX not available - skipping JAX distributed setup")
        return

    # JAX automatically detects GPUs
    devices = jax.devices("gpu")

    if len(devices) < num_devices:
        print(f"Warning: Requested {num_devices} GPUs but only {len(devices)} available")

    print(f"✓ JAX distributed setup")
    print(f"  Devices: {len(devices)}")
    print(f"  Device type: {devices[0].device_kind if devices else 'None'}")

    # Print device info
    for i, device in enumerate(devices[:num_devices]):
        print(f"  GPU {i}: {device}")


def all_reduce_mean(tensor: torch.Tensor) -> torch.Tensor:
    """All-reduce tensor across GPUs (mean).

    Args:
        tensor: Tensor to reduce

    Returns:
        Reduced tensor
    """
    if not dist.is_initialized():
        return tensor

    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    tensor = tensor / dist.get_world_size()

    return tensor


def gather_tensors(tensor: torch.Tensor) -> torch.Tensor:
    """Gather tensors from all GPUs.

    Args:
        tensor: Tensor to gather

    Returns:
        Gathered tensor (only valid on rank 0)
    """
    if not dist.is_initialized():
        return tensor

    world_size = dist.get_world_size()
    gathered = [torch.zeros_like(tensor) for _ in range(world_size)]

    dist.all_gather(gathered, tensor)

    if dist.get_rank() == 0:
        return torch.cat(gathered, dim=0)
    else:
        return tensor


def is_main_process() -> bool:
    """Check if current process is main (rank 0).

    Returns:
        True if main process
    """
    if not dist.is_initialized():
        return True

    return dist.get_rank() == 0


def get_world_size() -> int:
    """Get world size (number of processes).

    Returns:
        World size
    """
    if not dist.is_initialized():
        return 1

    return dist.get_world_size()


def get_rank() -> int:
    """Get current process rank.

    Returns:
        Process rank
    """
    if not dist.is_initialized():
        return 0

    return dist.get_rank()


class GradientAccumulator:
    """Gradient accumulation for memory-efficient training."""

    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        accumulation_steps: int = 1,
        max_grad_norm: Optional[float] = None,
    ):
        """Initialize gradient accumulator.

        Args:
            model: Model to train
            optimizer: Optimizer
            accumulation_steps: Number of steps to accumulate
            max_grad_norm: Maximum gradient norm for clipping
        """
        self.model = model
        self.optimizer = optimizer
        self.accumulation_steps = accumulation_steps
        self.max_grad_norm = max_grad_norm

        self.current_step = 0

    def step(self, loss: torch.Tensor) -> bool:
        """Accumulate gradients and step optimizer.

        Args:
            loss: Loss value

        Returns:
            True if optimizer stepped
        """
        # Scale loss by accumulation steps
        loss = loss / self.accumulation_steps

        # Backward pass
        loss.backward()

        self.current_step += 1

        # Step optimizer if accumulated enough gradients
        if self.current_step % self.accumulation_steps == 0:
            # Gradient clipping
            if self.max_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.max_grad_norm,
                )

            # Optimizer step
            self.optimizer.step()
            self.optimizer.zero_grad()

            return True

        return False


def setup_8gpu_training(
    use_ddp: bool = True,
    backend: str = "nccl",
) -> Optional[DistributedConfig]:
    """Setup training for 8x H100 GPUs.

    Args:
        use_ddp: Use DistributedDataParallel
        backend: Distributed backend

    Returns:
        Distributed config if using DDP
    """
    # Check GPU availability
    if not torch.cuda.is_available():
        print("CUDA not available!")
        return None

    num_gpus = torch.cuda.device_count()
    print(f"Available GPUs: {num_gpus}")

    for i in range(min(num_gpus, 8)):
        props = torch.cuda.get_device_properties(i)
        print(f"  GPU {i}: {props.name} ({props.total_memory / 1e9:.1f} GB)")

    if num_gpus < 8:
        print(f"Warning: Only {num_gpus} GPUs available (expected 8 H100s)")

    # Setup distributed if using DDP
    if use_ddp and num_gpus > 1:
        config = DistributedConfig(
            backend=backend,
            world_size=min(num_gpus, 8),
        )

        setup_distributed(config)
        setup_jax_distributed(num_devices=min(num_gpus, 8))

        return config
    else:
        print("Using single-GPU training")
        setup_jax_distributed(num_devices=1)
        return None
