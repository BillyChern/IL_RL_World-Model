"""Unit tests for distributed training utilities.

Tests distributed training setup and utilities (without actual multi-GPU).

Author: Billy Chern (Shichen)
License: MIT
"""

import pytest
import torch
import torch.nn as nn

from training.utils.distributed import (
    DistributedConfig,
    is_main_process,
    get_world_size,
    get_rank,
    GradientAccumulator,
)


class TestDistributedConfig:
    """Test suite for DistributedConfig."""

    def test_default_config(self):
        """Test default configuration."""
        config = DistributedConfig()

        assert config.backend == "nccl"
        assert config.init_method == "env://"
        assert config.world_size == 8
        assert config.use_amp is True
        assert config.gradient_accumulation_steps == 1

    def test_custom_config(self):
        """Test custom configuration."""
        config = DistributedConfig(
            backend="gloo",
            world_size=4,
            use_amp=False,
            gradient_accumulation_steps=4,
        )

        assert config.backend == "gloo"
        assert config.world_size == 4
        assert config.use_amp is False
        assert config.gradient_accumulation_steps == 4

    def test_rank_from_environment(self):
        """Test rank is read from environment."""
        import os

        # Set environment variables
        os.environ["RANK"] = "2"
        os.environ["LOCAL_RANK"] = "1"

        config = DistributedConfig()

        assert config.rank == 2
        assert config.local_rank == 1

        # Clean up
        del os.environ["RANK"]
        del os.environ["LOCAL_RANK"]


class TestDistributedUtilities:
    """Test distributed utility functions (single GPU mode)."""

    def test_is_main_process_single_gpu(self):
        """Test is_main_process in single GPU mode."""
        # Without distributed initialization, should always be True
        assert is_main_process() is True

    def test_get_world_size_single_gpu(self):
        """Test get_world_size in single GPU mode."""
        # Without distributed initialization, should be 1
        assert get_world_size() == 1

    def test_get_rank_single_gpu(self):
        """Test get_rank in single GPU mode."""
        # Without distributed initialization, should be 0
        assert get_rank() == 0


class TestGradientAccumulator:
    """Test suite for gradient accumulation."""

    @pytest.fixture
    def simple_model(self):
        """Create simple model for testing."""
        model = nn.Sequential(
            nn.Linear(10, 5),
            nn.ReLU(),
            nn.Linear(5, 1),
        )
        return model

    @pytest.fixture
    def optimizer(self, simple_model):
        """Create optimizer."""
        return torch.optim.Adam(simple_model.parameters(), lr=1e-3)

    def test_initialization(self, simple_model, optimizer):
        """Test accumulator initialization."""
        accumulator = GradientAccumulator(
            model=simple_model,
            optimizer=optimizer,
            accumulation_steps=4,
            max_grad_norm=1.0,
        )

        assert accumulator.accumulation_steps == 4
        assert accumulator.max_grad_norm == 1.0
        assert accumulator.current_step == 0

    def test_single_step(self, simple_model, optimizer):
        """Test single accumulation step."""
        accumulator = GradientAccumulator(
            model=simple_model,
            optimizer=optimizer,
            accumulation_steps=1,  # No accumulation
        )

        # Create dummy input and loss
        x = torch.randn(8, 10)
        y_pred = simple_model(x)
        loss = y_pred.mean()

        # Step should update optimizer
        updated = accumulator.step(loss)
        assert updated is True
        assert accumulator.current_step == 1

    def test_multi_step_accumulation(self, simple_model, optimizer):
        """Test multi-step gradient accumulation."""
        accumulator = GradientAccumulator(
            model=simple_model,
            optimizer=optimizer,
            accumulation_steps=4,
        )

        # First 3 steps should not update optimizer
        for i in range(3):
            x = torch.randn(8, 10)
            y_pred = simple_model(x)
            loss = y_pred.mean()

            updated = accumulator.step(loss)
            assert updated is False
            assert accumulator.current_step == i + 1

        # 4th step should update
        x = torch.randn(8, 10)
        y_pred = simple_model(x)
        loss = y_pred.mean()

        updated = accumulator.step(loss)
        assert updated is True
        assert accumulator.current_step == 4

    def test_gradient_clipping(self, simple_model, optimizer):
        """Test gradient clipping during accumulation."""
        accumulator = GradientAccumulator(
            model=simple_model,
            optimizer=optimizer,
            accumulation_steps=1,
            max_grad_norm=1.0,
        )

        # Create large loss to generate large gradients
        x = torch.randn(8, 10)
        y_pred = simple_model(x)
        loss = y_pred.mean() * 1000  # Large multiplier

        # Should clip gradients
        updated = accumulator.step(loss)
        assert updated is True

        # Check that gradients were clipped
        total_norm = 0.0
        for p in simple_model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5

        # After clipping, norm should be <= max_grad_norm
        assert total_norm <= accumulator.max_grad_norm + 1e-5


class TestSetup8GPUTraining:
    """Test 8-GPU training setup (mock)."""

    def test_setup_single_gpu(self):
        """Test setup in single GPU mode."""
        from training.utils.distributed import setup_8gpu_training

        # Should work even without 8 GPUs
        config = setup_8gpu_training(use_ddp=False)

        # In single GPU mode, config is None
        assert config is None

    def test_gpu_detection(self):
        """Test GPU detection."""
        num_gpus = torch.cuda.device_count()

        if num_gpus > 0:
            print(f"Detected {num_gpus} GPU(s)")
            for i in range(min(num_gpus, 8)):
                props = torch.cuda.get_device_properties(i)
                assert props is not None
                print(f"  GPU {i}: {props.name}")
        else:
            print("No GPUs detected (CPU-only mode)")


class TestJAXDistributed:
    """Test JAX distributed setup."""

    def test_jax_gpu_detection(self):
        """Test JAX GPU detection."""
        try:
            import jax

            devices = jax.devices("gpu")
            print(f"JAX detected {len(devices)} GPU(s)")

            if len(devices) > 0:
                for i, device in enumerate(devices[:8]):
                    print(f"  JAX GPU {i}: {device.device_kind}")

        except Exception as e:
            print(f"JAX not available or error: {e}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
