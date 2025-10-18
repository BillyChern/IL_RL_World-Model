"""Unit tests for π0.5 VLA policy.

Tests the Pi0Policy implementation to ensure correct behavior
for vision-language-action prediction.

Author: Billy Chern (Shichen)
License: MIT
"""

import numpy as np
import pytest
import torch

from models.vla import Pi0Config, Pi0Policy


class TestPi0Config:
    """Test suite for Pi0Config dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        config = Pi0Config()

        assert config.action_dim == 14
        assert config.chunk_size == 50
        assert config.num_diffusion_steps == 10
        assert config.num_cameras == 3
        assert config.image_size == (224, 224)
        assert config.freeze_vision is True
        assert config.freeze_language is True

    def test_custom_config(self):
        """Test custom configuration."""
        config = Pi0Config(
            action_dim=7,
            chunk_size=20,
            num_diffusion_steps=5,
            freeze_vision=False,
        )

        assert config.action_dim == 7
        assert config.chunk_size == 20
        assert config.num_diffusion_steps == 5
        assert config.freeze_vision is False


@pytest.mark.slow
class TestPi0Policy:
    """Test suite for Pi0Policy class.

    These tests require downloading pretrained models, so they're marked as slow.
    """

    @pytest.fixture
    def config(self):
        """Create minimal config for testing."""
        return Pi0Config(
            action_dim=14,
            chunk_size=10,  # Smaller for faster testing
            num_diffusion_steps=2,  # Fewer steps for testing
            device="cpu",  # Use CPU for testing
        )

    @pytest.mark.skip(reason="Requires pretrained weights download")
    def test_model_initialization(self, config):
        """Test model initializes correctly."""
        model = Pi0Policy(config)

        assert isinstance(model, Pi0Policy)
        assert model.config == config
        assert hasattr(model, "vision_encoder")
        assert hasattr(model, "language_encoder")
        assert hasattr(model, "action_decoder")

    @pytest.mark.skip(reason="Requires pretrained weights download")
    def test_forward_training(self, config):
        """Test forward pass in training mode."""
        model = Pi0Policy(config)
        model.train()

        # Create dummy inputs
        batch_size = 2
        images = {
            "left": torch.randn(batch_size, 3, 224, 224),
            "right": torch.randn(batch_size, 3, 224, 224),
            "base": torch.randn(batch_size, 3, 224, 224),
        }
        instructions = ["pick up the cup", "place the object"]
        actions = torch.randn(batch_size, config.chunk_size, config.action_dim)

        # Forward pass
        pred_noise, true_noise = model(images, instructions, actions)

        # Check shapes
        expected_shape = (batch_size, config.chunk_size * config.action_dim)
        assert pred_noise.shape == expected_shape
        assert true_noise.shape == expected_shape

    @pytest.mark.skip(reason="Requires pretrained weights download")
    def test_forward_inference(self, config):
        """Test forward pass in inference mode."""
        model = Pi0Policy(config)
        model.eval()

        # Create dummy inputs
        batch_size = 2
        images = {
            "left": torch.randn(batch_size, 3, 224, 224),
            "right": torch.randn(batch_size, 3, 224, 224),
            "base": torch.randn(batch_size, 3, 224, 224),
        }
        instructions = ["pick up the cup", "place the object"]

        # Forward pass
        with torch.no_grad():
            actions = model(images, instructions)

        # Check shape
        expected_shape = (batch_size, config.chunk_size, config.action_dim)
        assert actions.shape == expected_shape

    @pytest.mark.skip(reason="Requires pretrained weights download")
    def test_predict(self, config):
        """Test predict method for deployment."""
        model = Pi0Policy(config)

        # Create dummy images (numpy format)
        images = {
            "left": np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8),
            "right": np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8),
            "base": np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8),
        }
        instruction = "pick up the cup"

        # Predict
        actions = model.predict(images, instruction)

        # Check output
        assert isinstance(actions, np.ndarray)
        assert actions.shape == (config.chunk_size, config.action_dim)
        assert actions.dtype == np.float32 or actions.dtype == np.float64

    def test_checkpoint_save_load(self, config, tmp_path):
        """Test checkpoint saving and loading."""
        # Create model
        model1 = Pi0Policy(config)

        # Save checkpoint
        checkpoint_path = tmp_path / "test_checkpoint.pt"
        model1.save_checkpoint(checkpoint_path)

        assert checkpoint_path.exists()

        # Load checkpoint
        model2 = Pi0Policy.from_checkpoint(checkpoint_path)

        # Check config matches
        assert model2.config.action_dim == config.action_dim
        assert model2.config.chunk_size == config.chunk_size

    @pytest.mark.skip(reason="Requires pretrained weights download")
    def test_from_pretrained(self):
        """Test loading pretrained model."""
        config = Pi0Config(device="cpu")

        # Load pretrained (vision and language only)
        model = Pi0Policy.from_pretrained(config=config)

        assert isinstance(model, Pi0Policy)
        assert model.config.action_dim == 14


class TestDiffusionLoss:
    """Test suite for diffusion loss computation."""

    def test_diffusion_loss(self):
        """Test diffusion loss computation."""
        from models.vla.pi0_policy import compute_diffusion_loss

        batch_size = 4
        dim = 100

        pred_noise = torch.randn(batch_size, dim)
        true_noise = torch.randn(batch_size, dim)

        loss = compute_diffusion_loss(pred_noise, true_noise)

        # Check loss is scalar
        assert loss.ndim == 0
        assert loss.item() >= 0.0

    def test_diffusion_loss_zero(self):
        """Test diffusion loss is zero for identical predictions."""
        from models.vla.pi0_policy import compute_diffusion_loss

        batch_size = 4
        dim = 100

        noise = torch.randn(batch_size, dim)
        loss = compute_diffusion_loss(noise, noise)

        assert loss.item() < 1e-6


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "--tb=short", "-m", "not slow"])
