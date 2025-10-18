"""Unit tests for W&B logging utilities.

Tests the WandbLogger with disabled mode (no actual W&B calls).

Author: Billy Chern (Shichen)
License: MIT
"""

import numpy as np
import pytest
from pathlib import Path

from training.utils import WandbLogger


class TestWandbLogger:
    """Test suite for WandbLogger."""

    @pytest.fixture
    def disabled_logger(self):
        """Create logger with disabled mode (no W&B calls)."""
        return WandbLogger(
            project="test_project",
            name="test_run",
            mode="disabled",
        )

    def test_initialization_disabled(self, disabled_logger):
        """Test logger initializes in disabled mode."""
        assert disabled_logger.enabled is False

    def test_log_disabled(self, disabled_logger):
        """Test logging when disabled doesn't raise errors."""
        metrics = {
            'loss': 0.5,
            'accuracy': 0.95,
            'step': 100,
        }

        # Should not raise any errors
        disabled_logger.log(metrics, step=100)

    def test_log_video_disabled(self, disabled_logger):
        """Test video logging when disabled."""
        frames = np.random.randint(0, 255, (10, 64, 64, 3), dtype=np.uint8)

        # Should not raise errors
        disabled_logger.log_video(frames, "test_video", step=50)

    def test_log_image_disabled(self, disabled_logger):
        """Test image logging when disabled."""
        image = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)

        # Should not raise errors
        disabled_logger.log_image(image, "test_image", step=25, caption="Test")

    def test_log_histogram_disabled(self, disabled_logger):
        """Test histogram logging when disabled."""
        values = np.random.randn(1000)

        # Should not raise errors
        disabled_logger.log_histogram(values, "test_histogram", step=10)

    def test_save_artifact_disabled(self, disabled_logger, tmp_path):
        """Test artifact saving when disabled."""
        # Create temporary file
        test_file = tmp_path / "test_model.pt"
        test_file.write_text("test data")

        # Should not raise errors
        disabled_logger.save_artifact(test_file, "test_artifact", "model")

    def test_finish_disabled(self, disabled_logger):
        """Test finish when disabled."""
        # Should not raise errors
        disabled_logger.finish()

    def test_multiple_logs(self, disabled_logger):
        """Test multiple log calls."""
        for step in range(10):
            metrics = {
                'loss': 1.0 / (step + 1),
                'accuracy': step / 10.0,
            }
            disabled_logger.log(metrics, step=step)

        # Should complete without errors


class TestWandbLoggerConfiguration:
    """Test W&B logger configuration."""

    def test_initialization_with_config(self):
        """Test initialization with configuration dict."""
        config = {
            'learning_rate': 1e-4,
            'batch_size': 256,
            'num_layers': 3,
        }

        logger = WandbLogger(
            project="test_project",
            config=config,
            mode="disabled",
        )

        assert logger.enabled is False

    def test_initialization_with_entity(self):
        """Test initialization with entity."""
        logger = WandbLogger(
            project="test_project",
            entity="test_entity",
            mode="disabled",
        )

        assert logger.enabled is False

    def test_initialization_with_name(self):
        """Test initialization with run name."""
        logger = WandbLogger(
            project="test_project",
            name="my_experiment",
            mode="disabled",
        )

        assert logger.enabled is False


class TestSetupWandb:
    """Test W&B setup function."""

    def test_setup_wandb_default(self):
        """Test setup with default parameters."""
        from training.utils.wandb_logger import setup_wandb

        # Use disabled mode for testing
        logger = setup_wandb(mode="disabled")

        assert logger is not None
        assert logger.enabled is False

    def test_setup_wandb_custom_project(self):
        """Test setup with custom project."""
        from training.utils.wandb_logger import setup_wandb

        logger = setup_wandb(
            project="custom_project",
            mode="disabled",
        )

        assert logger is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
