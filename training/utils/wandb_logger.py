"""Weights & Biases logging utilities.

Provides structured logging for training metrics, videos, and system stats.

Author: Billy Chern (Shichen)
License: MIT
"""

import os
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import wandb


class WandbLogger:
    """W&B logger for training experiments.

    Example:
        >>> logger = WandbLogger(project="il_rl", api_key="...")
        >>> logger.log({"loss": 0.5}, step=100)
        >>> logger.log_video(frames, "rollout", step=100)
    """

    def __init__(
        self,
        project: str,
        entity: Optional[str] = None,
        name: Optional[str] = None,
        config: Optional[Dict] = None,
        api_key: Optional[str] = None,
        mode: str = "online",  # "online", "offline", or "disabled"
    ):
        """Initialize W&B logger.

        Args:
            project: W&B project name
            entity: W&B entity (team/user)
            name: Run name
            config: Configuration dictionary to log
            api_key: W&B API key (or set WANDB_API_KEY env var)
            mode: Logging mode
        """
        self.enabled = (mode != "disabled")

        if not self.enabled:
            print("W&B logging disabled")
            return

        # Set API key if provided
        if api_key:
            os.environ["WANDB_API_KEY"] = api_key
            print("✓ W&B API key configured")

        # Initialize W&B run
        self.run = wandb.init(
            project=project,
            entity=entity,
            name=name,
            config=config or {},
            mode=mode,
        )

        print(f"✓ W&B initialized: {self.run.url}")

    def log(self, metrics: Dict[str, Any], step: Optional[int] = None) -> None:
        """Log metrics to W&B.

        Args:
            metrics: Dictionary of metrics
            step: Global step (optional)
        """
        if not self.enabled:
            return

        wandb.log(metrics, step=step)

    def log_video(
        self,
        frames: np.ndarray,
        name: str,
        step: Optional[int] = None,
        fps: int = 10,
    ) -> None:
        """Log video to W&B.

        Args:
            frames: Video frames [T, H, W, C] (uint8)
            name: Video name
            step: Global step
            fps: Frames per second
        """
        if not self.enabled:
            return

        video = wandb.Video(frames, fps=fps, format="mp4")
        wandb.log({name: video}, step=step)

    def log_image(
        self,
        image: np.ndarray,
        name: str,
        step: Optional[int] = None,
        caption: Optional[str] = None,
    ) -> None:
        """Log image to W&B.

        Args:
            image: Image array [H, W, C]
            name: Image name
            step: Global step
            caption: Image caption
        """
        if not self.enabled:
            return

        img = wandb.Image(image, caption=caption)
        wandb.log({name: img}, step=step)

    def log_histogram(
        self,
        values: np.ndarray,
        name: str,
        step: Optional[int] = None,
    ) -> None:
        """Log histogram to W&B.

        Args:
            values: Value array
            name: Histogram name
            step: Global step
        """
        if not self.enabled:
            return

        wandb.log({name: wandb.Histogram(values)}, step=step)

    def save_artifact(
        self,
        file_path: Path,
        artifact_name: str,
        artifact_type: str = "model",
    ) -> None:
        """Save file as W&B artifact.

        Args:
            file_path: Path to file
            artifact_name: Artifact name
            artifact_type: Artifact type (model, dataset, etc.)
        """
        if not self.enabled:
            return

        artifact = wandb.Artifact(artifact_name, type=artifact_type)
        artifact.add_file(str(file_path))
        self.run.log_artifact(artifact)

    def finish(self) -> None:
        """Finish W&B run."""
        if not self.enabled:
            return

        wandb.finish()
        print("✓ W&B run finished")


def setup_wandb(
    api_key: str = "aeb213781951708e4ee8bdfa307cb9c3b53a2000",
    project: str = "il_rl_world_model",
    **kwargs,
) -> WandbLogger:
    """Setup W&B logger with default API key.

    Args:
        api_key: W&B API key
        project: Project name
        **kwargs: Additional arguments for WandbLogger

    Returns:
        Configured WandbLogger
    """
    return WandbLogger(
        project=project,
        api_key=api_key,
        **kwargs,
    )
