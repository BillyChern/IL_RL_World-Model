"""Data pipeline for robot demonstrations and RL training.

This module provides utilities for:
- Loading RLDS demonstration datasets
- Replay buffers for RL training
- Data preprocessing and augmentation

Author: Billy Chern (Shichen)
License: MIT
"""

from data.replay_buffer import EpisodeBuffer, ReplayBuffer, Transition

# Optional imports (require additional dependencies)
try:
    from data.rlds_dataset import RLDSDataset, create_demonstration_dataset

    __all__ = [
        "RLDSDataset",
        "create_demonstration_dataset",
        "ReplayBuffer",
        "EpisodeBuffer",
        "Transition",
    ]
except ImportError:
    __all__ = [
        "ReplayBuffer",
        "EpisodeBuffer",
        "Transition",
    ]
