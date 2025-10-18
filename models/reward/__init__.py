"""Reward prediction module.

This module provides learned reward functions for RL training.

Author: Billy Chern (Shichen)
License: MIT
"""

from models.reward.reward_classifier import RewardClassifier, RewardClassifierConfig

__all__ = [
    "RewardClassifier",
    "RewardClassifierConfig",
]
