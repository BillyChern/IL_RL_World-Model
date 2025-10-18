"""Replay Buffer for RL Training.

This module provides replay buffers for storing and sampling transitions
during RL training. Supports:
- Standard experience replay
- Prioritized experience replay
- Episode-based sampling
- Human intervention tracking

Author: Billy Chern (Shichen)
License: MIT
"""

import random
from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch


@dataclass
class Transition:
    """Single transition in the replay buffer.

    Represents one timestep of interaction with the environment.
    """

    # Observations
    images: Dict[str, np.ndarray]  # {"left": [224,224,3], "right": [...], "base": [...]}
    state: np.ndarray  # [14] - joint positions

    # Action and reward
    action: np.ndarray  # [14] - executed action
    reward: float  # Scalar reward

    # Next observation
    next_images: Dict[str, np.ndarray]
    next_state: np.ndarray  # [14]

    # Episode info
    done: bool  # Episode termination flag
    language: str  # Task instruction

    # Additional metadata
    is_demo: bool = False  # True if from demonstration
    is_intervention: bool = False  # True if from human intervention
    episode_id: int = 0  # Episode identifier
    step_id: int = 0  # Step within episode


class ReplayBuffer:
    """Standard replay buffer for off-policy RL.

    Stores transitions and samples uniformly for training.
    Supports mixing demonstrations with online experience.

    Example:
        >>> buffer = ReplayBuffer(capacity=100000)
        >>> buffer.add(transition)
        >>> batch = buffer.sample(batch_size=256)
    """

    def __init__(
        self,
        capacity: int = 100000,
        demo_ratio: float = 0.5,
        intervention_bonus: float = 2.0,
        seed: Optional[int] = None,
    ):
        """Initialize replay buffer.

        Args:
            capacity: Maximum number of transitions to store
            demo_ratio: Ratio of demonstrations in each batch (0.0-1.0)
            intervention_bonus: Sampling weight multiplier for interventions
            seed: Random seed for reproducibility
        """
        self.capacity = capacity
        self.demo_ratio = demo_ratio
        self.intervention_bonus = intervention_bonus

        # Storage
        self.buffer: deque = deque(maxlen=capacity)
        self.demo_buffer: deque = deque(maxlen=capacity)  # Separate demo storage
        self.intervention_indices: List[int] = []  # Track interventions

        # Random state
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        # Statistics
        self.total_added = 0
        self.num_demos = 0
        self.num_interventions = 0

    def add(self, transition: Transition) -> None:
        """Add a transition to the buffer.

        Args:
            transition: Transition to add
        """
        if transition.is_demo:
            # Add to demo buffer
            self.demo_buffer.append(transition)
            self.num_demos += 1
        else:
            # Add to main buffer
            self.buffer.append(transition)

            # Track interventions
            if transition.is_intervention:
                self.intervention_indices.append(len(self.buffer) - 1)
                self.num_interventions += 1

        self.total_added += 1

    def add_batch(self, transitions: List[Transition]) -> None:
        """Add multiple transitions at once.

        Args:
            transitions: List of transitions
        """
        for transition in transitions:
            self.add(transition)

    def sample(self, batch_size: int) -> Dict[str, torch.Tensor]:
        """Sample a batch of transitions.

        Samples a mix of demonstrations and online experience based on demo_ratio.
        Gives higher weight to human interventions.

        Args:
            batch_size: Number of transitions to sample

        Returns:
            Batch dictionary with tensors
        """
        if len(self) < batch_size:
            raise ValueError(f"Not enough data: {len(self)} < {batch_size}")

        # Determine how many demos vs online samples
        # Adjust based on what's actually available
        available_demos = len(self.demo_buffer)
        available_online = len(self.buffer)

        if available_demos == 0:
            # No demos, sample all from online
            num_demos = 0
            num_online = batch_size
        elif available_online == 0:
            # No online data, sample all from demos
            num_demos = batch_size
            num_online = 0
        else:
            # Both available, use demo_ratio
            num_demos = min(int(batch_size * self.demo_ratio), available_demos)
            num_online = batch_size - num_demos
            # If not enough online, take more demos
            if num_online > available_online:
                num_online = available_online
                num_demos = batch_size - num_online

        # Sample demos
        demo_transitions = []
        if num_demos > 0:
            demo_transitions = random.sample(list(self.demo_buffer), num_demos)

        # Sample online experience (with intervention bias)
        online_transitions = []
        if num_online > 0:
            # Create sampling weights
            weights = np.ones(len(self.buffer))
            for idx in self.intervention_indices:
                if idx < len(self.buffer):
                    weights[idx] *= self.intervention_bonus

            # Normalize weights
            weights = weights / weights.sum()

            # Sample indices
            indices = np.random.choice(
                len(self.buffer),
                size=num_online,
                replace=False,
                p=weights,
            )

            online_transitions = [self.buffer[i] for i in indices]

        # Combine and convert to batch
        transitions = demo_transitions + online_transitions
        return self._transitions_to_batch(transitions)

    def sample_episode(self, episode_id: Optional[int] = None) -> List[Transition]:
        """Sample a complete episode from the buffer.

        Args:
            episode_id: Specific episode to sample (None for random)

        Returns:
            List of transitions for the episode
        """
        # Get all episode IDs
        all_transitions = list(self.buffer) + list(self.demo_buffer)
        episode_ids = list(set(t.episode_id for t in all_transitions))

        if not episode_ids:
            return []

        # Select episode
        if episode_id is None:
            episode_id = random.choice(episode_ids)

        # Get all transitions for this episode
        episode_transitions = [
            t for t in all_transitions
            if t.episode_id == episode_id
        ]

        # Sort by step_id
        episode_transitions.sort(key=lambda t: t.step_id)

        return episode_transitions

    def _transitions_to_batch(
        self, transitions: List[Transition]
    ) -> Dict[str, torch.Tensor]:
        """Convert list of transitions to batched tensors.

        Args:
            transitions: List of transitions

        Returns:
            Dictionary of batched tensors
        """
        if not transitions:
            raise ValueError("Empty transition list")

        # Batch images (each camera separately)
        images = {
            camera: torch.stack([
                torch.from_numpy(t.images[camera]).float() / 255.0
                for t in transitions
            ])
            for camera in transitions[0].images.keys()
        }

        next_images = {
            camera: torch.stack([
                torch.from_numpy(t.next_images[camera]).float() / 255.0
                for t in transitions
            ])
            for camera in transitions[0].next_images.keys()
        }

        # Batch other components
        states = torch.stack([
            torch.from_numpy(t.state).float() for t in transitions
        ])

        actions = torch.stack([
            torch.from_numpy(t.action).float() for t in transitions
        ])

        rewards = torch.tensor([t.reward for t in transitions], dtype=torch.float32)

        next_states = torch.stack([
            torch.from_numpy(t.next_state).float() for t in transitions
        ])

        dones = torch.tensor([float(t.done) for t in transitions], dtype=torch.float32)

        # Collect language instructions
        languages = [t.language for t in transitions]

        # Metadata
        is_demo = torch.tensor([float(t.is_demo) for t in transitions], dtype=torch.float32)
        is_intervention = torch.tensor(
            [float(t.is_intervention) for t in transitions],
            dtype=torch.float32
        )

        return {
            "images": images,
            "states": states,
            "actions": actions,
            "rewards": rewards,
            "next_images": next_images,
            "next_states": next_states,
            "dones": dones,
            "languages": languages,
            "is_demo": is_demo,
            "is_intervention": is_intervention,
        }

    def get_latest_episodes(self, num_episodes: int = 10) -> List[List[Transition]]:
        """Get the most recent N episodes.

        Args:
            num_episodes: Number of recent episodes to retrieve

        Returns:
            List of episode trajectories
        """
        # Get all transitions
        all_transitions = list(self.buffer)

        # Group by episode
        episodes_dict = {}
        for t in all_transitions:
            if t.episode_id not in episodes_dict:
                episodes_dict[t.episode_id] = []
            episodes_dict[t.episode_id].append(t)

        # Sort episodes by ID (assuming higher ID = more recent)
        sorted_episodes = sorted(episodes_dict.items(), key=lambda x: x[0], reverse=True)

        # Get latest N episodes
        latest_episodes = []
        for episode_id, transitions in sorted_episodes[:num_episodes]:
            # Sort transitions by step_id
            transitions.sort(key=lambda t: t.step_id)
            latest_episodes.append(transitions)

        return latest_episodes

    def __len__(self) -> int:
        """Get total number of transitions in buffer."""
        return len(self.buffer) + len(self.demo_buffer)

    def clear(self) -> None:
        """Clear all transitions from buffer."""
        self.buffer.clear()
        self.demo_buffer.clear()
        self.intervention_indices.clear()
        self.total_added = 0
        self.num_demos = 0
        self.num_interventions = 0

    def get_stats(self) -> Dict[str, int]:
        """Get buffer statistics.

        Returns:
            Dictionary with buffer statistics
        """
        return {
            "total_transitions": len(self),
            "online_transitions": len(self.buffer),
            "demo_transitions": len(self.demo_buffer),
            "interventions": self.num_interventions,
            "capacity": self.capacity,
            "utilization": len(self) / self.capacity,
        }


class EpisodeBuffer:
    """Episode-based replay buffer.

    Stores complete episodes and samples episode chunks for training.
    Useful for recurrent policies and world models.

    Example:
        >>> buffer = EpisodeBuffer(capacity=1000)
        >>> buffer.add_episode(episode_transitions)
        >>> episodes = buffer.sample_episodes(batch_size=32, chunk_length=50)
    """

    def __init__(self, capacity: int = 1000, seed: Optional[int] = None):
        """Initialize episode buffer.

        Args:
            capacity: Maximum number of episodes to store
            seed: Random seed for reproducibility
        """
        self.capacity = capacity
        self.episodes: deque = deque(maxlen=capacity)

        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

    def add_episode(self, episode: List[Transition]) -> None:
        """Add a complete episode to the buffer.

        Args:
            episode: List of transitions forming an episode
        """
        self.episodes.append(episode)

    def sample_episodes(
        self,
        batch_size: int,
        chunk_length: int = 50,
    ) -> List[List[Transition]]:
        """Sample episode chunks for training.

        Args:
            batch_size: Number of episode chunks to sample
            chunk_length: Length of each chunk

        Returns:
            List of episode chunks
        """
        if len(self.episodes) == 0:
            raise ValueError("Buffer is empty")

        chunks = []
        for _ in range(batch_size):
            # Sample random episode
            episode = random.choice(self.episodes)

            # Sample random start point
            if len(episode) <= chunk_length:
                # Episode shorter than chunk, pad with zeros
                chunk = episode
            else:
                start_idx = random.randint(0, len(episode) - chunk_length)
                chunk = episode[start_idx:start_idx + chunk_length]

            chunks.append(chunk)

        return chunks

    def __len__(self) -> int:
        """Get number of episodes in buffer."""
        return len(self.episodes)
