"""RLDS Dataset for Robot Demonstrations.

This module provides utilities for loading and processing robot demonstration
data in RLDS (Robotic Learning Dataset Standard) format using LeRobot.

The dataset structure follows LeRobot's conventions:
- Parquet files for metadata and state/action data
- MP4 files for camera observations
- JSON metadata for dataset info

Author: Billy Chern (Shichen)
License: MIT
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from datasets import Dataset, load_dataset
from torch.utils.data import DataLoader


class RLDSDataset:
    """RLDS dataset loader for robot demonstrations.

    Loads demonstration data in LeRobot RLDS format and provides
    PyTorch-compatible data loading for training.

    Example:
        >>> dataset = RLDSDataset(
        ...     repo_id="lerobot/arx_x5_pick_place",
        ...     root="/data/demonstrations"
        ... )
        >>> batch = dataset[0]
        >>> # batch contains: images, proprioception, actions, language
    """

    def __init__(
        self,
        repo_id: Optional[str] = None,
        root: Optional[Union[str, Path]] = None,
        split: str = "train",
        delta_timestamps: Optional[Dict[str, List[float]]] = None,
        image_transforms: Optional[callable] = None,
        verbose: bool = True,
    ):
        """Initialize RLDS dataset.

        Args:
            repo_id: HuggingFace dataset repository ID (e.g., "lerobot/arx_x5_pick_place")
            root: Local path to dataset directory (if not using HF)
            split: Dataset split ("train", "val", "test")
            delta_timestamps: Temporal offsets for observations/actions
                Example: {"observation.image": [-0.1, 0.0], "action": [0.0, 0.02]}
            image_transforms: Optional transform function for images
            verbose: Print loading information
        """
        self.repo_id = repo_id
        self.root = Path(root) if root else None
        self.split = split
        self.verbose = verbose
        self.image_transforms = image_transforms

        # Default delta timestamps (current frame only)
        self.delta_timestamps = delta_timestamps or {
            "observation.images.left": [0.0],
            "observation.images.right": [0.0],
            "observation.images.base": [0.0],
            "observation.state": [0.0],
            "action": [0.0],
        }

        # Load dataset
        self.dataset = self._load_dataset()

        # Get dataset info
        self.num_episodes = self._get_num_episodes()
        self.episode_lengths = self._get_episode_lengths()

        if self.verbose:
            print(f"✓ Loaded {len(self.dataset)} transitions from {self.num_episodes} episodes")
            print(f"  Average episode length: {np.mean(self.episode_lengths):.1f} steps")

    def _load_dataset(self) -> Dataset:
        """Load dataset from HuggingFace or local path."""
        if self.repo_id:
            # Load from HuggingFace
            if self.verbose:
                print(f"Loading dataset from HuggingFace: {self.repo_id}")

            dataset = load_dataset(
                self.repo_id,
                split=self.split,
                trust_remote_code=True,
            )
        elif self.root:
            # Load from local path
            if self.verbose:
                print(f"Loading dataset from local path: {self.root}")

            # LeRobot datasets are stored as Parquet + MP4
            dataset = load_dataset(
                "parquet",
                data_dir=str(self.root),
                split=self.split,
            )
        else:
            raise ValueError("Either repo_id or root must be provided")

        return dataset

    def _get_num_episodes(self) -> int:
        """Get number of episodes in dataset."""
        if "episode_index" in self.dataset.column_names:
            return len(set(self.dataset["episode_index"]))
        else:
            # Fallback: assume each trajectory is one episode
            return len(self.dataset)

    def _get_episode_lengths(self) -> List[int]:
        """Get length of each episode."""
        if "episode_index" in self.dataset.column_names:
            episode_indices = self.dataset["episode_index"]
            unique_episodes = sorted(set(episode_indices))
            return [episode_indices.count(ep) for ep in unique_episodes]
        else:
            # Fallback: assume all episodes have same length
            return [len(self.dataset)]

    def __len__(self) -> int:
        """Get number of transitions in dataset."""
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single transition from dataset.

        Args:
            idx: Index of transition

        Returns:
            Dictionary containing:
                - images: Dict[str, Tensor] - {"left": [3,224,224], "right": [...], "base": [...]}
                - state: Tensor [14] - joint positions
                - action: Tensor [14] - target joint positions
                - language: str - instruction text
                - reward: Tensor [1] - reward signal (if available)
                - done: Tensor [1] - episode termination flag
        """
        item = self.dataset[idx]

        # Extract images (3 cameras)
        images = {}
        for camera in ["left", "right", "base"]:
            key = f"observation.images.{camera}"
            if key in item:
                img = item[key]

                # Convert to numpy if needed
                if not isinstance(img, np.ndarray):
                    img = np.array(img)

                # Apply transforms if provided
                if self.image_transforms:
                    img = self.image_transforms(img)
                else:
                    # Default: convert to float32 and normalize to [0, 1]
                    img = img.astype(np.float32) / 255.0
                    # Transpose from HWC to CHW for PyTorch
                    img = np.transpose(img, (2, 0, 1))

                images[camera] = torch.from_numpy(img)

        # Extract proprioceptive state (joint positions)
        state = item.get("observation.state", np.zeros(14))
        state = torch.from_numpy(np.array(state, dtype=np.float32))

        # Extract action
        action = item.get("action", np.zeros(14))
        action = torch.from_numpy(np.array(action, dtype=np.float32))

        # Extract language instruction
        language = item.get("language_instruction", "")
        if not isinstance(language, str):
            language = str(language)

        # Extract reward (if available)
        reward = item.get("reward", 0.0)
        reward = torch.tensor([reward], dtype=torch.float32)

        # Extract done flag
        done = item.get("done", False)
        done = torch.tensor([float(done)], dtype=torch.float32)

        return {
            "images": images,
            "state": state,
            "action": action,
            "language": language,
            "reward": reward,
            "done": done,
            "episode_index": item.get("episode_index", 0),
            "frame_index": item.get("frame_index", idx),
        }

    def get_episode(self, episode_idx: int) -> List[Dict[str, torch.Tensor]]:
        """Get all transitions from a specific episode.

        Args:
            episode_idx: Index of episode

        Returns:
            List of transitions for the episode
        """
        # Find all indices for this episode
        if "episode_index" in self.dataset.column_names:
            episode_indices = [
                i for i, ep in enumerate(self.dataset["episode_index"])
                if ep == episode_idx
            ]
        else:
            raise ValueError("Dataset does not have episode_index column")

        return [self[i] for i in episode_indices]

    def get_dataloader(
        self,
        batch_size: int = 32,
        shuffle: bool = True,
        num_workers: int = 4,
        **kwargs,
    ) -> DataLoader:
        """Create PyTorch DataLoader for this dataset.

        Args:
            batch_size: Batch size
            shuffle: Shuffle data
            num_workers: Number of data loading workers
            **kwargs: Additional arguments for DataLoader

        Returns:
            PyTorch DataLoader
        """
        return DataLoader(
            self,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=self._collate_fn,
            **kwargs,
        )

    @staticmethod
    def _collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
        """Collate function for DataLoader.

        Handles batching of images, states, actions, and language instructions.
        """
        # Batch images (each camera separately)
        images = {
            camera: torch.stack([item["images"][camera] for item in batch])
            for camera in batch[0]["images"].keys()
        }

        # Batch other tensors
        states = torch.stack([item["state"] for item in batch])
        actions = torch.stack([item["action"] for item in batch])
        rewards = torch.stack([item["reward"] for item in batch])
        dones = torch.stack([item["done"] for item in batch])

        # Collect language instructions (as list of strings)
        languages = [item["language"] for item in batch]

        # Collect episode and frame indices
        episode_indices = torch.tensor([item["episode_index"] for item in batch])
        frame_indices = torch.tensor([item["frame_index"] for item in batch])

        return {
            "images": images,
            "states": states,
            "actions": actions,
            "languages": languages,
            "rewards": rewards,
            "dones": dones,
            "episode_indices": episode_indices,
            "frame_indices": frame_indices,
        }

    def get_statistics(self) -> Dict[str, Dict[str, float]]:
        """Compute dataset statistics for normalization.

        Returns:
            Dictionary with mean/std for states and actions
        """
        if self.verbose:
            print("Computing dataset statistics...")

        # Collect all states and actions
        all_states = []
        all_actions = []

        for i in range(len(self)):
            item = self[i]
            all_states.append(item["state"].numpy())
            all_actions.append(item["action"].numpy())

        states = np.stack(all_states)
        actions = np.stack(all_actions)

        stats = {
            "state": {
                "mean": states.mean(axis=0).tolist(),
                "std": states.std(axis=0).tolist(),
                "min": states.min(axis=0).tolist(),
                "max": states.max(axis=0).tolist(),
            },
            "action": {
                "mean": actions.mean(axis=0).tolist(),
                "std": actions.std(axis=0).tolist(),
                "min": actions.min(axis=0).tolist(),
                "max": actions.max(axis=0).tolist(),
            },
        }

        if self.verbose:
            print("✓ Dataset statistics computed")
            print(f"  State range: [{states.min():.3f}, {states.max():.3f}]")
            print(f"  Action range: [{actions.min():.3f}, {actions.max():.3f}]")

        return stats

    def save_statistics(self, path: Union[str, Path]) -> None:
        """Save dataset statistics to JSON file.

        Args:
            path: Path to save statistics
        """
        stats = self.get_statistics()

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w") as f:
            json.dump(stats, f, indent=2)

        if self.verbose:
            print(f"✓ Statistics saved to {path}")


def create_demonstration_dataset(
    demonstrations_path: Union[str, Path],
    output_path: Union[str, Path],
    metadata: Optional[Dict] = None,
    verbose: bool = True,
) -> None:
    """Create RLDS dataset from raw demonstration files.

    This function converts raw demonstration data (e.g., from teleoperation)
    into LeRobot RLDS format for training.

    Args:
        demonstrations_path: Path to raw demonstration files
        output_path: Path to save RLDS dataset
        metadata: Optional metadata dict to include
        verbose: Print progress information
    """
    # TODO: Implement conversion from raw demos to RLDS format
    # This will be needed when collecting new demonstrations
    raise NotImplementedError("Demo to RLDS conversion not yet implemented")
