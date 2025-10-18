"""Reward Classifier for Learning from Demonstrations.

This module implements a binary classifier that learns to predict task success
from robot observations. It's trained on demonstration data to distinguish
successful task completions from failures.

The classifier is used to provide reward signals for RL training, following
the HIL-SERL approach.

Author: Billy Chern (Shichen)
License: MIT
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


@dataclass
class RewardClassifierConfig:
    """Configuration for reward classifier."""

    # Architecture
    vision_encoder: str = "resnet10"  # ResNet-10 as in HIL-SERL
    hidden_dim: int = 512
    num_cameras: int = 3

    # Input/output
    image_size: Tuple[int, int] = (224, 224)
    state_dim: int = 14  # Joint positions
    num_classes: int = 2  # Success vs. failure

    # Training
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    label_smoothing: float = 0.1

    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class ResNet10Encoder(nn.Module):
    """Simplified ResNet-10 vision encoder.

    Lightweight encoder for fast reward prediction during training.
    """

    def __init__(self, config: RewardClassifierConfig):
        super().__init__()
        self.config = config

        # Initial convolution
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Residual blocks (simplified ResNet-10)
        self.layer1 = self._make_layer(64, 64, blocks=1, stride=1)
        self.layer2 = self._make_layer(64, 128, blocks=1, stride=2)
        self.layer3 = self._make_layer(128, 256, blocks=1, stride=2)
        self.layer4 = self._make_layer(256, 512, blocks=1, stride=2)

        # Global average pooling
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # Output dimension
        self.output_dim = 512

    def _make_layer(
        self, in_channels: int, out_channels: int, blocks: int, stride: int
    ) -> nn.Sequential:
        """Create a residual layer."""
        layers = []

        # First block (with potential stride)
        layers.append(
            nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, stride, 1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, 3, 1, 1),
                nn.BatchNorm2d(out_channels),
            )
        )

        # Residual connection
        if stride != 1 or in_channels != out_channels:
            layers.insert(
                0,
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, 1, stride),
                    nn.BatchNorm2d(out_channels),
                ),
            )

        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode image to features.

        Args:
            x: Image tensor [B, 3, 224, 224]

        Returns:
            Features [B, 512]
        """
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        return x


class RewardClassifier(nn.Module):
    """Reward classifier for learning from demonstrations.

    Predicts binary success/failure from multi-camera observations and
    robot state (proprioception).

    Example:
        >>> config = RewardClassifierConfig()
        >>> classifier = RewardClassifier(config)
        >>> reward_prob = classifier.predict(images, state)
    """

    def __init__(self, config: RewardClassifierConfig):
        super().__init__()
        self.config = config

        # Vision encoder for each camera
        self.vision_encoder = ResNet10Encoder(config)

        # Fusion layer for multi-camera features
        self.camera_fusion = nn.Sequential(
            nn.Linear(
                config.num_cameras * self.vision_encoder.output_dim + config.state_dim,
                config.hidden_dim,
            ),
            nn.ReLU(),
            nn.Dropout(0.3),
        )

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(config.hidden_dim // 2, config.num_classes),
        )

        # Move to device
        self.to(config.device)

    def forward(
        self,
        images: Dict[str, torch.Tensor],
        state: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass through the classifier.

        Args:
            images: Dict with 'left', 'right', 'base' keys
                   Each value: [B, 3, 224, 224]
            state: Robot state [B, 14]

        Returns:
            Logits [B, 2] for success/failure classification
        """
        batch_size = state.shape[0]

        # Encode each camera view
        camera_features = []
        for camera in sorted(images.keys()):
            img = images[camera]  # [B, 3, 224, 224]
            feats = self.vision_encoder(img)  # [B, 512]
            camera_features.append(feats)

        # Concatenate all features
        visual_features = torch.cat(camera_features, dim=1)  # [B, num_cameras * 512]
        combined_features = torch.cat([visual_features, state], dim=1)  # [B, num_cameras*512 + 14]

        # Fuse features
        fused = self.camera_fusion(combined_features)  # [B, hidden_dim]

        # Classify
        logits = self.classifier(fused)  # [B, 2]

        return logits

    @torch.no_grad()
    def predict(
        self,
        images: Dict[str, np.ndarray],
        state: np.ndarray,
    ) -> float:
        """Predict reward probability for deployment.

        Args:
            images: Dict with 'left', 'right', 'base' keys
                   Each value: [224, 224, 3] numpy array (uint8 or float32)
            state: Robot state [14] numpy array

        Returns:
            Success probability (0.0 to 1.0)
        """
        self.eval()

        # Preprocess images
        images_tensor = {}
        for camera, img in images.items():
            # Convert to float
            if img.dtype == np.uint8:
                img = img.astype(np.float32) / 255.0

            # Transpose to [C, H, W]
            img_tensor = torch.from_numpy(img).permute(2, 0, 1)

            # Add batch dimension
            img_tensor = img_tensor.unsqueeze(0).to(self.config.device)
            images_tensor[camera] = img_tensor

        # Preprocess state
        state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(self.config.device)

        # Forward pass
        logits = self.forward(images_tensor, state_tensor)  # [1, 2]

        # Get probability of success
        probs = F.softmax(logits, dim=-1)  # [1, 2]
        success_prob = probs[0, 1].item()  # Probability of class 1 (success)

        return success_prob

    def predict_batch(
        self,
        images: Dict[str, torch.Tensor],
        states: torch.Tensor,
    ) -> torch.Tensor:
        """Predict rewards for a batch (used during training).

        Args:
            images: Dict with 'left', 'right', 'base' keys
                   Each value: [B, 3, 224, 224]
            states: Robot states [B, 14]

        Returns:
            Success probabilities [B]
        """
        self.eval()

        with torch.no_grad():
            logits = self.forward(images, states)  # [B, 2]
            probs = F.softmax(logits, dim=-1)  # [B, 2]
            success_probs = probs[:, 1]  # [B]

        return success_probs

    def compute_loss(
        self,
        images: Dict[str, torch.Tensor],
        states: torch.Tensor,
        labels: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute classification loss.

        Args:
            images: Dict with camera images [B, 3, 224, 224]
            states: Robot states [B, 14]
            labels: Binary labels [B] (0=failure, 1=success)

        Returns:
            loss: Loss scalar
            metrics: Dict with accuracy, precision, recall
        """
        # Forward pass
        logits = self.forward(images, states)  # [B, 2]

        # Compute cross-entropy loss with label smoothing
        loss = F.cross_entropy(
            logits,
            labels,
            label_smoothing=self.config.label_smoothing,
        )

        # Compute metrics
        with torch.no_grad():
            preds = logits.argmax(dim=-1)  # [B]
            accuracy = (preds == labels).float().mean().item()

            # Precision and recall for success class
            true_positives = ((preds == 1) & (labels == 1)).sum().item()
            false_positives = ((preds == 1) & (labels == 0)).sum().item()
            false_negatives = ((preds == 0) & (labels == 1)).sum().item()

            precision = (
                true_positives / (true_positives + false_positives)
                if (true_positives + false_positives) > 0
                else 0.0
            )
            recall = (
                true_positives / (true_positives + false_negatives)
                if (true_positives + false_negatives) > 0
                else 0.0
            )

        metrics = {
            "loss": loss.item(),
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
        }

        return loss, metrics

    def save_checkpoint(self, path: Union[str, Path]) -> None:
        """Save classifier checkpoint.

        Args:
            path: Path to save checkpoint
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            "config": self.config,
            "state_dict": self.state_dict(),
        }

        torch.save(checkpoint, path)
        print(f"✓ Reward classifier checkpoint saved to {path}")

    @classmethod
    def from_checkpoint(cls, path: Union[str, Path]) -> "RewardClassifier":
        """Load classifier from checkpoint.

        Args:
            path: Path to checkpoint

        Returns:
            Loaded RewardClassifier instance
        """
        checkpoint = torch.load(path, map_location="cpu")
        config = checkpoint["config"]

        # Create classifier
        classifier = cls(config)

        # Load weights
        classifier.load_state_dict(checkpoint["state_dict"])

        print(f"✓ Reward classifier loaded from {path}")
        return classifier
