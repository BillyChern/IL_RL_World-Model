"""π0.5 (pi0) Vision-Language-Action Policy.

This module provides a wrapper for Physical Intelligence's π0.5 model,
adapted for the ARX X5 dual-arm robot.

The model architecture:
- Vision Encoder: SigLIP ViT-SO400M (pre-trained)
- Language Encoder: Gemma 2B (pre-trained)
- Action Decoder: Diffusion-based (Gemma 300M action expert)
- Action Space: 14D (dual 6-DOF arms + grippers)

Based on: https://github.com/physical-intelligence/openpi

Author: Billy Chern (Shichen)
License: MIT
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from einops import rearrange
from transformers import AutoModel, AutoTokenizer


@dataclass
class Pi0Config:
    """Configuration for π0.5 model.

    Based on the official π0.5 configuration for robotic manipulation.
    """

    # Model architecture
    vision_encoder: str = "google/siglip-so400m-patch14-384"
    language_model: str = "google/gemma-2b"
    action_expert: str = "pi0-action-expert-300m"  # Gemma-based action decoder

    # Action space
    action_dim: int = 14  # 2 arms × (6 joints + 1 gripper)
    chunk_size: int = 50  # Action chunking horizon

    # Diffusion parameters
    num_diffusion_steps: int = 10
    noise_scheduler: str = "DDPM"

    # Vision parameters
    image_size: Tuple[int, int] = (224, 224)
    num_cameras: int = 3  # left, right, base

    # Training parameters
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0

    # Pretrained weights
    pretrained_path: Optional[str] = None  # Path to pretrained checkpoint
    freeze_vision: bool = True  # Freeze vision encoder
    freeze_language: bool = True  # Freeze language encoder

    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class VisionEncoder(nn.Module):
    """Vision encoder using SigLIP ViT-SO400M.

    Encodes multiple camera views into visual features.
    """

    def __init__(self, config: Pi0Config):
        super().__init__()
        self.config = config

        # Load SigLIP vision encoder
        self.encoder = AutoModel.from_pretrained(
            config.vision_encoder,
            trust_remote_code=True,
        )

        # Get embedding dimension
        self.embed_dim = self.encoder.config.hidden_size

        # Freeze if specified
        if config.freeze_vision:
            for param in self.encoder.parameters():
                param.requires_grad = False

    def forward(self, images: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Encode images from multiple cameras.

        Args:
            images: Dict with keys 'left', 'right', 'base'
                   Each value: [B, 3, 224, 224] tensor

        Returns:
            Visual features: [B, num_cameras * num_patches, embed_dim]
        """
        batch_size = images["left"].shape[0]
        camera_names = sorted(images.keys())  # Ensure consistent ordering

        # Encode each camera view
        features_list = []
        for camera in camera_names:
            img = images[camera]  # [B, 3, 224, 224]

            # Get image embeddings
            outputs = self.encoder.vision_model(pixel_values=img)
            feats = outputs.last_hidden_state  # [B, num_patches, embed_dim]

            features_list.append(feats)

        # Concatenate features from all cameras
        visual_features = torch.cat(features_list, dim=1)  # [B, 3*num_patches, embed_dim]

        return visual_features


class LanguageEncoder(nn.Module):
    """Language encoder using Gemma 2B.

    Encodes task instructions into language embeddings.
    """

    def __init__(self, config: Pi0Config):
        super().__init__()
        self.config = config

        # Load Gemma 2B
        self.tokenizer = AutoTokenizer.from_pretrained(
            config.language_model,
            trust_remote_code=True,
        )

        self.encoder = AutoModel.from_pretrained(
            config.language_model,
            trust_remote_code=True,
        )

        # Get embedding dimension
        self.embed_dim = self.encoder.config.hidden_size

        # Freeze if specified
        if config.freeze_language:
            for param in self.encoder.parameters():
                param.requires_grad = False

    def forward(self, instructions: List[str]) -> torch.Tensor:
        """Encode language instructions.

        Args:
            instructions: List of instruction strings

        Returns:
            Language features: [B, seq_len, embed_dim]
        """
        # Tokenize
        tokens = self.tokenizer(
            instructions,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        ).to(self.encoder.device)

        # Encode
        outputs = self.encoder(**tokens)
        language_features = outputs.last_hidden_state  # [B, seq_len, embed_dim]

        return language_features


class DiffusionActionDecoder(nn.Module):
    """Diffusion-based action decoder.

    Predicts action sequences using iterative denoising.
    Based on π0.5's Gemma 300M action expert.
    """

    def __init__(self, config: Pi0Config, visual_dim: int, language_dim: int):
        super().__init__()
        self.config = config
        self.action_dim = config.action_dim
        self.chunk_size = config.chunk_size
        self.num_steps = config.num_diffusion_steps

        # Compute conditioning dimension
        self.cond_dim = visual_dim + language_dim

        # Action prediction network (simplified Gemma-style)
        hidden_dim = 1024
        self.net = nn.Sequential(
            # Input: noisy action + timestep + conditioning
            nn.Linear(action_dim * chunk_size + 1 + self.cond_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim * chunk_size),
        )

        # Timestep embedding
        self.time_embed = nn.Sequential(
            nn.Linear(1, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, 1),
        )

    def forward(
        self,
        visual_features: torch.Tensor,
        language_features: torch.Tensor,
        actions: Optional[torch.Tensor] = None,
        num_samples: int = 1,
    ) -> torch.Tensor:
        """Generate action predictions using diffusion.

        Args:
            visual_features: [B, num_patches, visual_dim]
            language_features: [B, seq_len, language_dim]
            actions: [B, chunk_size, action_dim] ground truth (for training)
            num_samples: Number of diffusion samples to generate

        Returns:
            Predicted actions: [B, chunk_size, action_dim]
        """
        batch_size = visual_features.shape[0]
        device = visual_features.device

        # Pool visual and language features
        visual_pool = visual_features.mean(dim=1)  # [B, visual_dim]
        language_pool = language_features.mean(dim=1)  # [B, language_dim]

        # Concatenate conditioning
        conditioning = torch.cat([visual_pool, language_pool], dim=-1)  # [B, cond_dim]

        if self.training and actions is not None:
            # Training: add noise and predict
            # Flatten actions
            actions_flat = rearrange(actions, "b t d -> b (t d)")  # [B, chunk_size*action_dim]

            # Sample random timestep
            t = torch.randint(0, self.num_steps, (batch_size,), device=device).float()
            t_norm = t / self.num_steps  # Normalize to [0, 1]

            # Add noise
            noise = torch.randn_like(actions_flat)
            alpha = 1.0 - t_norm.unsqueeze(1)
            noisy_actions = alpha * actions_flat + (1 - alpha) * noise

            # Embed timestep
            t_embed = self.time_embed(t_norm.unsqueeze(1))  # [B, 1]

            # Predict noise
            net_input = torch.cat([noisy_actions, t_embed, conditioning], dim=-1)
            pred_noise = self.net(net_input)

            # Return predicted noise for loss computation
            return pred_noise, noise

        else:
            # Inference: iterative denoising
            # Start from random noise
            actions_flat = torch.randn(
                batch_size, self.chunk_size * self.action_dim, device=device
            )

            # Iterative denoising
            for step in reversed(range(self.num_steps)):
                t = torch.full((batch_size,), step, device=device).float()
                t_norm = t / self.num_steps

                # Embed timestep
                t_embed = self.time_embed(t_norm.unsqueeze(1))

                # Predict noise
                net_input = torch.cat([actions_flat, t_embed, conditioning], dim=-1)
                pred_noise = self.net(net_input)

                # Denoise
                alpha = 1.0 - t_norm.unsqueeze(1)
                actions_flat = (actions_flat - (1 - alpha) * pred_noise) / alpha

                # Add noise for next step (except last)
                if step > 0:
                    noise_scale = 0.1
                    actions_flat = actions_flat + noise_scale * torch.randn_like(actions_flat)

            # Reshape to action sequence
            actions = rearrange(
                actions_flat, "b (t d) -> b t d", t=self.chunk_size, d=self.action_dim
            )

            return actions


class Pi0Policy(nn.Module):
    """π0.5 Vision-Language-Action Policy.

    Complete VLA model for robotic manipulation with the ARX X5 robot.

    Example:
        >>> config = Pi0Config(action_dim=14)
        >>> policy = Pi0Policy(config)
        >>> actions = policy.predict(images, instructions)
    """

    def __init__(self, config: Pi0Config):
        super().__init__()
        self.config = config

        # Vision encoder
        self.vision_encoder = VisionEncoder(config)

        # Language encoder
        self.language_encoder = LanguageEncoder(config)

        # Action decoder
        self.action_decoder = DiffusionActionDecoder(
            config,
            visual_dim=self.vision_encoder.embed_dim,
            language_dim=self.language_encoder.embed_dim,
        )

        # Move to device
        self.to(config.device)

    def forward(
        self,
        images: Dict[str, torch.Tensor],
        instructions: List[str],
        actions: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Forward pass through the policy.

        Args:
            images: Dict with 'left', 'right', 'base' keys
                   Each value: [B, 3, 224, 224]
            instructions: List of instruction strings
            actions: [B, chunk_size, action_dim] (for training)

        Returns:
            If training: (pred_noise, true_noise) for loss computation
            If inference: predicted actions [B, chunk_size, action_dim]
        """
        # Encode vision
        visual_features = self.vision_encoder(images)

        # Encode language
        language_features = self.language_encoder(instructions)

        # Decode actions
        if self.training and actions is not None:
            # Training mode
            pred_noise, true_noise = self.action_decoder(
                visual_features, language_features, actions
            )
            return pred_noise, true_noise
        else:
            # Inference mode
            actions = self.action_decoder(visual_features, language_features)
            return actions

    @torch.no_grad()
    def predict(
        self,
        images: Dict[str, np.ndarray],
        instruction: str,
    ) -> np.ndarray:
        """Predict actions for deployment.

        Args:
            images: Dict with 'left', 'right', 'base' keys
                   Each value: [224, 224, 3] numpy array (uint8)
            instruction: Task instruction string

        Returns:
            Predicted actions: [chunk_size, action_dim] numpy array
        """
        self.eval()

        # Preprocess images
        images_tensor = {}
        for camera, img in images.items():
            # Convert to float and normalize
            img_tensor = torch.from_numpy(img).float() / 255.0
            # Transpose to [C, H, W]
            img_tensor = img_tensor.permute(2, 0, 1)
            # Add batch dimension
            img_tensor = img_tensor.unsqueeze(0).to(self.config.device)
            images_tensor[camera] = img_tensor

        # Predict
        actions = self.forward(images_tensor, [instruction])  # [1, chunk_size, action_dim]

        # Convert to numpy
        actions_np = actions[0].cpu().numpy()  # [chunk_size, action_dim]

        return actions_np

    def save_checkpoint(self, path: Union[str, Path]) -> None:
        """Save model checkpoint.

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
        print(f"✓ Checkpoint saved to {path}")

    @classmethod
    def from_checkpoint(cls, path: Union[str, Path]) -> "Pi0Policy":
        """Load model from checkpoint.

        Args:
            path: Path to checkpoint

        Returns:
            Loaded Pi0Policy instance
        """
        checkpoint = torch.load(path, map_location="cpu")
        config = checkpoint["config"]

        # Create model
        model = cls(config)

        # Load weights
        model.load_state_dict(checkpoint["state_dict"])

        print(f"✓ Checkpoint loaded from {path}")
        return model

    @classmethod
    def from_pretrained(
        cls,
        checkpoint_path: Optional[str] = None,
        config: Optional[Pi0Config] = None,
    ) -> "Pi0Policy":
        """Load pretrained π0.5 model.

        Args:
            checkpoint_path: Path to pretrained weights
                           If None, initializes with pretrained vision/language only
            config: Model configuration

        Returns:
            Pi0Policy instance with pretrained weights
        """
        if config is None:
            config = Pi0Config()

        # Create model
        model = cls(config)

        # Load checkpoint if provided
        if checkpoint_path is not None:
            checkpoint = torch.load(checkpoint_path, map_location=config.device)

            # Load state dict (handle partial loading)
            model_dict = model.state_dict()
            pretrained_dict = {
                k: v for k, v in checkpoint["state_dict"].items() if k in model_dict
            }
            model_dict.update(pretrained_dict)
            model.load_state_dict(model_dict)

            print(f"✓ Pretrained weights loaded from {checkpoint_path}")
        else:
            print("✓ Initialized with pretrained vision and language encoders")

        return model


def compute_diffusion_loss(pred_noise: torch.Tensor, true_noise: torch.Tensor) -> torch.Tensor:
    """Compute diffusion loss (MSE between predicted and true noise).

    Args:
        pred_noise: Predicted noise [B, chunk_size * action_dim]
        true_noise: True noise [B, chunk_size * action_dim]

    Returns:
        Loss scalar
    """
    return nn.functional.mse_loss(pred_noise, true_noise)
