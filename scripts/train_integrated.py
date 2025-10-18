#!/usr/bin/env python3
"""Main training script for IL+RL+World Model system.

Integrates all components:
- π0.5 VLA policy
- DreamerV3 world model
- SAC RL algorithm
- HIL-SERL with human interventions
- Asynchronous training (robot + GPU)
- Multi-GPU distributed training
- W&B logging

Usage:
    python scripts/train_integrated.py --config configs/default.yaml

Author: Billy Chern (Shichen)
License: MIT
"""

import argparse
import sys
from pathlib import Path

import torch
import jax

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.vla import Pi0Policy, Pi0Config
from models.world_model import DreamerV3, DreamerConfig
from models.reward import RewardClassifier, RewardClassifierConfig
from training.async_trainer import AsyncTrainer, AsyncTrainerConfig
from training.utils import setup_wandb, setup_8gpu_training


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train IL+RL+World Model system"
    )

    # Model checkpoints
    parser.add_argument(
        "--vla_checkpoint",
        type=str,
        default=None,
        help="Path to pretrained π0.5 checkpoint"
    )
    parser.add_argument(
        "--world_model_checkpoint",
        type=str,
        default=None,
        help="Path to pretrained world model checkpoint"
    )
    parser.add_argument(
        "--reward_checkpoint",
        type=str,
        default=None,
        help="Path to trained reward classifier checkpoint"
    )

    # Training
    parser.add_argument(
        "--num_episodes",
        type=int,
        default=1000,
        help="Number of episodes to train"
    )
    parser.add_argument(
        "--max_episode_steps",
        type=int,
        default=500,
        help="Maximum steps per episode"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=256,
        help="Training batch size"
    )

    # Robot
    parser.add_argument(
        "--robot_left_port",
        type=str,
        default="/dev/ttyUSB0",
        help="Left arm serial port"
    )
    parser.add_argument(
        "--robot_right_port",
        type=str,
        default="/dev/ttyUSB1",
        help="Right arm serial port"
    )

    # Multi-GPU
    parser.add_argument(
        "--num_gpus",
        type=int,
        default=8,
        help="Number of GPUs to use"
    )
    parser.add_argument(
        "--use_ddp",
        action="store_true",
        help="Use DistributedDataParallel"
    )

    # W&B
    parser.add_argument(
        "--wandb_project",
        type=str,
        default="il_rl_world_model",
        help="W&B project name"
    )
    parser.add_argument(
        "--wandb_run_name",
        type=str,
        default=None,
        help="W&B run name"
    )
    parser.add_argument(
        "--no_wandb",
        action="store_true",
        help="Disable W&B logging"
    )

    # Checkpointing
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="checkpoints",
        help="Directory to save checkpoints"
    )
    parser.add_argument(
        "--checkpoint_freq",
        type=int,
        default=50,
        help="Save checkpoint every N episodes"
    )

    # Intervention
    parser.add_argument(
        "--enable_interventions",
        action="store_true",
        default=True,
        help="Enable human interventions"
    )

    # Imagination
    parser.add_argument(
        "--use_imagination",
        action="store_true",
        default=True,
        help="Use world model imagination"
    )
    parser.add_argument(
        "--imagination_horizon",
        type=int,
        default=10,
        help="Imagination rollout horizon"
    )

    return parser.parse_args()


def main():
    """Main training function."""
    args = parse_args()

    print("=" * 80)
    print("IL + RL + World Model Training")
    print("=" * 80)
    print(f"Episodes: {args.num_episodes}")
    print(f"GPUs: {args.num_gpus}")
    print(f"Batch size: {args.batch_size}")
    print(f"Interventions: {args.enable_interventions}")
    print(f"Imagination: {args.use_imagination}")
    print("=" * 80)

    # Setup distributed training
    distributed_config = None
    if args.num_gpus > 1 and args.use_ddp:
        print("\n[Setup] Initializing multi-GPU training...")
        distributed_config = setup_8gpu_training(use_ddp=True)
    else:
        print("\n[Setup] Using single-GPU training")
        setup_8gpu_training(use_ddp=False)

    # Setup W&B logging
    wandb_logger = None
    if not args.no_wandb:
        print("\n[Setup] Initializing W&B logging...")
        wandb_logger = setup_wandb(
            project=args.wandb_project,
            name=args.wandb_run_name,
            config=vars(args),
        )

    # Initialize models
    print("\n[Setup] Initializing models...")

    # VLA Policy (π0.5)
    vla_config = Pi0Config(
        action_dim=14,
        chunk_size=50,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

    if args.vla_checkpoint:
        print(f"Loading VLA from checkpoint: {args.vla_checkpoint}")
        vla_policy = Pi0Policy.from_checkpoint(args.vla_checkpoint)
    else:
        print("Initializing VLA from pretrained weights")
        vla_policy = Pi0Policy.from_pretrained(config=vla_config)

    # World Model (DreamerV3)
    world_model = None
    if args.use_imagination:
        print("Initializing DreamerV3 world model...")
        world_model_config = DreamerConfig(
            deter_dim=4096,
            stoch_dim=32,
            action_dim=14,
            imagination_horizon=args.imagination_horizon,
        )

        # TODO: Load pretrained world model if checkpoint provided
        # world_model = DreamerV3(config=world_model_config)

    # Reward Classifier
    reward_fn = None
    if args.reward_checkpoint:
        print(f"Loading reward classifier: {args.reward_checkpoint}")
        reward_fn = RewardClassifier.from_checkpoint(args.reward_checkpoint)
    else:
        print("Initializing reward classifier...")
        reward_config = RewardClassifierConfig()
        reward_fn = RewardClassifier(reward_config)

    # Create trainer configuration
    trainer_config = AsyncTrainerConfig(
        num_episodes=args.num_episodes,
        max_episode_steps=args.max_episode_steps,
        batch_size=args.batch_size,
        robot_left_port=args.robot_left_port,
        robot_right_port=args.robot_right_port,
        enable_interventions=args.enable_interventions,
        use_imagination=args.use_imagination,
        imagination_horizon=args.imagination_horizon,
        checkpoint_dir=args.checkpoint_dir,
        checkpoint_freq=args.checkpoint_freq,
        use_wandb=(not args.no_wandb),
        wandb_project=args.wandb_project,
        wandb_run_name=args.wandb_run_name,
    )

    # Create trainer
    print("\n[Setup] Creating async trainer...")
    trainer = AsyncTrainer(
        config=trainer_config,
        policy=vla_policy,
        world_model=world_model,
        reward_fn=reward_fn,
    )

    # Start training
    print("\n" + "=" * 80)
    print("Starting Training")
    print("=" * 80 + "\n")

    try:
        trainer.train()
    except KeyboardInterrupt:
        print("\n⚠ Training interrupted by user")
    except Exception as e:
        print(f"\n❌ Training failed with error: {e}")
        raise
    finally:
        # Cleanup
        if wandb_logger:
            wandb_logger.finish()

        print("\n" + "=" * 80)
        print("Training Complete")
        print("=" * 80)


if __name__ == "__main__":
    main()
