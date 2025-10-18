#!/usr/bin/env python3
"""Evaluation script for trained policies.

Usage:
    python scripts/evaluate.py --checkpoint checkpoints/final.pt --num_episodes 50

Author: Billy Chern (Shichen)
License: MIT
"""

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.vla import Pi0Policy
from models.reward import RewardClassifier
from evaluation.evaluator import PolicyEvaluator, EvaluationConfig


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Evaluate trained policy")

    # Model checkpoint
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to trained policy checkpoint"
    )
    parser.add_argument(
        "--reward_checkpoint",
        type=str,
        default=None,
        help="Path to reward classifier checkpoint"
    )

    # Evaluation
    parser.add_argument(
        "--num_episodes",
        type=int,
        default=50,
        help="Number of episodes per task"
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=500,
        help="Maximum steps per episode"
    )
    parser.add_argument(
        "--tasks",
        type=str,
        nargs="+",
        default=None,
        help="Tasks to evaluate (default: all)"
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

    # Output
    parser.add_argument(
        "--output_dir",
        type=str,
        default="evaluation_results",
        help="Directory to save results"
    )
    parser.add_argument(
        "--save_videos",
        action="store_true",
        help="Save episode videos"
    )
    parser.add_argument(
        "--save_trajectories",
        action="store_true",
        help="Save episode trajectories"
    )

    # Mock mode (for testing)
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Run in mock mode (no hardware)"
    )

    return parser.parse_args()


def main():
    """Main evaluation function."""
    args = parse_args()

    print("=" * 80)
    print("Policy Evaluation")
    print("=" * 80)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Episodes per task: {args.num_episodes}")
    print(f"Tasks: {args.tasks or 'All'}")
    print(f"Mock mode: {args.mock}")
    print("=" * 80)

    # Load policy
    print("\n[Setup] Loading policy...")
    if args.mock:
        # Create dummy policy for mock mode
        from models.vla import Pi0Config
        policy = Pi0Policy(Pi0Config())
        print("✓ Loaded mock policy")
    else:
        policy = Pi0Policy.from_checkpoint(args.checkpoint)
        print(f"✓ Loaded policy from {args.checkpoint}")

    # Load reward classifier
    reward_fn = None
    if args.reward_checkpoint:
        print(f"\n[Setup] Loading reward classifier...")
        reward_fn = RewardClassifier.from_checkpoint(args.reward_checkpoint)
        print(f"✓ Loaded reward classifier")

    # Create evaluation config
    config = EvaluationConfig(
        num_episodes=args.num_episodes,
        max_episode_steps=args.max_steps,
        tasks=args.tasks,
        robot_left_port=args.robot_left_port,
        robot_right_port=args.robot_right_port,
        save_videos=args.save_videos,
        save_trajectories=args.save_trajectories,
        output_dir=args.output_dir,
        mock_mode=args.mock,
    )

    # Create evaluator
    print("\n[Setup] Creating evaluator...")
    evaluator = PolicyEvaluator(config, policy, reward_fn)

    # Run evaluation
    print("\n" + "=" * 80)
    print("Starting Evaluation")
    print("=" * 80)

    try:
        metrics = evaluator.evaluate()

        print("\n" + "=" * 80)
        print("Evaluation Complete!")
        print("=" * 80)
        print(f"\nSuccess Rate: {metrics['success_rate']:.2%}")
        print(f"Average Return: {metrics['avg_return']:.3f}")
        print(f"Average Length: {metrics['avg_length']:.1f} steps")
        print(f"Average Time: {metrics['avg_completion_time']:.2f}s")

        print(f"\nResults saved to: {config.output_dir}")

    except KeyboardInterrupt:
        print("\n⚠ Evaluation interrupted by user")
    except Exception as e:
        print(f"\n❌ Evaluation failed: {e}")
        raise


if __name__ == "__main__":
    main()
