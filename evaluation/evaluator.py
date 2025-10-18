"""Evaluation system for trained policies.

Provides comprehensive evaluation metrics:
- Success rate
- Task completion time
- Intervention rate
- Generalization to new tasks
- Multi-task performance

Author: Billy Chern (Shichen)
License: MIT
"""

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from robot.arx_x5.controller import DualARXController


@dataclass
class EvaluationConfig:
    """Configuration for evaluation."""

    # Episodes
    num_episodes: int = 50  # Episodes per task
    max_episode_steps: int = 500  # Maximum steps per episode

    # Robot
    robot_left_port: str = "/dev/ttyUSB0"
    robot_right_port: str = "/dev/ttyUSB1"
    control_freq: int = 50  # Hz

    # Tasks
    tasks: List[str] = None  # Task names to evaluate

    # Success criteria
    success_threshold: float = 0.8  # Reward threshold for success
    timeout_penalty: float = -1.0  # Penalty for timeout

    # Logging
    save_videos: bool = True
    save_trajectories: bool = True
    output_dir: str = "evaluation_results"

    # Mock mode (for testing without hardware)
    mock_mode: bool = False


@dataclass
class EpisodeResult:
    """Results from a single episode."""

    task: str
    episode_id: int
    success: bool
    return_: float
    length: int
    completion_time: float
    interventions: int
    trajectory: Optional[List] = None


class PolicyEvaluator:
    """Evaluate trained policies on robot tasks.

    Example:
        >>> evaluator = PolicyEvaluator(config, policy)
        >>> results = evaluator.evaluate()
        >>> print(f"Success rate: {results['success_rate']:.2%}")
    """

    def __init__(
        self,
        config: EvaluationConfig,
        policy,  # Pi0Policy or trained model
        reward_fn=None,  # RewardClassifier
    ):
        """Initialize evaluator.

        Args:
            config: Evaluation configuration
            policy: Trained policy to evaluate
            reward_fn: Reward function for success detection
        """
        self.config = config
        self.policy = policy
        self.reward_fn = reward_fn

        # Results storage
        self.episode_results: List[EpisodeResult] = []

        # Create output directory
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        print(f"✓ Evaluator initialized")
        print(f"  Output dir: {self.output_dir}")
        print(f"  Episodes per task: {config.num_episodes}")
        print(f"  Tasks: {config.tasks or 'All'}")

    def evaluate(self) -> Dict[str, float]:
        """Run complete evaluation.

        Returns:
            Dictionary of evaluation metrics
        """
        print("\n" + "=" * 80)
        print("Starting Evaluation")
        print("=" * 80)

        # Get task list
        tasks = self.config.tasks or ["pick_and_place", "fold_towel"]

        # Evaluate each task
        for task in tasks:
            print(f"\n[Task: {task}]")
            self._evaluate_task(task)

        # Compute aggregate metrics
        metrics = self._compute_metrics()

        # Save results
        self._save_results(metrics)

        print("\n" + "=" * 80)
        print("Evaluation Complete")
        print("=" * 80)

        return metrics

    def _evaluate_task(self, task: str) -> None:
        """Evaluate policy on a specific task.

        Args:
            task: Task name
        """
        if self.config.mock_mode:
            # Mock evaluation for testing
            self._evaluate_task_mock(task)
            return

        # Initialize robot controller
        controller = DualARXController(
            left_port=self.config.robot_left_port,
            right_port=self.config.robot_right_port,
            control_freq=self.config.control_freq,
        )

        try:
            # Run episodes
            for episode_id in range(self.config.num_episodes):
                result = self._run_episode(
                    controller,
                    task,
                    episode_id,
                )

                self.episode_results.append(result)

                # Log progress
                if (episode_id + 1) % 10 == 0:
                    success_count = sum(
                        1 for r in self.episode_results[-10:]
                        if r.success
                    )
                    print(f"  Episodes {episode_id - 9}-{episode_id}: "
                          f"{success_count}/10 successful")

        finally:
            # Clean up controller
            pass

    def _evaluate_task_mock(self, task: str) -> None:
        """Mock evaluation for testing.

        Args:
            task: Task name
        """
        print(f"  Running mock evaluation for {task}...")

        for episode_id in range(self.config.num_episodes):
            # Simulate episode with random success
            success = np.random.rand() > 0.3  # 70% success rate
            return_ = np.random.rand() if success else 0.0
            length = np.random.randint(50, 200)
            completion_time = length / self.config.control_freq

            result = EpisodeResult(
                task=task,
                episode_id=episode_id,
                success=success,
                return_=return_,
                length=length,
                completion_time=completion_time,
                interventions=0,
            )

            self.episode_results.append(result)

        print(f"  ✓ Completed {self.config.num_episodes} episodes")

    def _run_episode(
        self,
        controller: DualARXController,
        task: str,
        episode_id: int,
    ) -> EpisodeResult:
        """Run single evaluation episode.

        Args:
            controller: Robot controller
            task: Task name
            episode_id: Episode number

        Returns:
            Episode results
        """
        # Reset environment
        controller.reset()

        # Episode tracking
        start_time = time.time()
        trajectory = []
        episode_return = 0.0
        success = False

        for step in range(self.config.max_episode_steps):
            # Get observation
            obs = controller.get_observation()
            images = {}  # Would get from camera system
            state = obs.joint_positions

            # Get action from policy
            action = self._get_policy_action(images, state, task)

            # Execute action
            next_obs = controller.step(action)

            # Compute reward
            reward = self._compute_reward(images, state)
            episode_return += reward

            # Check success
            if reward > self.config.success_threshold:
                success = True
                break

            # Store trajectory
            if self.config.save_trajectories:
                trajectory.append({
                    'state': state,
                    'action': action,
                    'reward': reward,
                })

        completion_time = time.time() - start_time

        return EpisodeResult(
            task=task,
            episode_id=episode_id,
            success=success,
            return_=episode_return,
            length=step + 1,
            completion_time=completion_time,
            interventions=0,
            trajectory=trajectory if self.config.save_trajectories else None,
        )

    def _get_policy_action(
        self, images: Dict, state: np.ndarray, task: str
    ) -> np.ndarray:
        """Get action from policy.

        Args:
            images: Camera images
            state: Robot state
            task: Task instruction

        Returns:
            Action to execute
        """
        # Placeholder - would use actual policy
        return np.random.randn(14) * 0.1

    def _compute_reward(self, images: Dict, state: np.ndarray) -> float:
        """Compute reward for current state.

        Args:
            images: Camera images
            state: Robot state

        Returns:
            Reward value
        """
        if self.reward_fn is not None:
            # Use learned reward classifier
            return self.reward_fn.predict(images, state)
        else:
            # Placeholder
            return 0.0

    def _compute_metrics(self) -> Dict[str, float]:
        """Compute evaluation metrics from results.

        Returns:
            Dictionary of metrics
        """
        if not self.episode_results:
            return {}

        # Overall metrics
        total_episodes = len(self.episode_results)
        successful_episodes = sum(1 for r in self.episode_results if r.success)

        metrics = {
            'total_episodes': total_episodes,
            'success_rate': successful_episodes / total_episodes,
            'avg_return': np.mean([r.return_ for r in self.episode_results]),
            'avg_length': np.mean([r.length for r in self.episode_results]),
            'avg_completion_time': np.mean(
                [r.completion_time for r in self.episode_results]
            ),
        }

        # Per-task metrics
        tasks = set(r.task for r in self.episode_results)
        for task in tasks:
            task_results = [r for r in self.episode_results if r.task == task]
            task_successes = sum(1 for r in task_results if r.success)

            metrics[f'{task}_success_rate'] = task_successes / len(task_results)
            metrics[f'{task}_avg_return'] = np.mean([r.return_ for r in task_results])

        return metrics

    def _save_results(self, metrics: Dict[str, float]) -> None:
        """Save evaluation results.

        Args:
            metrics: Evaluation metrics
        """
        import json

        # Save metrics
        metrics_file = self.output_dir / "metrics.json"
        with open(metrics_file, 'w') as f:
            json.dump(metrics, f, indent=2)

        print(f"\n✓ Results saved to {metrics_file}")

        # Print summary
        print("\nEvaluation Summary:")
        print(f"  Total episodes: {metrics['total_episodes']}")
        print(f"  Success rate: {metrics['success_rate']:.2%}")
        print(f"  Avg return: {metrics['avg_return']:.3f}")
        print(f"  Avg length: {metrics['avg_length']:.1f} steps")
        print(f"  Avg time: {metrics['avg_completion_time']:.2f}s")


def compare_policies(
    policies: List[Tuple[str, any]],
    config: EvaluationConfig,
) -> Dict[str, Dict[str, float]]:
    """Compare multiple policies.

    Args:
        policies: List of (name, policy) tuples
        config: Evaluation config

    Returns:
        Comparison results
    """
    results = {}

    for name, policy in policies:
        print(f"\n{'=' * 80}")
        print(f"Evaluating: {name}")
        print(f"{'=' * 80}")

        evaluator = PolicyEvaluator(config, policy)
        metrics = evaluator.evaluate()
        results[name] = metrics

    # Print comparison
    print("\n" + "=" * 80)
    print("Policy Comparison")
    print("=" * 80)

    for name, metrics in results.items():
        print(f"\n{name}:")
        print(f"  Success rate: {metrics['success_rate']:.2%}")
        print(f"  Avg return: {metrics['avg_return']:.3f}")

    return results
