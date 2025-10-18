"""Asynchronous training system for real-time RL.

Implements dual-loop training:
- Robot collection thread: Collects real-world data
- GPU training thread: Updates policy and world model
- Runs simultaneously for maximum sample efficiency

Based on DayDreamer and HIL-SERL approaches.

Author: Billy Chern (Shichen)
License: MIT
"""

import queue
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import wandb

from data.replay_buffer import ReplayBuffer, Transition
from robot.arx_x5.controller import DualARXController
from robot.intervention import create_intervention_system, InterventionConfig


@dataclass
class AsyncTrainerConfig:
    """Configuration for async training."""

    # Training
    num_episodes: int = 1000
    max_episode_steps: int = 500
    train_freq: int = 1  # Train every N steps
    batch_size: int = 256

    # Robot collection
    robot_left_port: str = "/dev/ttyUSB0"
    robot_right_port: str = "/dev/ttyUSB1"
    control_freq: int = 50  # Hz

    # Replay buffer
    buffer_capacity: int = 100000
    demo_ratio: float = 0.5  # Mix of demos in batch

    # World model imagination
    use_imagination: bool = True
    imagination_freq: int = 4  # Imagine every N train steps
    imagination_horizon: int = 10  # Steps to imagine ahead
    imagination_batch_size: int = 64

    # Interventions
    enable_interventions: bool = True
    intervention_timeout: float = 0.5

    # Logging
    log_freq: int = 10  # Log every N episodes
    checkpoint_freq: int = 50  # Save every N episodes
    checkpoint_dir: str = "checkpoints"

    # W&B
    use_wandb: bool = True
    wandb_project: str = "il_rl_world_model"
    wandb_entity: Optional[str] = None
    wandb_run_name: Optional[str] = None

    # Safety
    emergency_stop: bool = False


class AsyncTrainer:
    """Asynchronous trainer for real-time RL.

    Runs robot collection and GPU training in parallel threads.

    Example:
        >>> config = AsyncTrainerConfig()
        >>> trainer = AsyncTrainer(config, policy, world_model, reward_fn)
        >>> trainer.train()
    """

    def __init__(
        self,
        config: AsyncTrainerConfig,
        policy,  # Pi0Policy or SACAgent
        world_model=None,  # DreamerV3 (optional)
        reward_fn=None,  # RewardClassifier
    ):
        """Initialize async trainer.

        Args:
            config: Training configuration
            policy: Policy network
            world_model: World model (optional, for imagination)
            reward_fn: Reward function
        """
        self.config = config
        self.policy = policy
        self.world_model = world_model
        self.reward_fn = reward_fn

        # Replay buffer
        self.replay_buffer = ReplayBuffer(
            capacity=config.buffer_capacity,
            demo_ratio=config.demo_ratio,
        )

        # Thread-safe queues for communication
        self.transition_queue = queue.Queue(maxsize=1000)
        self.metrics_queue = queue.Queue(maxsize=100)

        # Thread control
        self.stop_event = threading.Event()
        self.collection_thread = None
        self.training_thread = None

        # Statistics
        self.global_step = 0
        self.episode_count = 0
        self.total_interventions = 0

        # W&B logging
        if config.use_wandb:
            wandb.init(
                project=config.wandb_project,
                entity=config.wandb_entity,
                name=config.wandb_run_name,
                config=config.__dict__,
            )

    def train(self) -> None:
        """Start asynchronous training.

        Launches robot collection and GPU training threads.
        """
        print("=" * 80)
        print("Starting Asynchronous Training")
        print("=" * 80)

        # Start threads
        self.collection_thread = threading.Thread(
            target=self._robot_collection_loop,
            name="RobotCollection",
            daemon=True,
        )

        self.training_thread = threading.Thread(
            target=self._gpu_training_loop,
            name="GPUTraining",
            daemon=True,
        )

        self.collection_thread.start()
        self.training_thread.start()

        print("✓ Collection thread started")
        print("✓ Training thread started")

        # Monitor threads
        try:
            while not self.stop_event.is_set():
                time.sleep(1.0)

                # Check if we've completed training
                if self.episode_count >= self.config.num_episodes:
                    print(f"\n✓ Training complete ({self.config.num_episodes} episodes)")
                    self.stop()
                    break

                # Check for emergency stop
                if self.config.emergency_stop:
                    print("\n🛑 Emergency stop triggered!")
                    self.stop()
                    break

        except KeyboardInterrupt:
            print("\n⚠ Training interrupted by user")
            self.stop()

        # Wait for threads to finish
        self.collection_thread.join(timeout=5.0)
        self.training_thread.join(timeout=5.0)

        print("=" * 80)
        print("Training Finished")
        print(f"Episodes: {self.episode_count}")
        print(f"Steps: {self.global_step}")
        print(f"Interventions: {self.total_interventions}")
        print("=" * 80)

        if self.config.use_wandb:
            wandb.finish()

    def stop(self) -> None:
        """Stop training threads."""
        self.stop_event.set()

    def _robot_collection_loop(self) -> None:
        """Robot collection thread.

        Continuously collects data from the robot.
        """
        print("[Collection] Starting robot collection loop...")

        # Initialize robot controller
        controller = DualARXController(
            left_port=self.config.robot_left_port,
            right_port=self.config.robot_right_port,
            control_freq=self.config.control_freq,
        )

        # Initialize intervention system
        intervention_config = InterventionConfig(
            intervention_timeout=self.config.intervention_timeout,
            enable_interventions=self.config.enable_interventions,
        )
        intervention_system = create_intervention_system(intervention_config)
        intervention_system.start()

        episode_num = 0

        try:
            while not self.stop_event.is_set():
                # Run one episode
                episode_data = self._collect_episode(
                    controller,
                    intervention_system,
                    episode_num,
                )

                # Add transitions to queue for training thread
                for transition in episode_data['transitions']:
                    try:
                        self.transition_queue.put(transition, timeout=1.0)
                    except queue.Full:
                        print("[Collection] Warning: Transition queue full, dropping data")

                # Update statistics
                episode_num += 1
                self.episode_count = episode_num

                # Log episode metrics
                if episode_num % self.config.log_freq == 0:
                    print(f"[Collection] Episode {episode_num}: "
                          f"Return={episode_data['return']:.2f}, "
                          f"Interventions={episode_data['num_interventions']}")

        finally:
            intervention_system.stop()
            print("[Collection] Robot collection loop stopped")

    def _collect_episode(
        self,
        controller: DualARXController,
        intervention_system,
        episode_num: int,
    ) -> Dict:
        """Collect one episode of data.

        Args:
            controller: Robot controller
            intervention_system: Intervention system
            episode_num: Current episode number

        Returns:
            Episode data dictionary
        """
        # Reset environment
        controller.reset()

        # Get task instruction (placeholder - would come from task sampler)
        instruction = "pick and place the object"

        # Episode tracking
        transitions = []
        episode_return = 0.0
        num_interventions = 0

        for step in range(self.config.max_episode_steps):
            # Get current observation
            obs = controller.get_observation()
            images = {}  # Would get from camera system
            state = obs.joint_positions

            # Check for intervention
            if intervention_system.is_intervening():
                action, modified = intervention_system.get_intervention_action(state)
                is_intervention = modified
                if modified:
                    num_interventions += 1
            else:
                # Get action from policy
                action = self._get_policy_action(images, state, instruction)
                is_intervention = False

            # Execute action
            next_obs = controller.step(action)
            next_state = next_obs.joint_positions

            # Compute reward (placeholder)
            reward = 0.0  # Would use reward classifier
            done = (step == self.config.max_episode_steps - 1)

            # Create transition
            transition = Transition(
                images=images,
                state=state,
                action=action,
                reward=reward,
                next_images={},  # Would get next images
                next_state=next_state,
                done=done,
                language=instruction,
                is_demo=False,
                is_intervention=is_intervention,
                episode_id=episode_num,
                step_id=step,
            )

            transitions.append(transition)
            episode_return += reward

            if done:
                break

        return {
            'transitions': transitions,
            'return': episode_return,
            'num_interventions': num_interventions,
            'length': len(transitions),
        }

    def _get_policy_action(self, images: Dict, state: np.ndarray, instruction: str) -> np.ndarray:
        """Get action from policy.

        Args:
            images: Camera images
            state: Robot state
            instruction: Task instruction

        Returns:
            Action to execute
        """
        # Placeholder - would use actual policy
        return np.random.randn(14) * 0.1

    def _gpu_training_loop(self) -> None:
        """GPU training thread.

        Continuously trains policy and world model.
        """
        print("[Training] Starting GPU training loop...")

        train_step = 0

        while not self.stop_event.is_set():
            # Get transitions from collection thread
            try:
                transition = self.transition_queue.get(timeout=1.0)
                self.replay_buffer.add(transition)
            except queue.Empty:
                # No new data yet, sleep briefly
                time.sleep(0.01)
                continue

            # Train if we have enough data
            if len(self.replay_buffer) < self.config.batch_size:
                continue

            # Train every N steps
            if train_step % self.config.train_freq == 0:
                metrics = self._train_step()

                # Log metrics
                if self.config.use_wandb:
                    wandb.log(metrics, step=train_step)

                # Imagination rollouts
                if (self.config.use_imagination and
                    train_step % self.config.imagination_freq == 0 and
                    self.world_model is not None):
                    self._imagination_step()

            train_step += 1
            self.global_step = train_step

            # Checkpoint
            if train_step % (self.config.checkpoint_freq * 100) == 0:
                self._save_checkpoint(train_step)

        print("[Training] GPU training loop stopped")

    def _train_step(self) -> Dict[str, float]:
        """Single training step.

        Returns:
            Training metrics
        """
        # Sample batch from replay buffer
        batch = self.replay_buffer.sample(self.config.batch_size)

        # Train policy (placeholder)
        metrics = {
            'train/loss': 0.0,
            'train/buffer_size': len(self.replay_buffer),
        }

        return metrics

    def _imagination_step(self) -> None:
        """Generate imagined rollouts with world model."""
        # Placeholder for imagination
        pass

    def _save_checkpoint(self, step: int) -> None:
        """Save training checkpoint.

        Args:
            step: Current training step
        """
        checkpoint_dir = Path(self.config.checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        checkpoint_path = checkpoint_dir / f"checkpoint_step_{step}.pt"

        # Placeholder - would save actual models
        torch.save({
            'step': step,
            'episode': self.episode_count,
        }, checkpoint_path)

        print(f"[Training] Checkpoint saved: {checkpoint_path}")
