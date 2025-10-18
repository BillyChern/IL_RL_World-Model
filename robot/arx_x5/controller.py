"""ARX X5 Dual-Arm Robot Controller.

This module provides a high-level interface for controlling dual ARX X5 robot arms
for bi-manual manipulation tasks. It handles:
- 50Hz joint position control
- Proprioceptive feedback (joint angles, velocities, currents, gripper state)
- Safety constraints and workspace limits
- Observation collection for RL/IL training

Based on the ARX X5 SDK: https://github.com/ARXroboticsX/ARX_X5

Author: Billy Chern (Shichen)
License: MIT
"""

import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

# Note: Import ARX X5 SDK when available
# For now, we'll use a mock interface that matches the SDK API
try:
    from arx_x5_sdk import SingleArm
except ImportError:
    # Mock class for development/testing without hardware
    class SingleArm:
        """Mock ARX X5 arm for development without hardware."""

        def __init__(self, port: str):
            self.port = port
            self._joint_positions = np.zeros(6)
            self._joint_velocities = np.zeros(6)
            self._joint_currents = np.zeros(6)
            self._gripper_position = 0.0

        def set_joint_positions(self, positions: np.ndarray) -> None:
            """Set target joint positions."""
            self._joint_positions = np.array(positions)

        def set_gripper(self, position: float) -> None:
            """Set gripper position (0=closed, 1=open)."""
            self._gripper_position = np.clip(position, 0.0, 1.0)

        def get_joint_positions(self) -> np.ndarray:
            """Get current joint positions."""
            return self._joint_positions.copy()

        def get_joint_velocities(self) -> np.ndarray:
            """Get current joint velocities."""
            return self._joint_velocities.copy()

        def get_joint_currents(self) -> np.ndarray:
            """Get current joint currents (torque proxy)."""
            return self._joint_currents.copy()

        def get_gripper_position(self) -> float:
            """Get current gripper position."""
            return self._gripper_position

        def enable(self) -> None:
            """Enable the arm."""
            pass

        def disable(self) -> None:
            """Disable the arm."""
            pass


@dataclass
class SafetyLimits:
    """Safety limits for ARX X5 robot arms.

    These limits should be configured based on your specific workspace
    and task requirements. Default values are conservative.
    """

    # Joint position limits (radians) for 6-DOF arm
    joint_pos_min: np.ndarray = None
    joint_pos_max: np.ndarray = None

    # Joint velocity limits (rad/s)
    joint_vel_max: np.ndarray = None

    # Joint current limits (A) - proxy for torque limits
    joint_current_max: np.ndarray = None

    # Gripper limits
    gripper_min: float = 0.0  # Closed
    gripper_max: float = 1.0  # Open

    # Maximum action delta per timestep (for action smoothing)
    max_action_delta: float = 0.1  # radians per control step

    def __post_init__(self):
        """Initialize default arrays if not provided."""
        if self.joint_pos_min is None:
            self.joint_pos_min = np.array([-3.14, -2.0, -2.0, -3.14, -1.57, -3.14])
        if self.joint_pos_max is None:
            self.joint_pos_max = np.array([3.14, 2.0, 2.0, 3.14, 1.57, 3.14])
        if self.joint_vel_max is None:
            self.joint_vel_max = np.array([2.0, 2.0, 2.0, 2.0, 2.0, 2.0])
        if self.joint_current_max is None:
            self.joint_current_max = np.array([3.0, 3.0, 3.0, 2.0, 2.0, 2.0])


@dataclass
class Observation:
    """Complete observation from dual ARX X5 robot.

    This structure matches the format expected by VLA policies and
    RL algorithms. It includes all proprioceptive information and
    placeholders for visual observations (cameras handled separately).
    """

    # Proprioception (14D total: 2 arms × 7D)
    joint_positions: np.ndarray  # [14] - 6 joints + gripper per arm
    joint_velocities: np.ndarray  # [14]
    joint_currents: np.ndarray  # [14] - torque proxy

    # Gripper states
    gripper_states: np.ndarray  # [2] - left, right

    # Timestamp
    timestamp: float

    # Images will be added by camera module
    # images: Dict[str, np.ndarray]  # {'left': [...], 'right': [...], 'base': [...]}


class DualARXController:
    """High-level controller for dual ARX X5 arms.

    This controller manages bi-manual manipulation with two ARX X5 robot arms.
    It provides:
    - Synchronized control of both arms at 50Hz
    - Comprehensive proprioceptive feedback
    - Safety monitoring and constraint enforcement
    - Action smoothing for stable control
    - Easy integration with RL/IL pipelines

    Example:
        >>> controller = DualARXController(
        ...     left_port='/dev/ttyUSB0',
        ...     right_port='/dev/ttyUSB1',
        ...     control_freq=50
        ... )
        >>> controller.reset()
        >>> obs = controller.get_observation()
        >>> action = policy(obs)  # 14D: [left_arm_7D, right_arm_7D]
        >>> next_obs = controller.step(action)
    """

    def __init__(
        self,
        left_port: str = "/dev/ttyUSB0",
        right_port: str = "/dev/ttyUSB1",
        control_freq: int = 50,
        safety_limits: Optional[SafetyLimits] = None,
        enable_action_smoothing: bool = True,
        verbose: bool = True,
    ):
        """Initialize dual ARX X5 controller.

        Args:
            left_port: Serial port for left arm
            right_port: Serial port for right arm
            control_freq: Control frequency in Hz (default: 50)
            safety_limits: Safety constraints (uses defaults if None)
            enable_action_smoothing: Enable action delta limiting
            verbose: Print status messages
        """
        self.control_freq = control_freq
        self.dt = 1.0 / control_freq
        self.safety_limits = safety_limits or SafetyLimits()
        self.enable_action_smoothing = enable_action_smoothing
        self.verbose = verbose

        # Initialize robot arms
        if self.verbose:
            print(f"Initializing ARX X5 arms on ports {left_port}, {right_port}...")

        self.left_arm = SingleArm(port=left_port)
        self.right_arm = SingleArm(port=right_port)

        # State tracking
        self.last_action: Optional[np.ndarray] = None
        self.step_count = 0
        self.start_time = time.time()

        # Enable arms
        self.left_arm.enable()
        self.right_arm.enable()

        if self.verbose:
            print("✓ ARX X5 dual-arm controller initialized successfully")

    def reset(self, home_positions: Optional[np.ndarray] = None) -> Observation:
        """Reset robot to home position.

        Args:
            home_positions: Target home positions [14D]. If None, uses zero positions.

        Returns:
            Initial observation after reset
        """
        if home_positions is None:
            # Default home: all joints at 0, grippers open
            home_positions = np.zeros(14)
            home_positions[[6, 13]] = 0.5  # Grippers half-open

        # Split into left and right arm actions
        left_action = home_positions[:7]
        right_action = home_positions[7:14]

        # Move to home position
        self._execute_action_internal(left_action, right_action)

        # Wait for robot to settle
        time.sleep(0.5)

        # Reset state
        self.last_action = home_positions
        self.step_count = 0
        self.start_time = time.time()

        if self.verbose:
            print("✓ Robot reset to home position")

        return self.get_observation()

    def step(self, action: np.ndarray) -> Observation:
        """Execute one control step.

        Args:
            action: Joint positions [14D] - [left_6_joints, left_gripper,
                                            right_6_joints, right_gripper]

        Returns:
            Observation after action execution

        Raises:
            ValueError: If action has wrong shape or violates safety constraints
        """
        if action.shape != (14,):
            raise ValueError(f"Action must be 14D, got shape {action.shape}")

        # Apply safety constraints
        action = self._enforce_safety(action)

        # Apply action smoothing if enabled
        if self.enable_action_smoothing and self.last_action is not None:
            action = self._smooth_action(action, self.last_action)

        # Split into left and right arm actions
        left_action = action[:7]
        right_action = action[7:14]

        # Execute action
        start_time = time.perf_counter()
        self._execute_action_internal(left_action, right_action)

        # Get observation
        obs = self.get_observation()

        # Update state
        self.last_action = action
        self.step_count += 1

        # Maintain control frequency
        elapsed = time.perf_counter() - start_time
        if elapsed < self.dt:
            time.sleep(self.dt - elapsed)

        return obs

    def get_observation(self) -> Observation:
        """Get current robot observation.

        Returns:
            Complete proprioceptive observation from both arms
        """
        # Left arm proprioception
        left_joint_pos = self.left_arm.get_joint_positions()
        left_joint_vel = self.left_arm.get_joint_velocities()
        left_joint_current = self.left_arm.get_joint_currents()
        left_gripper = self.left_arm.get_gripper_position()

        # Right arm proprioception
        right_joint_pos = self.right_arm.get_joint_positions()
        right_joint_vel = self.right_arm.get_joint_velocities()
        right_joint_current = self.right_arm.get_joint_currents()
        right_gripper = self.right_arm.get_gripper_position()

        # Concatenate into observation
        joint_positions = np.concatenate([left_joint_pos, [left_gripper],
                                         right_joint_pos, [right_gripper]])
        joint_velocities = np.concatenate([left_joint_vel, [0.0],  # Gripper vel placeholder
                                          right_joint_vel, [0.0]])
        joint_currents = np.concatenate([left_joint_current, [0.0],  # Gripper current placeholder
                                        right_joint_current, [0.0]])

        return Observation(
            joint_positions=joint_positions,
            joint_velocities=joint_velocities,
            joint_currents=joint_currents,
            gripper_states=np.array([left_gripper, right_gripper]),
            timestamp=time.time(),
        )

    def _execute_action_internal(self, left_action: np.ndarray, right_action: np.ndarray) -> None:
        """Execute action on both arms simultaneously.

        Args:
            left_action: [7D] left arm joint positions + gripper
            right_action: [7D] right arm joint positions + gripper
        """
        # Left arm
        self.left_arm.set_joint_positions(left_action[:6])
        self.left_arm.set_gripper(left_action[6])

        # Right arm
        self.right_arm.set_joint_positions(right_action[:6])
        self.right_arm.set_gripper(right_action[6])

    def _enforce_safety(self, action: np.ndarray) -> np.ndarray:
        """Enforce safety constraints on action.

        Args:
            action: Raw action [14D]

        Returns:
            Safety-constrained action [14D]
        """
        action = action.copy()

        # Joint position limits (separate for each arm)
        for arm_idx in [0, 1]:
            start_idx = arm_idx * 7
            joint_start = start_idx
            joint_end = start_idx + 6

            # Clip joint positions
            action[joint_start:joint_end] = np.clip(
                action[joint_start:joint_end],
                self.safety_limits.joint_pos_min,
                self.safety_limits.joint_pos_max,
            )

            # Clip gripper
            gripper_idx = start_idx + 6
            action[gripper_idx] = np.clip(
                action[gripper_idx],
                self.safety_limits.gripper_min,
                self.safety_limits.gripper_max,
            )

        return action

    def _smooth_action(self, action: np.ndarray, last_action: np.ndarray) -> np.ndarray:
        """Apply action smoothing to prevent large jumps.

        Args:
            action: Current action [14D]
            last_action: Previous action [14D]

        Returns:
            Smoothed action [14D]
        """
        delta = action - last_action
        max_delta = self.safety_limits.max_action_delta

        # Clip delta
        delta = np.clip(delta, -max_delta, max_delta)

        return last_action + delta

    def close(self) -> None:
        """Disable arms and close connection."""
        if self.verbose:
            print("Closing ARX X5 controller...")

        self.left_arm.disable()
        self.right_arm.disable()

        if self.verbose:
            print("✓ ARX X5 controller closed")

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()

    def get_state_dict(self) -> Dict:
        """Get controller state for logging/debugging.

        Returns:
            Dictionary containing current state information
        """
        obs = self.get_observation()
        return {
            "step_count": self.step_count,
            "elapsed_time": time.time() - self.start_time,
            "joint_positions": obs.joint_positions.tolist(),
            "joint_velocities": obs.joint_velocities.tolist(),
            "joint_currents": obs.joint_currents.tolist(),
            "gripper_states": obs.gripper_states.tolist(),
            "last_action": self.last_action.tolist() if self.last_action is not None else None,
        }
