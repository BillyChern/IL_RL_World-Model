"""Keyboard-based human intervention system for HIL-SERL.

Allows human operators to take control of the robot during training to:
- Prevent unsafe actions
- Provide corrective demonstrations
- Guide exploration

Based on HIL-SERL intervention mechanism.

Author: Billy Chern (Shichen)
License: MIT
"""

import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np

try:
    from pynput import keyboard
    PYNPUT_AVAILABLE = True
except ImportError:
    PYNPUT_AVAILABLE = False
    print("Warning: pynput not available. Keyboard intervention disabled.")


@dataclass
class InterventionConfig:
    """Configuration for intervention system."""

    # Control mapping
    delta_per_key: float = 0.05  # Radians to move per key press
    gripper_delta: float = 0.1  # Gripper open/close increment

    # Intervention detection
    intervention_timeout: float = 0.5  # Seconds without input to end intervention

    # Safety
    enable_interventions: bool = True  # Master enable/disable


class KeyboardInterventionSystem:
    """Keyboard-based human intervention for robot control.

    Allows operator to override policy actions using keyboard:
    - Arrow keys: Control arm joints
    - WASD: Alternative joint control
    - Q/E: Gripper open/close
    - Space: Trigger intervention mode
    - Esc: Emergency stop

    Example:
        >>> system = KeyboardInterventionSystem()
        >>> system.start()
        >>>
        >>> # During training loop:
        >>> if system.is_intervening():
        ...     action = system.get_intervention_action(current_state)
        ...     # Mark this transition as intervention
        >>> else:
        ...     action = policy.predict(observation)
    """

    def __init__(self, config: Optional[InterventionConfig] = None):
        """Initialize intervention system.

        Args:
            config: Intervention configuration
        """
        self.config = config or InterventionConfig()

        # State
        self.active = False
        self.intervening = False
        self.last_intervention_time = 0.0

        # Key state tracking
        self.pressed_keys = set()
        self.key_lock = threading.Lock()

        # Action delta accumulator
        self.action_delta = np.zeros(14)  # 14D for dual ARX X5

        # Keyboard listener
        self.listener = None

        # Statistics
        self.intervention_count = 0
        self.intervention_history = deque(maxlen=1000)

        if not PYNPUT_AVAILABLE:
            print("WARNING: pynput not installed. Keyboard intervention disabled.")
            self.config.enable_interventions = False

    def start(self) -> None:
        """Start listening for keyboard input."""
        if not self.config.enable_interventions or not PYNPUT_AVAILABLE:
            print("Keyboard intervention system disabled")
            return

        if self.active:
            return

        # Create keyboard listener
        self.listener = keyboard.Listener(
            on_press=self._on_key_press,
            on_release=self._on_key_release,
        )
        self.listener.start()
        self.active = True

        print("✓ Keyboard intervention system started")
        print("  Space: Enter intervention mode")
        print("  Arrow keys: Control joints")
        print("  Q/E: Gripper control")
        print("  Esc: Emergency stop")

    def stop(self) -> None:
        """Stop listening for keyboard input."""
        if not self.active:
            return

        if self.listener:
            self.listener.stop()
            self.listener = None

        self.active = False
        print("✓ Keyboard intervention system stopped")

    def _on_key_press(self, key: keyboard.Key) -> None:
        """Handle key press event.

        Args:
            key: Pressed key
        """
        with self.key_lock:
            self.pressed_keys.add(key)

            # Check for intervention trigger (Space)
            if key == keyboard.Key.space:
                self.intervening = True
                self.last_intervention_time = time.time()
                self.intervention_count += 1
                print("⚠ INTERVENTION MODE ACTIVE")

            # Emergency stop (Esc)
            elif key == keyboard.Key.esc:
                self.emergency_stop = True
                print("🛑 EMERGENCY STOP")

    def _on_key_release(self, key: keyboard.Key) -> None:
        """Handle key release event.

        Args:
            key: Released key
        """
        with self.key_lock:
            if key in self.pressed_keys:
                self.pressed_keys.remove(key)

    def is_intervening(self) -> bool:
        """Check if human is currently intervening.

        Returns:
            True if intervention is active
        """
        if not self.config.enable_interventions:
            return False

        # Check timeout
        if self.intervening:
            time_since_last = time.time() - self.last_intervention_time
            if time_since_last > self.config.intervention_timeout:
                self.intervening = False
                print("✓ Intervention ended (timeout)")

        return self.intervening

    def get_intervention_action(
        self,
        current_state: np.ndarray,
    ) -> Tuple[np.ndarray, bool]:
        """Get action from keyboard input.

        Args:
            current_state: Current robot state [14] (joint positions)

        Returns:
            action: Commanded action [14]
            modified: True if action was modified by operator
        """
        if not self.is_intervening():
            return np.zeros(14), False

        with self.key_lock:
            # Reset action delta
            self.action_delta.fill(0.0)

            # Track if any keys are pressed
            modified = len(self.pressed_keys) > 0

            # Parse keyboard input to action deltas
            for key in self.pressed_keys:
                self._process_key(key)

            # Update last intervention time if keys pressed
            if modified:
                self.last_intervention_time = time.time()

        # Compute target action (current state + delta)
        action = current_state + self.action_delta

        # Clip to safe range
        action[:6] = np.clip(action[:6], -3.14, 3.14)  # Left arm joints
        action[7:13] = np.clip(action[7:13], -3.14, 3.14)  # Right arm joints
        action[6] = np.clip(action[6], 0.0, 1.0)  # Left gripper
        action[13] = np.clip(action[13], 0.0, 1.0)  # Right gripper

        return action, modified

    def _process_key(self, key: keyboard.Key) -> None:
        """Process single key press into action delta.

        Maps keyboard keys to robot joint movements:
        - Arrow keys: Left arm control
        - WASD: Right arm control
        - Q/E: Gripper control

        Args:
            key: Keyboard key
        """
        delta = self.config.delta_per_key

        try:
            # Left arm control (Arrow keys)
            if key == keyboard.Key.up:
                self.action_delta[0] += delta  # Joint 0 forward
            elif key == keyboard.Key.down:
                self.action_delta[0] -= delta  # Joint 0 backward
            elif key == keyboard.Key.left:
                self.action_delta[1] += delta  # Joint 1 left
            elif key == keyboard.Key.right:
                self.action_delta[1] -= delta  # Joint 1 right

            # Right arm control (WASD)
            elif hasattr(key, 'char'):
                if key.char == 'w':
                    self.action_delta[7] += delta  # Joint 0 forward
                elif key.char == 's':
                    self.action_delta[7] -= delta  # Joint 0 backward
                elif key.char == 'a':
                    self.action_delta[8] += delta  # Joint 1 left
                elif key.char == 'd':
                    self.action_delta[8] -= delta  # Joint 1 right

                # Gripper control
                elif key.char == 'q':
                    self.action_delta[6] += self.config.gripper_delta  # Left gripper open
                    self.action_delta[13] += self.config.gripper_delta  # Right gripper open
                elif key.char == 'e':
                    self.action_delta[6] -= self.config.gripper_delta  # Left gripper close
                    self.action_delta[13] -= self.config.gripper_delta  # Right gripper close

        except AttributeError:
            # Key doesn't have 'char' attribute
            pass

    def get_statistics(self) -> Dict[str, float]:
        """Get intervention statistics.

        Returns:
            Dictionary with intervention metrics
        """
        return {
            'total_interventions': self.intervention_count,
            'currently_intervening': self.intervening,
        }

    def __enter__(self):
        """Context manager entry."""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop()


class MockInterventionSystem:
    """Mock intervention system for testing without keyboard.

    Useful for automated testing and headless environments.
    """

    def __init__(self, config: Optional[InterventionConfig] = None):
        """Initialize mock system."""
        self.config = config or InterventionConfig()
        self.intervening = False
        self.intervention_count = 0

    def start(self) -> None:
        """Mock start."""
        print("✓ Mock intervention system started (no keyboard)")

    def stop(self) -> None:
        """Mock stop."""
        pass

    def is_intervening(self) -> bool:
        """Always returns False for mock."""
        return False

    def get_intervention_action(
        self, current_state: np.ndarray
    ) -> Tuple[np.ndarray, bool]:
        """Returns no modification."""
        return np.zeros(14), False

    def get_statistics(self) -> Dict[str, float]:
        """Get mock statistics."""
        return {
            'total_interventions': 0,
            'currently_intervening': False,
        }

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()


def create_intervention_system(
    config: Optional[InterventionConfig] = None,
    use_mock: bool = False,
) -> KeyboardInterventionSystem:
    """Create intervention system (real or mock).

    Args:
        config: Intervention configuration
        use_mock: If True, create mock system for testing

    Returns:
        Intervention system instance
    """
    if use_mock or not PYNPUT_AVAILABLE:
        return MockInterventionSystem(config)
    else:
        return KeyboardInterventionSystem(config)
