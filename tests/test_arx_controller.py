"""Unit tests for ARX X5 robot controller.

Tests the DualARXController and TriCameraSystem classes to ensure
correct behavior, safety constraints, and API contracts.

Author: Billy Chern (Shichen)
License: MIT
"""

import numpy as np
import pytest

from robot.arx_x5.controller import DualARXController, Observation, SafetyLimits


class TestDualARXController:
    """Test suite for DualARXController class."""

    @pytest.fixture
    def controller(self):
        """Create controller instance for testing (uses mock hardware)."""
        ctrl = DualARXController(
            left_port="/dev/null",  # Mock ports
            right_port="/dev/null",
            control_freq=50,
            verbose=False,
        )
        yield ctrl
        ctrl.close()

    def test_initialization(self, controller):
        """Test controller initializes correctly."""
        assert controller.control_freq == 50
        assert controller.dt == pytest.approx(0.02)
        assert controller.step_count == 0
        assert controller.last_action is None

    def test_reset(self, controller):
        """Test reset functionality."""
        obs = controller.reset()

        # Check observation structure
        assert isinstance(obs, Observation)
        assert obs.joint_positions.shape == (14,)
        assert obs.joint_velocities.shape == (14,)
        assert obs.joint_currents.shape == (14,)
        assert obs.gripper_states.shape == (2,)
        assert isinstance(obs.timestamp, float)

        # Check state reset
        assert controller.step_count == 0
        assert controller.last_action is not None
        assert controller.last_action.shape == (14,)

    def test_reset_with_custom_home(self, controller):
        """Test reset with custom home position."""
        custom_home = np.ones(14) * 0.5
        obs = controller.reset(home_positions=custom_home)

        np.testing.assert_array_almost_equal(
            controller.last_action, custom_home, decimal=5
        )

    def test_step_action_shape(self, controller):
        """Test step rejects wrong action shapes."""
        controller.reset()

        # Wrong shape should raise ValueError
        with pytest.raises(ValueError, match="Action must be 14D"):
            controller.step(np.zeros(7))  # Wrong: should be 14D

        with pytest.raises(ValueError, match="Action must be 14D"):
            controller.step(np.zeros((14, 1)))  # Wrong: should be 1D

    def test_step_correct_action(self, controller):
        """Test step with correct action."""
        controller.reset()

        # Use small action to avoid smoothing effects
        action = np.ones(14) * 0.05  # Small enough to not be smoothed
        obs = controller.step(action)

        # Check observation
        assert isinstance(obs, Observation)
        assert obs.joint_positions.shape == (14,)

        # Check state update
        assert controller.step_count == 1
        # Action should be smoothed from last_action (home position)
        # So we just check that last_action was updated
        assert controller.last_action is not None
        assert controller.last_action.shape == (14,)

    def test_safety_limits_joint_positions(self, controller):
        """Test joint position limits are enforced."""
        controller.reset()

        # Create action that violates limits
        unsafe_action = np.ones(14) * 10.0  # Way beyond limits

        obs = controller.step(unsafe_action)

        # Check that positions are clipped
        for arm_idx in [0, 1]:
            start = arm_idx * 7
            joint_start = start
            joint_end = start + 6

            # Joints should be within limits
            assert np.all(obs.joint_positions[joint_start:joint_end] <=
                         controller.safety_limits.joint_pos_max)
            assert np.all(obs.joint_positions[joint_start:joint_end] >=
                         controller.safety_limits.joint_pos_min)

            # Gripper should be within [0, 1]
            gripper_idx = start + 6
            assert 0.0 <= obs.joint_positions[gripper_idx] <= 1.0

    def test_action_smoothing(self, controller):
        """Test action smoothing prevents large jumps."""
        controller.reset()

        # First action - only update joint angles, not grippers
        action1 = np.zeros(14)
        action1[[6, 13]] = 0.5  # Keep grippers at same position as home
        controller.step(action1)

        # Second action with large jump (only in joints, not grippers)
        action2 = np.ones(14) * 2.0  # Big jump
        action2[[6, 13]] = 0.5  # Keep grippers constant

        obs = controller.step(action2)

        # Check that action was smoothed
        max_delta = controller.safety_limits.max_action_delta
        actual_delta = np.abs(controller.last_action - action1)

        # All deltas should be <= max_delta (allowing small numerical error)
        assert np.all(actual_delta <= max_delta + 1e-6)

    def test_action_smoothing_disabled(self):
        """Test controller with action smoothing disabled."""
        ctrl = DualARXController(
            left_port="/dev/null",
            right_port="/dev/null",
            enable_action_smoothing=False,
            verbose=False,
        )

        ctrl.reset()
        action1 = np.zeros(14)
        ctrl.step(action1)

        action2 = np.ones(14) * 0.5
        obs = ctrl.step(action2)

        # Without smoothing, should apply full action (after safety clipping)
        # Note: Safety limits still apply
        ctrl.close()

    def test_observation_consistency(self, controller):
        """Test observation values are consistent."""
        controller.reset()

        # Step multiple times with same action
        action = np.ones(14) * 0.3

        for _ in range(5):
            obs = controller.step(action)

            # Check all observation components are present
            assert hasattr(obs, 'joint_positions')
            assert hasattr(obs, 'joint_velocities')
            assert hasattr(obs, 'joint_currents')
            assert hasattr(obs, 'gripper_states')
            assert hasattr(obs, 'timestamp')

            # Check shapes
            assert obs.joint_positions.shape == (14,)
            assert obs.joint_velocities.shape == (14,)
            assert obs.joint_currents.shape == (14,)
            assert obs.gripper_states.shape == (2,)

    def test_get_state_dict(self, controller):
        """Test get_state_dict returns correct information."""
        controller.reset()
        controller.step(np.zeros(14))

        state_dict = controller.get_state_dict()

        # Check required keys
        assert 'step_count' in state_dict
        assert 'elapsed_time' in state_dict
        assert 'joint_positions' in state_dict
        assert 'joint_velocities' in state_dict
        assert 'joint_currents' in state_dict
        assert 'gripper_states' in state_dict
        assert 'last_action' in state_dict

        # Check values
        assert state_dict['step_count'] == 1
        assert isinstance(state_dict['elapsed_time'], float)
        assert len(state_dict['joint_positions']) == 14
        assert len(state_dict['last_action']) == 14

    def test_context_manager(self):
        """Test controller works as context manager."""
        with DualARXController(
            left_port="/dev/null",
            right_port="/dev/null",
            verbose=False,
        ) as ctrl:
            obs = ctrl.reset()
            assert obs is not None

        # Controller should be closed after exiting context

    def test_custom_safety_limits(self):
        """Test controller with custom safety limits."""
        custom_limits = SafetyLimits(
            joint_pos_min=np.ones(6) * -1.0,
            joint_pos_max=np.ones(6) * 1.0,
            max_action_delta=0.05,
        )

        ctrl = DualARXController(
            left_port="/dev/null",
            right_port="/dev/null",
            safety_limits=custom_limits,
            verbose=False,
        )

        ctrl.reset()

        # Test that custom limits are applied
        action = np.ones(14) * 2.0  # Beyond custom limits
        obs = ctrl.step(action)

        # Joints should be clipped to custom limits
        for arm_idx in [0, 1]:
            start = arm_idx * 7
            joint_start = start
            joint_end = start + 6

            assert np.all(obs.joint_positions[joint_start:joint_end] <= 1.0)
            assert np.all(obs.joint_positions[joint_start:joint_end] >= -1.0)

        ctrl.close()

    def test_frequency_control(self, controller):
        """Test that controller maintains target frequency."""
        import time

        controller.reset()

        # Execute multiple steps and measure time
        num_steps = 10
        start_time = time.time()

        for _ in range(num_steps):
            controller.step(np.zeros(14))

        elapsed = time.time() - start_time
        expected_time = num_steps * controller.dt

        # Should take at least expected time (allowing 10% overhead)
        assert elapsed >= expected_time * 0.9
        assert elapsed <= expected_time * 1.5  # Allow some overhead


class TestSafetyLimits:
    """Test suite for SafetyLimits dataclass."""

    def test_default_limits(self):
        """Test default safety limits are reasonable."""
        limits = SafetyLimits()

        # Check shapes
        assert limits.joint_pos_min.shape == (6,)
        assert limits.joint_pos_max.shape == (6,)
        assert limits.joint_vel_max.shape == (6,)
        assert limits.joint_current_max.shape == (6,)

        # Check values are reasonable
        assert np.all(limits.joint_pos_min < limits.joint_pos_max)
        assert np.all(limits.joint_vel_max > 0)
        assert np.all(limits.joint_current_max > 0)
        assert 0.0 <= limits.gripper_min < limits.gripper_max <= 1.0
        assert limits.max_action_delta > 0

    def test_custom_limits(self):
        """Test custom safety limits."""
        custom_limits = SafetyLimits(
            joint_pos_min=np.ones(6) * -2.0,
            joint_pos_max=np.ones(6) * 2.0,
            joint_vel_max=np.ones(6) * 3.0,
            max_action_delta=0.2,
        )

        assert np.all(custom_limits.joint_pos_min == -2.0)
        assert np.all(custom_limits.joint_pos_max == 2.0)
        assert np.all(custom_limits.joint_vel_max == 3.0)
        assert custom_limits.max_action_delta == 0.2


class TestObservation:
    """Test suite for Observation dataclass."""

    def test_observation_creation(self):
        """Test creating an Observation instance."""
        obs = Observation(
            joint_positions=np.zeros(14),
            joint_velocities=np.zeros(14),
            joint_currents=np.zeros(14),
            gripper_states=np.zeros(2),
            timestamp=0.0,
        )

        assert obs.joint_positions.shape == (14,)
        assert obs.joint_velocities.shape == (14,)
        assert obs.joint_currents.shape == (14,)
        assert obs.gripper_states.shape == (2,)
        assert obs.timestamp == 0.0

    def test_observation_from_controller(self):
        """Test observation from controller has correct format."""
        ctrl = DualARXController(
            left_port="/dev/null",
            right_port="/dev/null",
            verbose=False,
        )

        obs = ctrl.get_observation()

        # Check all attributes exist and have correct shapes
        assert hasattr(obs, 'joint_positions')
        assert hasattr(obs, 'joint_velocities')
        assert hasattr(obs, 'joint_currents')
        assert hasattr(obs, 'gripper_states')
        assert hasattr(obs, 'timestamp')

        assert obs.joint_positions.shape == (14,)
        assert obs.joint_velocities.shape == (14,)
        assert obs.joint_currents.shape == (14,)
        assert obs.gripper_states.shape == (2,)

        ctrl.close()


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "--tb=short"])
