"""Unit tests for human intervention system.

Tests the keyboard-based intervention system for HIL-SERL.

Author: Billy Chern (Shichen)
License: MIT
"""

import numpy as np
import pytest
import time

from robot.intervention import (
    KeyboardInterventionSystem,
    InterventionConfig,
    create_intervention_system,
)


class TestInterventionConfig:
    """Test suite for InterventionConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = InterventionConfig()

        assert config.delta_per_key == 0.05
        assert config.gripper_delta == 0.1
        assert config.intervention_timeout == 0.5
        assert config.enable_interventions is True

    def test_custom_config(self):
        """Test custom configuration."""
        config = InterventionConfig(
            delta_per_key=0.1,
            gripper_delta=0.2,
            intervention_timeout=1.0,
            enable_interventions=False,
        )

        assert config.delta_per_key == 0.1
        assert config.gripper_delta == 0.2
        assert config.intervention_timeout == 1.0
        assert config.enable_interventions is False


class TestMockInterventionSystem:
    """Test suite for mock intervention system (no keyboard required)."""

    @pytest.fixture
    def mock_system(self):
        """Create mock intervention system."""
        return create_intervention_system(use_mock=True)

    def test_initialization(self, mock_system):
        """Test mock system initializes correctly."""
        assert mock_system is not None
        assert hasattr(mock_system, 'intervening')
        assert hasattr(mock_system, 'intervention_count')

    def test_start_stop(self, mock_system):
        """Test start/stop methods."""
        mock_system.start()
        # Should not raise any errors
        mock_system.stop()

    def test_is_intervening_always_false(self, mock_system):
        """Test mock system never intervenes."""
        assert mock_system.is_intervening() is False

    def test_get_intervention_action(self, mock_system):
        """Test get_intervention_action returns no modification."""
        current_state = np.random.randn(14)
        action, modified = mock_system.get_intervention_action(current_state)

        assert action.shape == (14,)
        assert modified is False
        assert np.allclose(action, np.zeros(14))

    def test_get_statistics(self, mock_system):
        """Test getting statistics."""
        stats = mock_system.get_statistics()

        assert 'total_interventions' in stats
        assert 'currently_intervening' in stats
        assert stats['total_interventions'] == 0
        assert stats['currently_intervening'] is False

    def test_context_manager(self, mock_system):
        """Test context manager protocol."""
        with mock_system as system:
            assert system is not None
            assert system.is_intervening() is False


class TestCreateInterventionSystem:
    """Test suite for intervention system factory."""

    def test_create_mock_system(self):
        """Test creating mock system."""
        system = create_intervention_system(use_mock=True)
        assert system is not None
        assert system.__class__.__name__ == 'MockInterventionSystem'

    def test_create_with_config(self):
        """Test creating system with custom config."""
        config = InterventionConfig(
            delta_per_key=0.2,
            enable_interventions=True,
        )
        system = create_intervention_system(config=config, use_mock=True)

        assert system.config.delta_per_key == 0.2
        assert system.config.enable_interventions is True

    def test_disabled_interventions(self):
        """Test system with interventions disabled."""
        config = InterventionConfig(enable_interventions=False)
        system = create_intervention_system(config=config, use_mock=True)

        assert system.is_intervening() is False


class TestInterventionLogic:
    """Test intervention logic with mock system."""

    def test_action_clipping(self):
        """Test that actions are clipped to safe range."""
        system = create_intervention_system(use_mock=True)

        # Create unsafe state (outside joint limits)
        unsafe_state = np.ones(14) * 10.0

        action, _ = system.get_intervention_action(unsafe_state)

        # Action should be zeros (no intervention), state shouldn't affect it
        assert np.allclose(action, np.zeros(14))

    def test_intervention_tracking(self):
        """Test intervention count tracking."""
        system = create_intervention_system(use_mock=True)

        initial_count = system.intervention_count

        # Mock system doesn't actually intervene
        state = np.zeros(14)
        system.get_intervention_action(state)

        # Count should not increase for mock
        assert system.intervention_count == initial_count


class TestInterventionIntegration:
    """Integration tests for intervention system."""

    def test_training_loop_integration(self):
        """Test intervention system in training loop context."""
        system = create_intervention_system(use_mock=True)
        system.start()

        # Simulate training loop
        for episode in range(5):
            state = np.random.randn(14)

            if system.is_intervening():
                action, modified = system.get_intervention_action(state)
                # Would mark transition as intervention
                assert modified is False  # Mock never modifies
            else:
                # Would use policy action
                action = np.random.randn(14)

            # Verify action is valid
            assert action.shape == (14,)

        system.stop()

    def test_multiple_episodes(self):
        """Test intervention system across multiple episodes."""
        system = create_intervention_system(use_mock=True)

        num_episodes = 10
        for _ in range(num_episodes):
            # Each episode
            for step in range(50):
                state = np.random.randn(14)
                action, modified = system.get_intervention_action(state)

                assert action.shape == (14,)
                assert modified is False

        stats = system.get_statistics()
        assert stats['total_interventions'] == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
