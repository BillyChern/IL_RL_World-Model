"""Unit tests for replay buffer.

Tests the ReplayBuffer and EpisodeBuffer classes to ensure correct
behavior for storing, sampling, and mixing demonstrations.

Author: Billy Chern (Shichen)
License: MIT
"""

import numpy as np
import pytest
import torch

from data.replay_buffer import EpisodeBuffer, ReplayBuffer, Transition


class TestTransition:
    """Test suite for Transition dataclass."""

    def test_transition_creation(self):
        """Test creating a Transition instance."""
        images = {
            "left": np.zeros((224, 224, 3), dtype=np.uint8),
            "right": np.zeros((224, 224, 3), dtype=np.uint8),
            "base": np.zeros((224, 224, 3), dtype=np.uint8),
        }

        transition = Transition(
            images=images,
            state=np.zeros(14),
            action=np.zeros(14),
            reward=1.0,
            next_images=images,
            next_state=np.zeros(14),
            done=False,
            language="pick up the cup",
            is_demo=True,
            is_intervention=False,
            episode_id=0,
            step_id=0,
        )

        assert transition.reward == 1.0
        assert transition.done is False
        assert transition.is_demo is True
        assert transition.language == "pick up the cup"


class TestReplayBuffer:
    """Test suite for ReplayBuffer class."""

    @pytest.fixture
    def buffer(self):
        """Create buffer instance for testing."""
        return ReplayBuffer(capacity=1000, demo_ratio=0.5, seed=42)

    @pytest.fixture
    def sample_transition(self):
        """Create sample transition for testing."""
        images = {
            "left": np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8),
            "right": np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8),
            "base": np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8),
        }

        return Transition(
            images=images,
            state=np.random.randn(14).astype(np.float32),
            action=np.random.randn(14).astype(np.float32),
            reward=np.random.rand(),
            next_images=images,
            next_state=np.random.randn(14).astype(np.float32),
            done=False,
            language="test instruction",
            is_demo=False,
            is_intervention=False,
            episode_id=0,
            step_id=0,
        )

    def test_initialization(self, buffer):
        """Test buffer initializes correctly."""
        assert buffer.capacity == 1000
        assert buffer.demo_ratio == 0.5
        assert len(buffer) == 0
        assert buffer.num_demos == 0
        assert buffer.num_interventions == 0

    def test_add_transition(self, buffer, sample_transition):
        """Test adding a single transition."""
        buffer.add(sample_transition)

        assert len(buffer) == 1
        assert buffer.total_added == 1

    def test_add_demo_transition(self, buffer, sample_transition):
        """Test adding demonstration transition."""
        sample_transition.is_demo = True
        buffer.add(sample_transition)

        assert len(buffer) == 1
        assert buffer.num_demos == 1
        assert len(buffer.demo_buffer) == 1
        assert len(buffer.buffer) == 0

    def test_add_intervention(self, buffer, sample_transition):
        """Test adding intervention transition."""
        sample_transition.is_intervention = True
        buffer.add(sample_transition)

        assert len(buffer) == 1
        assert buffer.num_interventions == 1
        assert len(buffer.intervention_indices) == 1

    def test_add_batch(self, buffer, sample_transition):
        """Test adding multiple transitions."""
        transitions = [sample_transition for _ in range(10)]
        buffer.add_batch(transitions)

        assert len(buffer) == 10
        assert buffer.total_added == 10

    def test_sample(self, buffer, sample_transition):
        """Test sampling from buffer."""
        # Add enough transitions
        for i in range(100):
            transition = sample_transition
            transition.episode_id = i // 10
            transition.step_id = i % 10
            buffer.add(transition)

        # Sample batch
        batch = buffer.sample(batch_size=32)

        # Check batch structure
        assert "images" in batch
        assert "states" in batch
        assert "actions" in batch
        assert "rewards" in batch
        assert "next_states" in batch
        assert "dones" in batch
        assert "languages" in batch

        # Check shapes
        assert batch["states"].shape == (32, 14)
        assert batch["actions"].shape == (32, 14)
        assert batch["rewards"].shape == (32,)

        # Check images
        for camera in ["left", "right", "base"]:
            assert camera in batch["images"]
            assert batch["images"][camera].shape == (32, 224, 224, 3)

    def test_demo_ratio(self, buffer):
        """Test that demo_ratio is respected in sampling."""
        # Add 50 demos
        for i in range(50):
            images = {
                "left": np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8),
                "right": np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8),
                "base": np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8),
            }
            demo = Transition(
                images=images,
                state=np.random.randn(14).astype(np.float32),
                action=np.random.randn(14).astype(np.float32),
                reward=np.random.rand(),
                next_images=images,
                next_state=np.random.randn(14).astype(np.float32),
                done=False,
                language="test instruction",
                is_demo=True,
                is_intervention=False,
                episode_id=i,
                step_id=0,
            )
            buffer.add(demo)

        # Add 50 online
        for i in range(50):
            images = {
                "left": np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8),
                "right": np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8),
                "base": np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8),
            }
            online = Transition(
                images=images,
                state=np.random.randn(14).astype(np.float32),
                action=np.random.randn(14).astype(np.float32),
                reward=np.random.rand(),
                next_images=images,
                next_state=np.random.randn(14).astype(np.float32),
                done=False,
                language="test instruction",
                is_demo=False,
                is_intervention=False,
                episode_id=50 + i,
                step_id=0,
            )
            buffer.add(online)

        # Sample multiple batches and check ratio
        num_batches = 10
        total_demos = 0

        for _ in range(num_batches):
            batch = buffer.sample(batch_size=32)
            total_demos += batch["is_demo"].sum().item()

        # Should be approximately 50% demos (with some variance)
        demo_percentage = total_demos / (32 * num_batches)
        assert 0.4 <= demo_percentage <= 0.6

    def test_sample_episode(self, buffer):
        """Test sampling complete episodes."""
        # Add complete episode
        episode_id = 5
        for step in range(20):
            images = {
                "left": np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8),
                "right": np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8),
                "base": np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8),
            }
            transition = Transition(
                images=images,
                state=np.random.randn(14).astype(np.float32),
                action=np.random.randn(14).astype(np.float32),
                reward=np.random.rand(),
                next_images=images,
                next_state=np.random.randn(14).astype(np.float32),
                done=(step == 19),
                language="test instruction",
                is_demo=False,
                is_intervention=False,
                episode_id=episode_id,
                step_id=step,
            )
            buffer.add(transition)

        # Sample the episode
        episode = buffer.sample_episode(episode_id=episode_id)

        assert len(episode) == 20
        assert all(t.episode_id == episode_id for t in episode)
        assert episode[-1].done is True

        # Check ordering
        for i, transition in enumerate(episode):
            assert transition.step_id == i

    def test_capacity_limit(self):
        """Test that buffer respects capacity limit."""
        buffer = ReplayBuffer(capacity=10)

        # Add more transitions than capacity
        for i in range(20):
            transition = Transition(
                images={
                    "left": np.zeros((224, 224, 3), dtype=np.uint8),
                    "right": np.zeros((224, 224, 3), dtype=np.uint8),
                    "base": np.zeros((224, 224, 3), dtype=np.uint8),
                },
                state=np.zeros(14),
                action=np.zeros(14),
                reward=0.0,
                next_images={
                    "left": np.zeros((224, 224, 3), dtype=np.uint8),
                    "right": np.zeros((224, 224, 3), dtype=np.uint8),
                    "base": np.zeros((224, 224, 3), dtype=np.uint8),
                },
                next_state=np.zeros(14),
                done=False,
                language="test",
                episode_id=i,
                step_id=0,
            )
            buffer.add(transition)

        # Should only keep most recent 10
        assert len(buffer) <= 10

    def test_clear(self, buffer, sample_transition):
        """Test clearing the buffer."""
        # Add some transitions
        for _ in range(10):
            buffer.add(sample_transition)

        assert len(buffer) > 0

        # Clear
        buffer.clear()

        assert len(buffer) == 0
        assert buffer.num_demos == 0
        assert buffer.num_interventions == 0
        assert buffer.total_added == 0

    def test_get_stats(self, buffer, sample_transition):
        """Test getting buffer statistics."""
        # Add mixed transitions
        for i in range(30):
            transition = sample_transition
            transition.is_demo = (i < 10)
            transition.is_intervention = (i >= 20)
            buffer.add(transition)

        stats = buffer.get_stats()

        assert stats["total_transitions"] == 30
        assert stats["demo_transitions"] == 10
        assert stats["online_transitions"] == 20
        assert stats["interventions"] == 10
        assert stats["capacity"] == 1000


class TestEpisodeBuffer:
    """Test suite for EpisodeBuffer class."""

    @pytest.fixture
    def episode_buffer(self):
        """Create episode buffer for testing."""
        return EpisodeBuffer(capacity=100, seed=42)

    @pytest.fixture
    def sample_episode(self):
        """Create sample episode for testing."""
        episode = []
        for step in range(50):
            images = {
                "left": np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8),
                "right": np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8),
                "base": np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8),
            }

            transition = Transition(
                images=images,
                state=np.random.randn(14).astype(np.float32),
                action=np.random.randn(14).astype(np.float32),
                reward=np.random.rand(),
                next_images=images,
                next_state=np.random.randn(14).astype(np.float32),
                done=(step == 49),
                language="test instruction",
                episode_id=0,
                step_id=step,
            )
            episode.append(transition)

        return episode

    def test_add_episode(self, episode_buffer, sample_episode):
        """Test adding an episode."""
        episode_buffer.add_episode(sample_episode)

        assert len(episode_buffer) == 1

    def test_sample_episodes(self, episode_buffer, sample_episode):
        """Test sampling episode chunks."""
        # Add multiple episodes
        for ep_id in range(10):
            episode = sample_episode.copy()
            for t in episode:
                t.episode_id = ep_id
            episode_buffer.add_episode(episode)

        # Sample chunks
        chunks = episode_buffer.sample_episodes(batch_size=8, chunk_length=20)

        assert len(chunks) == 8
        for chunk in chunks:
            assert len(chunk) <= 20

    def test_capacity_limit(self):
        """Test episode buffer capacity limit."""
        buffer = EpisodeBuffer(capacity=5)

        # Add more episodes than capacity
        for i in range(10):
            episode = [
                Transition(
                    images={
                        "left": np.zeros((224, 224, 3), dtype=np.uint8),
                        "right": np.zeros((224, 224, 3), dtype=np.uint8),
                        "base": np.zeros((224, 224, 3), dtype=np.uint8),
                    },
                    state=np.zeros(14),
                    action=np.zeros(14),
                    reward=0.0,
                    next_images={
                        "left": np.zeros((224, 224, 3), dtype=np.uint8),
                        "right": np.zeros((224, 224, 3), dtype=np.uint8),
                        "base": np.zeros((224, 224, 3), dtype=np.uint8),
                    },
                    next_state=np.zeros(14),
                    done=True,
                    language="test",
                    episode_id=i,
                    step_id=0,
                )
            ]
            buffer.add_episode(episode)

        # Should only keep most recent 5
        assert len(buffer) == 5


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "--tb=short"])
