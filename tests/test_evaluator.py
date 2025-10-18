"""Unit tests for evaluation system.

Tests the PolicyEvaluator and benchmarking utilities.

Author: Billy Chern (Shichen)
License: MIT
"""

import pytest
import numpy as np
from pathlib import Path

from evaluation import (
    PolicyEvaluator,
    EvaluationConfig,
    EpisodeResult,
    StandardBenchmark,
    BenchmarkResult,
)


class TestEvaluationConfig:
    """Test suite for EvaluationConfig."""

    def test_default_config(self):
        """Test default configuration."""
        config = EvaluationConfig()

        assert config.num_episodes == 50
        assert config.max_episode_steps == 500
        assert config.success_threshold == 0.8
        assert config.mock_mode is False

    def test_custom_config(self):
        """Test custom configuration."""
        config = EvaluationConfig(
            num_episodes=100,
            max_episode_steps=1000,
            tasks=["pick_and_place"],
            mock_mode=True,
        )

        assert config.num_episodes == 100
        assert config.max_episode_steps == 1000
        assert config.tasks == ["pick_and_place"]
        assert config.mock_mode is True


class TestEpisodeResult:
    """Test suite for EpisodeResult."""

    def test_episode_result_creation(self):
        """Test creating episode result."""
        result = EpisodeResult(
            task="pick_and_place",
            episode_id=0,
            success=True,
            return_=0.95,
            length=150,
            completion_time=3.0,
            interventions=2,
        )

        assert result.task == "pick_and_place"
        assert result.success is True
        assert result.return_ == 0.95
        assert result.length == 150
        assert result.interventions == 2


class TestPolicyEvaluator:
    """Test suite for PolicyEvaluator."""

    @pytest.fixture
    def mock_policy(self):
        """Create mock policy."""
        class MockPolicy:
            def predict(self, images, instruction):
                return np.random.randn(14) * 0.1

        return MockPolicy()

    @pytest.fixture
    def mock_config(self, tmp_path):
        """Create mock evaluation config."""
        return EvaluationConfig(
            num_episodes=5,  # Small number for testing
            max_episode_steps=50,
            tasks=["pick_and_place"],
            mock_mode=True,  # Use mock mode
            output_dir=str(tmp_path / "eval_results"),
        )

    def test_initialization(self, mock_config, mock_policy):
        """Test evaluator initialization."""
        evaluator = PolicyEvaluator(mock_config, mock_policy)

        assert evaluator.config == mock_config
        assert evaluator.policy == mock_policy
        assert len(evaluator.episode_results) == 0

    def test_evaluate_mock_mode(self, mock_config, mock_policy):
        """Test evaluation in mock mode."""
        evaluator = PolicyEvaluator(mock_config, mock_policy)

        # Run evaluation
        metrics = evaluator.evaluate()

        # Check metrics exist
        assert 'total_episodes' in metrics
        assert 'success_rate' in metrics
        assert 'avg_return' in metrics
        assert 'avg_length' in metrics

        # Check correct number of episodes
        assert metrics['total_episodes'] == 5

        # Check success rate is reasonable
        assert 0.0 <= metrics['success_rate'] <= 1.0

    def test_multiple_tasks(self, mock_policy, tmp_path):
        """Test evaluation with multiple tasks."""
        config = EvaluationConfig(
            num_episodes=3,
            tasks=["pick_and_place", "fold_towel"],
            mock_mode=True,
            output_dir=str(tmp_path / "eval_multi"),
        )

        evaluator = PolicyEvaluator(config, mock_policy)
        metrics = evaluator.evaluate()

        # Should have metrics for both tasks
        assert 'pick_and_place_success_rate' in metrics
        assert 'fold_towel_success_rate' in metrics

        # Total episodes should be 3 per task = 6
        assert metrics['total_episodes'] == 6

    def test_results_saved(self, mock_config, mock_policy):
        """Test that results are saved to disk."""
        evaluator = PolicyEvaluator(mock_config, mock_policy)
        metrics = evaluator.evaluate()

        # Check that results file exists
        results_file = Path(mock_config.output_dir) / "metrics.json"
        assert results_file.exists()

        # Load and verify
        import json
        with open(results_file) as f:
            saved_metrics = json.load(f)

        assert saved_metrics['total_episodes'] == metrics['total_episodes']


class TestBenchmarkResult:
    """Test suite for BenchmarkResult."""

    def test_benchmark_result_creation(self):
        """Test creating benchmark result."""
        result = BenchmarkResult(
            method="IL+RL+WorldModel",
            success_rate=0.85,
            avg_return=0.82,
            avg_episode_length=125.5,
            avg_training_time=2.5,
            total_samples=5000,
            imagined_samples=50000,
        )

        assert result.method == "IL+RL+WorldModel"
        assert result.success_rate == 0.85
        assert result.total_samples == 5000
        assert result.imagined_samples == 50000


class TestStandardBenchmark:
    """Test suite for StandardBenchmark."""

    def test_initialization(self):
        """Test benchmark initialization."""
        benchmark = StandardBenchmark(num_episodes_per_task=10)

        assert benchmark.num_episodes == 10
        assert len(benchmark.TASKS) == 5
        assert "pick_and_place" in benchmark.TASKS

    def test_run_suite(self):
        """Test running benchmark suite."""
        class MockPolicy:
            def predict(self, *args, **kwargs):
                return np.zeros(14)

        benchmark = StandardBenchmark(num_episodes_per_task=5)
        policy = MockPolicy()

        result = benchmark.run_suite(policy, "TestMethod")

        assert result.method == "TestMethod"
        assert 0.0 <= result.success_rate <= 1.0

    def test_print_comparison(self):
        """Test comparison printing."""
        results = [
            BenchmarkResult(
                method="IL-only",
                success_rate=0.60,
                avg_return=0.55,
                avg_episode_length=150,
                avg_training_time=1.0,
                total_samples=10000,
                imagined_samples=0,
            ),
            BenchmarkResult(
                method="IL+RL+WorldModel",
                success_rate=0.85,
                avg_return=0.80,
                avg_episode_length=120,
                avg_training_time=2.0,
                total_samples=5000,
                imagined_samples=50000,
            ),
        ]

        benchmark = StandardBenchmark()

        # Should not raise errors
        benchmark.print_comparison(results)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
