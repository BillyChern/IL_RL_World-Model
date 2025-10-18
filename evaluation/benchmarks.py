"""Benchmarking utilities for comparing models and baselines.

Provides standardized benchmarks for:
- IL-only baseline
- RL-only baseline
- IL+RL (HIL-SERL)
- IL+RL+World Model (full system)

Author: Billy Chern (Shichen)
License: MIT
"""

import time
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np


@dataclass
class BenchmarkResult:
    """Results from a benchmark run."""

    method: str  # "IL-only", "RL-only", "IL+RL", "IL+RL+WorldModel"
    success_rate: float
    avg_return: float
    avg_episode_length: float
    avg_training_time: float  # Hours
    total_samples: int  # Real-world samples used
    imagined_samples: int  # Simulated samples (world model)


class StandardBenchmark:
    """Standard benchmark suite for robotic manipulation.

    Based on tasks from:
    - HIL-SERL paper
    - π0.5 paper
    - DayDreamer paper

    Example:
        >>> benchmark = StandardBenchmark()
        >>> results = benchmark.run_suite(policy)
        >>> benchmark.print_comparison(results)
    """

    # Standard benchmark tasks
    TASKS = [
        "pick_and_place",  # Pick object from table
        "fold_towel",  # Fold towel in half
        "open_drawer",  # Open drawer
        "close_drawer",  # Close drawer
        "stack_blocks",  # Stack 3 blocks
    ]

    # Success criteria per task
    SUCCESS_CRITERIA = {
        "pick_and_place": 0.8,  # Object lifted
        "fold_towel": 0.7,  # Towel folded
        "open_drawer": 0.9,  # Drawer fully open
        "close_drawer": 0.9,  # Drawer fully closed
        "stack_blocks": 0.6,  # All blocks stacked
    }

    def __init__(self, num_episodes_per_task: int = 50):
        """Initialize benchmark.

        Args:
            num_episodes_per_task: Episodes to run per task
        """
        self.num_episodes = num_episodes_per_task

    def run_suite(
        self,
        policy,
        method_name: str = "Unknown",
    ) -> BenchmarkResult:
        """Run full benchmark suite.

        Args:
            policy: Policy to evaluate
            method_name: Name of method being benchmarked

        Returns:
            Benchmark results
        """
        print(f"\n{'=' * 80}")
        print(f"Running Benchmark: {method_name}")
        print(f"{'=' * 80}")

        task_results = {}

        for task in self.TASKS:
            print(f"\n[Task: {task}]")
            success_rate = self._evaluate_task(policy, task)
            task_results[task] = success_rate

            print(f"  Success rate: {success_rate:.2%}")

        # Aggregate results
        overall_success = np.mean(list(task_results.values()))

        result = BenchmarkResult(
            method=method_name,
            success_rate=overall_success,
            avg_return=0.0,  # Placeholder
            avg_episode_length=0.0,  # Placeholder
            avg_training_time=0.0,  # Placeholder
            total_samples=0,  # Placeholder
            imagined_samples=0,  # Placeholder
        )

        print(f"\n{'=' * 80}")
        print(f"Benchmark Complete: {method_name}")
        print(f"Overall Success Rate: {overall_success:.2%}")
        print(f"{'=' * 80}")

        return result

    def _evaluate_task(self, policy, task: str) -> float:
        """Evaluate policy on single task.

        Args:
            policy: Policy to evaluate
            task: Task name

        Returns:
            Success rate
        """
        # Mock evaluation for now
        # Would run actual episodes on robot
        return np.random.uniform(0.5, 0.9)

    def print_comparison(self, results: List[BenchmarkResult]) -> None:
        """Print comparison table of results.

        Args:
            results: List of benchmark results to compare
        """
        print("\n" + "=" * 80)
        print("Benchmark Comparison")
        print("=" * 80)

        # Print header
        print(f"{'Method':<25} {'Success Rate':>15} {'Real Samples':>15} {'Imagined':>15}")
        print("-" * 80)

        # Print results
        for result in results:
            print(f"{result.method:<25} "
                  f"{result.success_rate:>14.1%} "
                  f"{result.total_samples:>15,} "
                  f"{result.imagined_samples:>15,}")

        print("=" * 80)


def compare_sample_efficiency(
    methods: Dict[str, List[float]],
) -> None:
    """Compare sample efficiency across methods.

    Args:
        methods: Dict mapping method name to list of success rates over training
    """
    print("\n" + "=" * 80)
    print("Sample Efficiency Comparison")
    print("=" * 80)

    for method, success_rates in methods.items():
        # Find samples needed to reach 80% success
        samples_to_80 = None
        for i, rate in enumerate(success_rates):
            if rate >= 0.8:
                samples_to_80 = (i + 1) * 100  # Assuming 100 samples per checkpoint
                break

        print(f"\n{method}:")
        print(f"  Final success rate: {success_rates[-1]:.1%}")
        if samples_to_80:
            print(f"  Samples to 80%: {samples_to_80:,}")
        else:
            print(f"  Did not reach 80%")


def benchmark_inference_speed(policy, num_iterations: int = 100) -> Dict[str, float]:
    """Benchmark policy inference speed.

    Args:
        policy: Policy to benchmark
        num_iterations: Number of iterations to run

    Returns:
        Timing statistics
    """
    print("\nBenchmarking inference speed...")

    import torch

    # Create dummy inputs
    images = {
        'left': torch.randn(1, 3, 224, 224),
        'right': torch.randn(1, 3, 224, 224),
        'base': torch.randn(1, 3, 224, 224),
    }
    instruction = "pick up the cup"

    # Warmup
    for _ in range(10):
        # Would call policy.predict()
        pass

    # Benchmark
    times = []
    for _ in range(num_iterations):
        start = time.time()
        # Would call policy.predict()
        times.append(time.time() - start)

    stats = {
        'mean_ms': np.mean(times) * 1000,
        'std_ms': np.std(times) * 1000,
        'min_ms': np.min(times) * 1000,
        'max_ms': np.max(times) * 1000,
        'fps': 1.0 / np.mean(times),
    }

    print(f"  Mean: {stats['mean_ms']:.2f} ms")
    print(f"  Std: {stats['std_ms']:.2f} ms")
    print(f"  FPS: {stats['fps']:.1f}")

    return stats
