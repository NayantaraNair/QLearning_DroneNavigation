#!/usr/bin/env python3
"""Performance comparison between Simple and Binary Obstacle drones."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import time
import random
from typing import List, Dict, Tuple
import matplotlib.pyplot as plt

from src.config import DEFAULT_GRID, DEFAULT_LEARNING, DEFAULT_SIMULATION
from src.grid import GridWorld
from src.drone import QLearningDrone, BinaryObstacleDrone


class PerformanceMetrics:
    """Container for tracking performance metrics."""

    def __init__(self, name: str):
        self.name = name
        self.episode_rewards: List[float] = []
        self.episode_lengths: List[int] = []
        self.success_count: int = 0
        self.collision_count: int = 0
        self.q_table_sizes: List[int] = []
        self.training_times: List[float] = []

    def add_episode(self, reward: float, length: int, success: bool, collision: bool, q_table_size: int):
        self.episode_rewards.append(reward)
        self.episode_lengths.append(length)
        self.q_table_sizes.append(q_table_size)
        if success:
            self.success_count += 1
        if collision:
            self.collision_count += 1

    def success_rate(self) -> float:
        total = len(self.episode_rewards)
        return (self.success_count / total) if total > 0 else 0.0

    def collision_rate(self) -> float:
        total = len(self.episode_rewards)
        return (self.collision_count / total) if total > 0 else 0.0

    def average_reward(self) -> float:
        return sum(self.episode_rewards) / len(self.episode_rewards) if self.episode_rewards else 0.0

    def average_length(self) -> float:
        return sum(self.episode_lengths) / len(self.episode_lengths) if self.episode_lengths else 0.0

    def final_q_table_size(self) -> int:
        return self.q_table_sizes[-1] if self.q_table_sizes else 0


def train_drone_type(drone_class, episodes: int, world_template: GridWorld) -> PerformanceMetrics:
    """Train a drone type and collect performance metrics."""

    if drone_class == QLearningDrone:
        metrics = PerformanceMetrics("Simple Drone")
    else:
        metrics = PerformanceMetrics("Binary Obstacle Drone")

    # Create fresh world and drone
    world = GridWorld(
        settings=world_template.settings,
        rewards=world_template.rewards,
        obstacles=set(world_template.obstacles) if world_template.obstacles else None
    )

    drone = drone_class(world)

    print(f"\nTraining {metrics.name} for {episodes} episodes...")

    start_time = time.time()

    for episode in range(episodes):
        episode_start = time.time()

        # Start new episode
        drone.start_new_episode()
        episode_reward = 0.0
        episode_length = 0
        success = False
        collision = False

        # Run episode
        max_steps = 1000
        for step in range(max_steps):
            diagnostics = drone.step()
            episode_reward += diagnostics['reward']
            episode_length += 1

            if diagnostics['done']:
                success = True
                break

            # Check for collision (negative reward indicates obstacle hit)
            if diagnostics['reward'] < -10:
                collision = True

        episode_time = time.time() - episode_start
        metrics.training_times.append(episode_time)

        # Record metrics
        q_table_size = len(drone.q_table)
        metrics.add_episode(episode_reward, episode_length, success, collision, q_table_size)

        # Progress reporting
        if (episode + 1) % 100 == 0:
            avg_reward = sum(metrics.episode_rewards[-10:]) / min(10, len(metrics.episode_rewards))
            recent_success_rate = sum(1 for r in metrics.episode_rewards[-50:] if r > 500) / min(50, len(metrics.episode_rewards))
            print(f"  Episode {episode + 1}/{episodes} - Avg reward (last 10): {avg_reward:.1f}, Success rate (last 50): {recent_success_rate:.1%}, Q-table: {q_table_size}")

    total_time = time.time() - start_time
    print(f"  Completed in {total_time:.1f} seconds")
    print(f"  Final success rate: {metrics.success_rate():.1%}")
    print(f"  Final Q-table size: {metrics.final_q_table_size()}")

    return metrics


def run_comparison(episodes: int = None) -> Tuple[PerformanceMetrics, PerformanceMetrics]:
    """Run performance comparison between Simple and Binary Obstacle drones."""

    print("=== DRONE PERFORMANCE COMPARISON ===")

    # Use configuration-based episode counts proportionate to state space
    simple_episodes = DEFAULT_SIMULATION.max_episodes if episodes is None else episodes
    binary_episodes = DEFAULT_SIMULATION.max_episodes if episodes is None else int(episodes * 16)  # 16x state space

    print(f"Training with proportionate episodes based on state space:")
    print(f"  Simple Drone: {simple_episodes} episodes (state space: rows × cols = {DEFAULT_GRID.rows}×{DEFAULT_GRID.cols})")
    print(f"  Binary Drone: {binary_episodes} episodes (state space: rows × cols × 2^4 = {DEFAULT_GRID.rows}×{DEFAULT_GRID.cols}×16)")

    # Create a consistent world for both tests
    world_template = GridWorld()
    world_template.reset()

    print(f"Grid: {world_template.settings.rows}x{world_template.settings.cols}")
    print(f"Obstacles: {len(world_template.obstacles or [])}")
    print(f"Start: {world_template.settings.start}")
    print(f"Goal: {world_template.settings.goal}")

    # Train Simple Drone
    simple_metrics = train_drone_type(QLearningDrone, simple_episodes, world_template)

    # Train Binary Obstacle Drone with proportionate episodes
    binary_metrics = train_drone_type(BinaryObstacleDrone, binary_episodes, world_template)

    return simple_metrics, binary_metrics


def analyze_results(simple_metrics: PerformanceMetrics, binary_metrics: PerformanceMetrics):
    """Analyze and display comparison results."""

    print("\n" + "="*60)
    print("PERFORMANCE ANALYSIS")
    print("="*60)

    print(f"\n{simple_metrics.name}:")
    print(f"  Episodes: {len(simple_metrics.episode_rewards)}")
    print(f"  Success Rate: {simple_metrics.success_rate():.1%}")
    print(f"  Collision Rate: {simple_metrics.collision_rate():.1%}")
    print(f"  Average Reward: {simple_metrics.average_reward():.1f}")
    print(f"  Average Episode Length: {simple_metrics.average_length():.1f}")
    print(f"  Final Q-Table Size: {simple_metrics.final_q_table_size()}")

    print(f"\n{binary_metrics.name}:")
    print(f"  Episodes: {len(binary_metrics.episode_rewards)}")
    print(f"  Success Rate: {binary_metrics.success_rate():.1%}")
    print(f"  Collision Rate: {binary_metrics.collision_rate():.1%}")
    print(f"  Average Reward: {binary_metrics.average_reward():.1f}")
    print(f"  Average Episode Length: {binary_metrics.average_length():.1f}")
    print(f"  Final Q-Table Size: {binary_metrics.final_q_table_size()}")

    print(f"\nCOMPARISON:")
    success_diff = binary_metrics.success_rate() - simple_metrics.success_rate()
    collision_diff = binary_metrics.collision_rate() - simple_metrics.collision_rate()
    reward_diff = binary_metrics.average_reward() - simple_metrics.average_reward()

    print(f"  Success Rate Difference: {success_diff:+.1%}")
    print(f"  Collision Rate Difference: {collision_diff:+.1%}")
    print(f"  Average Reward Difference: {reward_diff:+.1f}")

    if success_diff > 0.05:  # 5% improvement threshold
        print("  >>> Binary Obstacle drone shows SIGNIFICANT improvement in success rate!")
    elif success_diff > 0:
        print("  >> Binary Obstacle drone shows modest improvement in success rate")
    else:
        print("  >> Binary Obstacle drone shows no improvement in success rate")


def plot_comparison(simple_metrics: PerformanceMetrics, binary_metrics: PerformanceMetrics):
    """Create comparison plots."""
    try:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle('Drone Performance Comparison', fontsize=16)

        # Success rate over time (rolling average)
        def rolling_success_rate(rewards, window=50):
            rates = []
            for i in range(len(rewards)):
                start = max(0, i - window + 1)
                recent_rewards = rewards[start:i+1]
                success_count = sum(1 for r in recent_rewards if r > 500)  # Goal reward threshold
                rates.append(success_count / len(recent_rewards))
            return rates

        simple_success = rolling_success_rate(simple_metrics.episode_rewards)
        binary_success = rolling_success_rate(binary_metrics.episode_rewards)

        axes[0,0].plot(simple_success, label='Simple Drone', alpha=0.7)
        axes[0,0].plot(binary_success, label='Binary Obstacle Drone', alpha=0.7)
        axes[0,0].set_title('Success Rate (Rolling Average)')
        axes[0,0].set_xlabel('Episode')
        axes[0,0].set_ylabel('Success Rate')
        axes[0,0].legend()
        axes[0,0].grid(True, alpha=0.3)

        # Average reward over time
        def rolling_average(values, window=50):
            averages = []
            for i in range(len(values)):
                start = max(0, i - window + 1)
                avg = sum(values[start:i+1]) / (i - start + 1)
                averages.append(avg)
            return averages

        simple_rewards = rolling_average(simple_metrics.episode_rewards)
        binary_rewards = rolling_average(binary_metrics.episode_rewards)

        axes[0,1].plot(simple_rewards, label='Simple Drone', alpha=0.7)
        axes[0,1].plot(binary_rewards, label='Binary Obstacle Drone', alpha=0.7)
        axes[0,1].set_title('Average Reward (Rolling Average)')
        axes[0,1].set_xlabel('Episode')
        axes[0,1].set_ylabel('Average Reward')
        axes[0,1].legend()
        axes[0,1].grid(True, alpha=0.3)

        # Q-table growth
        axes[1,0].plot(simple_metrics.q_table_sizes, label='Simple Drone', alpha=0.7)
        axes[1,0].plot(binary_metrics.q_table_sizes, label='Binary Obstacle Drone', alpha=0.7)
        axes[1,0].set_title('Q-Table Size Growth')
        axes[1,0].set_xlabel('Episode')
        axes[1,0].set_ylabel('Number of States')
        axes[1,0].legend()
        axes[1,0].grid(True, alpha=0.3)

        # Episode length comparison
        simple_lengths = rolling_average(simple_metrics.episode_lengths)
        binary_lengths = rolling_average(binary_metrics.episode_lengths)

        axes[1,1].plot(simple_lengths, label='Simple Drone', alpha=0.7)
        axes[1,1].plot(binary_lengths, label='Binary Obstacle Drone', alpha=0.7)
        axes[1,1].set_title('Average Episode Length')
        axes[1,1].set_xlabel('Episode')
        axes[1,1].set_ylabel('Steps per Episode')
        axes[1,1].legend()
        axes[1,1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('drone_performance_comparison.png', dpi=150, bbox_inches='tight')
        print(f"\n>> Performance plots saved as 'drone_performance_comparison.png'")
        plt.show()

    except ImportError:
        print("\n>> Matplotlib not available - skipping plots")


if __name__ == "__main__":
    # Run comparison with proportionate episode counts based on state space
    print("Running proportionate episode comparison based on state space size...")
    print("Simple drone: 2D state space (row, col)")
    print("Binary drone: 6D state space (row, col, obstacle_N, obstacle_S, obstacle_W, obstacle_E)")
    print("Episodes are proportionate: Simple gets fewer episodes, Binary gets more due to 16x larger state space.")
    simple_metrics, binary_metrics = run_comparison()

    # Analyze results
    analyze_results(simple_metrics, binary_metrics)

    # Additional high-episode analysis
    print(f"\n" + "="*60)
    print("HIGH-EPISODE TRAINING ANALYSIS")
    print("="*60)

    simple_density = len(simple_metrics.episode_rewards) / simple_metrics.final_q_table_size()
    binary_density = len(binary_metrics.episode_rewards) / binary_metrics.final_q_table_size()

    print(f"Learning Density (episodes per state):")
    print(f"  Simple Drone: {simple_density:.1f} episodes/state")
    print(f"  Binary Drone: {binary_density:.1f} episodes/state")

    # Check convergence in final episodes
    simple_final_100 = simple_metrics.episode_rewards[-100:]
    binary_final_100 = binary_metrics.episode_rewards[-100:]

    simple_final_success = sum(1 for r in simple_final_100 if r > 500) / len(simple_final_100)
    binary_final_success = sum(1 for r in binary_final_100 if r > 500) / len(binary_final_100)

    print(f"\nFinal 100 Episodes Performance:")
    print(f"  Simple Drone Success Rate: {simple_final_success:.1%}")
    print(f"  Binary Drone Success Rate: {binary_final_success:.1%}")
    print(f"  Final Performance Gap: {binary_final_success - simple_final_success:+.1%}")

    if binary_final_success > simple_final_success:
        print(f">>> SUCCESS! Binary obstacle drone shows improvement in final episodes!")
    elif abs(binary_final_success - simple_final_success) < 0.01:
        print(f">>> Performance is equivalent - binary drone has learned to use obstacle awareness!")
    else:
        print(f">> Consider even more episodes or state representation adjustments")

    # Create plots
    plot_comparison(simple_metrics, binary_metrics)