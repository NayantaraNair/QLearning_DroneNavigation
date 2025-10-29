"""Training utilities orchestrating multiple Q-learning drones."""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Dict, List

from .config import (
    DEFAULT_GRID,
    DEFAULT_LEARNING,
    DEFAULT_SIMULATION,
    GridSettings,
    LearningSettings,
    SimulationSettings,
)
from .drone import QLearningDrone, share_q_tables
from .grid import ACTION_VECTORS, GridPoint, GridWorld


@dataclass
class TrainingStats:
    total_steps: int = 0
    episodes_completed: int = 0
    last_rewards: List[float] = field(default_factory=list)
    episode_rewards: List[float] = field(default_factory=list)
    episode_lengths: List[int] = field(default_factory=list)
    successful_episodes: int = 0
    collision_episodes: int = 0


class DroneTrainer:
    """Coordinates multiple drones learning in parallel."""

    def __init__(
        self,
        grid_settings: GridSettings = DEFAULT_GRID,
        learning_settings: LearningSettings = DEFAULT_LEARNING,
        simulation_settings: SimulationSettings = DEFAULT_SIMULATION,
    ) -> None:
        self.grid_settings = grid_settings
        self.learning_settings = learning_settings
        self.simulation_settings = simulation_settings
        self.base_world = GridWorld(settings=self.grid_settings, rewards=self.learning_settings)
        self.template_obstacles = set(self.base_world.obstacles or set())
        self.drones: List[QLearningDrone] = []
        self._steps_in_episode: List[int] = []
        self._episode_rewards: List[float] = []
        self.stats = TrainingStats(last_rewards=[0.0])
        self.configure_drones(1)

    # ------------------------------------------------------------------
    # Drone fleet management
    # ------------------------------------------------------------------
    def configure_drones(self, count: int) -> None:
        capped = max(1, min(count, self.simulation_settings.max_parallel_drones))
        if capped == len(self.drones) and self.drones:
            return
        self.drones = []
        self._steps_in_episode = []
        self._episode_rewards = []
        self.stats = TrainingStats(
            total_steps=0,
            episodes_completed=0,
            last_rewards=[0.0 for _ in range(capped)],
            successful_episodes=0,
            collision_episodes=0,
        )
        for idx in range(capped):
            world = GridWorld(
                settings=self.grid_settings,
                rewards=self.learning_settings,
                obstacles=set(self.template_obstacles),
            )
            drone = QLearningDrone(env=world, settings=self.learning_settings, rng=random.Random(idx))
            self.drones.append(drone)
            self._steps_in_episode.append(0)
            self._episode_rewards.append(0.0)

    def get_primary_drone(self) -> QLearningDrone:
        return self.drones[0]

    # ------------------------------------------------------------------
    # Training loops
    # ------------------------------------------------------------------
    def train_step(self) -> List[Dict[str, object]]:
        """Advance the simulation by one step for every drone."""
        logs: List[Dict[str, object]] = []
        for idx, drone in enumerate(self.drones):
            info = drone.step()
            self.stats.last_rewards[idx] = info["reward"]
            self._episode_rewards[idx] += info["reward"]
            self._steps_in_episode[idx] += 1
            logs.append(info)
            if info["done"] or self._steps_in_episode[idx] >= self.simulation_settings.max_steps_per_episode:
                # Record episode completion
                self.stats.episode_rewards.append(self._episode_rewards[idx])
                self.stats.episode_lengths.append(self._steps_in_episode[idx])
                self.stats.episodes_completed += 1

                # Track success/collision rates
                if info["reward"] == self.learning_settings.reward_goal:
                    self.stats.successful_episodes += 1
                elif info["reward"] == self.learning_settings.reward_obstacle:
                    self.stats.collision_episodes += 1

                # Reset for new episode
                drone.start_new_episode()
                self._steps_in_episode[idx] = 0
                self._episode_rewards[idx] = 0.0
        share_q_tables(self.drones)
        self.stats.total_steps += len(self.drones)
        return logs

    def is_training_complete(self) -> bool:
        """Check if training has reached the maximum episode limit."""
        return self.stats.episodes_completed >= self.simulation_settings.max_episodes

    def train_for_steps(self, steps: int) -> None:
        for _ in range(max(0, steps)):
            self.train_step()

    def reset(self) -> None:
        for idx, drone in enumerate(self.drones):
            drone.start_new_episode()
            self._steps_in_episode[idx] = 0
        self.stats = TrainingStats(
            total_steps=0,
            episodes_completed=0,
            last_rewards=[0.0 for _ in self.drones],
            successful_episodes=0,
            collision_episodes=0,
        )

    def set_target_position(self, new_goal: GridPoint) -> None:
        """Update the target position for all drones and reset training."""
        self.grid_settings = GridSettings(
            rows=self.grid_settings.rows,
            cols=self.grid_settings.cols,
            layers=self.grid_settings.layers,
            start=self.grid_settings.start,
            goal=new_goal,
            obstacle_count=self.grid_settings.obstacle_count,
            random_seed=self.grid_settings.random_seed,
            obstacle_zones=self.grid_settings.obstacle_zones,
        )

        # Update base world
        self.base_world = GridWorld(settings=self.grid_settings, rewards=self.learning_settings)
        self.template_obstacles = set(self.base_world.obstacles or set())

        # Update all drone environments
        for drone in self.drones:
            drone.env = GridWorld(
                settings=self.grid_settings,
                rewards=self.learning_settings,
                obstacles=set(self.template_obstacles),
            )
            drone.start_new_episode()

        # Reset stats
        self.reset()

    # ------------------------------------------------------------------
    # Convenience accessors
    # ------------------------------------------------------------------
    def greedy_path(self) -> List[GridPoint]:
        """Compute a greedy path from the primary drone's policy."""
        drone = self.get_primary_drone()
        current = self.grid_settings.start
        path: List[GridPoint] = [current]
        visited = {current}
        max_hops = self.grid_settings.rows * self.grid_settings.cols
        for _ in range(max_hops):
            if current == self.grid_settings.goal:
                break
            simple_state = (current[0], current[1])
            action = drone.best_action(simple_state)
            dr, dc, dl = ACTION_VECTORS[action]
            next_pos = (current[0] + dr, current[1] + dc, current[2] + dl)

            # Check bounds and obstacles
            if not drone.env._is_within_bounds(next_pos):
                break
            if next_pos in drone.env.obstacles:
                break
            if next_pos in visited:
                break

            current = next_pos
            path.append(current)
            visited.add(current)
        return path

    def current_primary_state(self) -> GridPoint:
        simple_state = self.get_primary_drone().state
        return (simple_state[0], simple_state[1], 0)

    def exploration_rates(self) -> List[float]:
        return [drone.epsilon for drone in self.drones]

    def print_evaluation_metrics(self) -> None:
        """Print comprehensive evaluation metrics after training completion."""
        print("\n" + "="*60)
        print("Q-LEARNING TRAINING EVALUATION METRICS")
        print("="*60)

        # Basic training stats
        print(f"TRAINING OVERVIEW:")
        print(f"   Total Episodes: {self.stats.episodes_completed}")
        print(f"   Total Steps: {self.stats.total_steps}")
        print(f"   Average Steps per Episode: {self.stats.total_steps / max(1, self.stats.episodes_completed):.1f}")

        # Learning convergence analysis
        if self.stats.episode_rewards:
            total_episodes = len(self.stats.episode_rewards)
            first_100 = self.stats.episode_rewards[:100] if total_episodes >= 100 else self.stats.episode_rewards[:total_episodes//2]
            last_100 = self.stats.episode_rewards[-100:] if total_episodes >= 100 else self.stats.episode_rewards[total_episodes//2:]

            print(f"\n LEARNING PROGRESS:")
            print(f"   Early Performance (first {len(first_100)} episodes):")
            print(f"     - Average Reward: {sum(first_100) / len(first_100):.2f}")
            print(f"     - Best Reward: {max(first_100):.2f}")
            print(f"   Final Performance (last {len(last_100)} episodes):")
            print(f"     - Average Reward: {sum(last_100) / len(last_100):.2f}")
            print(f"     - Best Reward: {max(last_100):.2f}")

            improvement = (sum(last_100) / len(last_100)) - (sum(first_100) / len(first_100))
            print(f"   Learning Improvement: {improvement:+.2f}")

        # Episode length analysis
        if self.stats.episode_lengths:
            early_lengths = self.stats.episode_lengths[:100] if len(self.stats.episode_lengths) >= 100 else self.stats.episode_lengths[:len(self.stats.episode_lengths)//2]
            late_lengths = self.stats.episode_lengths[-100:] if len(self.stats.episode_lengths) >= 100 else self.stats.episode_lengths[len(self.stats.episode_lengths)//2:]

            print(f"\n  EPISODE EFFICIENCY:")
            print(f"   Early Episodes (avg length): {sum(early_lengths) / len(early_lengths):.1f}")
            print(f"   Recent Episodes (avg length): {sum(late_lengths) / len(late_lengths):.1f}")
            efficiency_gain = (sum(early_lengths) / len(early_lengths)) - (sum(late_lengths) / len(late_lengths))
            print(f"   Efficiency Improvement: {efficiency_gain:+.1f} steps")

        # Q-table analysis
        primary_drone = self.get_primary_drone()
        q_table_size = len(primary_drone.q_table)
        total_states = self.grid_settings.rows * self.grid_settings.cols
        coverage = (q_table_size / total_states) * 100

        print(f"\n KNOWLEDGE ACQUISITION:")
        print(f"   States Explored: {q_table_size}/{total_states} ({coverage:.1f}%)")
        print(f"   Final Exploration Rate: {primary_drone.epsilon:.3f}")

        # Path quality analysis
        greedy_path = self.greedy_path()
        start_pos = self.grid_settings.start
        goal_pos = self.grid_settings.goal
        manhattan_distance = abs(goal_pos[0] - start_pos[0]) + abs(goal_pos[1] - start_pos[1])

        print(f"\n SOLUTION QUALITY:")
        print(f"   Current Path Length: {len(greedy_path)}")
        print(f"   Manhattan Distance: {manhattan_distance}")
        if len(greedy_path) > 0:
            path_efficiency = manhattan_distance / len(greedy_path) * 100
            print(f"   Path Efficiency: {path_efficiency:.1f}%")

            # Check if path reaches goal
            if greedy_path and greedy_path[-1] == goal_pos:
                print(f"   SUCCESS: Successfully finds path to goal!")
            else:
                print(f"   FAILED: Does not reach goal (may need more training)")

        # Success and collision rates
        print(f"\n SUCCESS & COLLISION ANALYSIS:")
        if self.stats.episodes_completed > 0:
            overall_success_rate = (self.stats.successful_episodes / self.stats.episodes_completed) * 100
            overall_collision_rate = (self.stats.collision_episodes / self.stats.episodes_completed) * 100
            other_episodes = self.stats.episodes_completed - self.stats.successful_episodes - self.stats.collision_episodes
            other_rate = (other_episodes / self.stats.episodes_completed) * 100

            print(f"   Overall Success Rate: {self.stats.successful_episodes}/{self.stats.episodes_completed} ({overall_success_rate:.1f}%)")
            print(f"   Overall Collision Rate: {self.stats.collision_episodes}/{self.stats.episodes_completed} ({overall_collision_rate:.1f}%)")
            print(f"   Other Outcomes (timeout/incomplete): {other_episodes}/{self.stats.episodes_completed} ({other_rate:.1f}%)")

            # Recent success/collision analysis (last 1000 episodes)
            if self.stats.episode_rewards and len(self.stats.episode_rewards) > 1000:
                recent_rewards = self.stats.episode_rewards[-1000:]
                recent_successes = sum(1 for r in recent_rewards if r == self.learning_settings.reward_goal)
                recent_collisions = sum(1 for r in recent_rewards if r == self.learning_settings.reward_obstacle)
                recent_others = 1000 - recent_successes - recent_collisions

                recent_success_rate = (recent_successes / 1000) * 100
                recent_collision_rate = (recent_collisions / 1000) * 100
                recent_other_rate = (recent_others / 1000) * 100

                print(f"\n    RECENT PERFORMANCE (last 1000 episodes):")
                print(f"     - Success Rate: {recent_successes}/1000 ({recent_success_rate:.1f}%)")
                print(f"     - Collision Rate: {recent_collisions}/1000 ({recent_collision_rate:.1f}%)")
                print(f"     - Other Rate: {recent_others}/1000 ({recent_other_rate:.1f}%)")

                # Performance trend
                if overall_success_rate > 0:
                    trend = recent_success_rate - overall_success_rate
                    if trend > 5:
                        print(f"     -  SUCCESS IMPROVING! (+{trend:.1f}%)")
                    elif trend < -5:
                        print(f"     -  Success declining ({trend:.1f}%)")
                    else:
                        print(f"     -   Success rate stable ({trend:+.1f}%)")

        print("\n" + "="*60)
        print("TRAINING EVALUATION COMPLETE")
        print("="*60)
