#!/usr/bin/env python3
"""Training utilities for Binary Obstacle drones (enhanced state awareness)."""

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
    GridPoint,
)
from .drone import BinaryObstacleDrone, share_binary_obstacle_q_tables
from .grid import ACTION_VECTORS, GridPoint, GridWorld


@dataclass
class BinaryTrainingStats:
    total_steps: int = 0
    episodes_completed: int = 0
    last_rewards: List[float] = field(default_factory=list)
    episode_rewards: List[float] = field(default_factory=list)
    episode_lengths: List[int] = field(default_factory=list)
    successful_episodes: int = 0
    collision_episodes: int = 0


class BinaryObstacleDroneTrainer:
    """Coordinates multiple binary obstacle drones learning in parallel."""

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
        self.drones: List[BinaryObstacleDrone] = []
        self._steps_in_episode: List[int] = []
        self._episode_rewards: List[float] = []
        self.stats = BinaryTrainingStats(last_rewards=[0.0])
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
        self.stats = BinaryTrainingStats(
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
            drone = BinaryObstacleDrone(env=world, settings=self.learning_settings, rng=random.Random(idx))
            self.drones.append(drone)
            self._steps_in_episode.append(0)
            self._episode_rewards.append(0.0)

    def get_primary_drone(self) -> BinaryObstacleDrone:
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
                if info["reward"] > 500:  # Goal reached
                    self.stats.successful_episodes += 1
                elif info["reward"] < -10:  # Hit obstacle
                    self.stats.collision_episodes += 1

                # Start new episode
                drone.start_new_episode()
                self._steps_in_episode[idx] = 0
                self._episode_rewards[idx] = 0.0

        self.stats.total_steps += len(self.drones)
        return logs

    def train_batch(self, episodes: int) -> None:
        """Train for the specified number of episodes (per drone)."""
        for episode in range(episodes):
            completed = False
            while not completed:
                logs = self.train_step()
                # Check if any drone completed an episode
                completed = any(log["done"] for log in logs)

    def share_knowledge(self) -> None:
        """Enable knowledge sharing between drones."""
        share_binary_obstacle_q_tables(self.drones)

    # ------------------------------------------------------------------
    # Policy analysis
    # ------------------------------------------------------------------
    def greedy_path(self) -> List[GridPoint]:
        """Generate path using the primary drone's greedy policy."""
        drone = self.get_primary_drone()
        world = drone.env
        path = [world.settings.start]
        current = world.settings.start
        visited = {current}
        max_steps = 100

        for _ in range(max_steps):
            # Create binary obstacle state for current position
            binary_state = drone._create_state(current)

            if binary_state in drone.q_table:
                action = drone.best_action(binary_state)
                next_pos = (
                    current[0] + ACTION_VECTORS[action][0],
                    current[1] + ACTION_VECTORS[action][1],
                    current[2] + ACTION_VECTORS[action][2],
                )
                if next_pos == world.settings.goal:
                    path.append(next_pos)
                    break
                if (world._is_within_bounds(next_pos) and
                    next_pos not in world.obstacles and
                    next_pos not in visited):
                    path.append(next_pos)
                    visited.add(next_pos)
                    current = next_pos
                else:
                    break
            else:
                break

        return path

    def current_primary_state(self) -> GridPoint:
        """Get the current position of the primary drone."""
        state = self.get_primary_drone().state
        # Binary obstacle state is (row, col, obstacle_N, obstacle_S, obstacle_W, obstacle_E)
        return (state[0], state[1], 0)  # Convert to GridPoint format

    def exploration_rates(self) -> List[float]:
        return [drone.epsilon for drone in self.drones]

    def is_training_complete(self) -> bool:
        return self.stats.episodes_completed >= self.simulation_settings.max_episodes

    def reset(self) -> None:
        """Reset all drones and stats to start fresh training."""
        self.stats = BinaryTrainingStats(
            total_steps=0,
            episodes_completed=0,
            last_rewards=[0.0 for _ in range(len(self.drones))],
            successful_episodes=0,
            collision_episodes=0,
        )
        self._steps_in_episode = [0 for _ in self.drones]
        self._episode_rewards = [0.0 for _ in self.drones]
        for drone in self.drones:
            drone.start_new_episode()

    def set_target_position(self, new_goal: GridPoint) -> None:
        """Update the target position for all binary obstacle drones and reset training."""
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

    def print_evaluation_metrics(self) -> None:
        print("\n" + "="*50)
        print("BINARY OBSTACLE DRONE TRAINING RESULTS")
        print("="*50)
        print(f"Episodes completed: {self.stats.episodes_completed}")
        print(f"Total steps: {self.stats.total_steps}")
        print(f"Success rate: {self.stats.successful_episodes / self.stats.episodes_completed:.1%}")
        print(f"Collision rate: {self.stats.collision_episodes / self.stats.episodes_completed:.1%}")

        if self.stats.episode_rewards:
            avg_reward = sum(self.stats.episode_rewards) / len(self.stats.episode_rewards)
            recent_rewards = self.stats.episode_rewards[-10:]
            recent_avg = sum(recent_rewards) / len(recent_rewards)
            print(f"Average reward: {avg_reward:.1f}")
            print(f"Recent average (last 10): {recent_avg:.1f}")

        if self.stats.episode_lengths:
            avg_length = sum(self.stats.episode_lengths) / len(self.stats.episode_lengths)
            print(f"Average episode length: {avg_length:.1f} steps")

        primary_drone = self.get_primary_drone()
        print(f"Primary drone Q-table size: {len(primary_drone.q_table)} states")
        print(f"Primary drone exploration rate: {primary_drone.epsilon:.3f}")

        # Show some example states
        if primary_drone.q_table:
            print(f"\nExample learned states:")
            for i, state in enumerate(list(primary_drone.q_table.keys())[:5]):
                q_values = primary_drone.q_table[state]
                max_q = max(q_values)
                best_action = q_values.index(max_q)
                print(f"  State {state} -> Action {best_action} (Q={max_q:.2f})")