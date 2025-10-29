"""Q-learning based drone agent implementation."""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Tuple

from config import DEFAULT_LEARNING, LearningSettings, SimpleState, BinaryObstacleState
from grid import Action, GridPoint, GridWorld

QTable = Dict[SimpleState, List[float]]
BinaryObstacleQTable = Dict[BinaryObstacleState, List[float]]


@dataclass
class QLearningDrone:
    """Encapsulates Q-learning behaviour for navigating the grid."""

    env: GridWorld
    settings: LearningSettings = field(default_factory=lambda: DEFAULT_LEARNING)
    rng: random.Random = field(default_factory=random.Random)

    def __post_init__(self) -> None:
        self.epsilon: float = self.settings.exploration_rate
        self.q_table: QTable = {}
        position = self.env.reset()
        self.state: SimpleState = (position[0], position[1])  # Just row, col

    # ------------------------------------------------------------------
    # Interaction with environment
    # ------------------------------------------------------------------
    def start_new_episode(self) -> SimpleState:
        position = self.env.reset()
        self.state = (position[0], position[1])
        self.decay_exploration()
        return self.state

    def select_action(self, state: SimpleState) -> Action:
        self._ensure_state_exists(state)
        if self.rng.random() < self.epsilon:
            return self.rng.choice(self._available_actions())
        return self._greedy_action(state)

    def best_action(self, state: SimpleState) -> Action:
        self._ensure_state_exists(state)
        return self._greedy_action(state)

    def step(self) -> Dict[str, object]:
        """Perform a single Q-learning step and return diagnostics."""
        action = self.select_action(self.state)
        next_position, reward, done = self.env.step(action)
        next_state = (next_position[0], next_position[1])
        self._ensure_state_exists(next_state)
        self._learn(self.state, action, reward, next_state, done)
        diagnostics = {
            "state": self.state,
            "action": action,
            "next_state": next_state,
            "reward": reward,
            "done": done,
        }
        self.state = next_state
        return diagnostics

    def decay_exploration(self) -> None:
        self.epsilon = max(
            self.settings.min_exploration,
            self.epsilon * self.settings.exploration_decay,
        )

    # ------------------------------------------------------------------
    # Policy inspection helpers
    # ------------------------------------------------------------------
    def greedy_policy(self) -> Dict[SimpleState, Action]:
        policy: Dict[SimpleState, Action] = {}
        for state in self.q_table:
            policy[state] = self._greedy_action(state)
        return policy

    def get_state_value(self, state: SimpleState) -> float:
        self._ensure_state_exists(state)
        return max(self.q_table[state])

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _greedy_action(self, state: SimpleState) -> Action:
        q_values = self.q_table[state]
        max_value = max(q_values)
        greedy_actions = [a for a, value in enumerate(q_values) if value == max_value]
        return self.rng.choice(greedy_actions)

    def _learn(
        self,
        state: SimpleState,
        action: Action,
        reward: float,
        next_state: SimpleState,
        done: bool,
    ) -> None:
        old_value = self.q_table[state][action]
        next_max = 0.0 if done else max(self.q_table[next_state])
        updated = (1 - self.settings.learning_rate) * old_value + self.settings.learning_rate * (
            reward + self.settings.discount_factor * next_max
        )
        self.q_table[state][action] = updated

    def _ensure_state_exists(self, state: SimpleState) -> None:
        if state not in self.q_table:
            self.q_table[state] = [0.0 for _ in self._available_actions()]

    def _available_actions(self) -> Sequence[Action]:
        return self.env.get_actions()


def share_q_tables(drones: Sequence[QLearningDrone]) -> None:
    """Average Q-tables across drones to encourage shared learning."""
    if not drones:
        return
    aggregate: Dict[SimpleState, List[float]] = {}
    for drone in drones:
        for state, values in drone.q_table.items():
            if state not in aggregate:
                aggregate[state] = [0.0] * len(values)
            for idx, value in enumerate(values):
                aggregate[state][idx] += value
    for state_values in aggregate.values():
        for idx in range(len(state_values)):
            state_values[idx] /= len(drones)
    for drone in drones:
        for state, values in aggregate.items():
            drone.q_table[state] = list(values)


@dataclass
class BinaryObstacleDrone:
    """Enhanced Q-learning drone with binary obstacle awareness in state representation."""

    env: GridWorld
    settings: LearningSettings = field(default_factory=lambda: DEFAULT_LEARNING)
    rng: random.Random = field(default_factory=random.Random)

    def __post_init__(self) -> None:
        self.epsilon: float = self.settings.exploration_rate
        self.q_table: BinaryObstacleQTable = {}
        position = self.env.reset()
        self.state: BinaryObstacleState = self._create_state(position)

    # ------------------------------------------------------------------
    # Interaction with environment
    # ------------------------------------------------------------------
    def start_new_episode(self) -> BinaryObstacleState:
        position = self.env.reset()
        self.state = self._create_state(position)
        self.decay_exploration()
        return self.state

    def select_action(self, state: BinaryObstacleState) -> Action:
        self._ensure_state_exists(state)
        if self.rng.random() < self.epsilon:
            return self.rng.choice(self._available_actions())
        return self._greedy_action(state)

    def best_action(self, state: BinaryObstacleState) -> Action:
        self._ensure_state_exists(state)
        return self._greedy_action(state)

    def step(self) -> Dict[str, object]:
        """Perform a single Q-learning step and return diagnostics."""
        action = self.select_action(self.state)
        next_position, reward, done = self.env.step(action)
        next_state = self._create_state(next_position)
        self._ensure_state_exists(next_state)
        self._learn(self.state, action, reward, next_state, done)
        diagnostics = {
            "state": self.state,
            "action": action,
            "next_state": next_state,
            "reward": reward,
            "done": done,
        }
        self.state = next_state
        return diagnostics

    def decay_exploration(self) -> None:
        self.epsilon = max(
            self.settings.min_exploration,
            self.epsilon * self.settings.exploration_decay,
        )

    # ------------------------------------------------------------------
    # Policy inspection helpers
    # ------------------------------------------------------------------
    def greedy_policy(self) -> Dict[BinaryObstacleState, Action]:
        policy: Dict[BinaryObstacleState, Action] = {}
        for state in self.q_table:
            policy[state] = self._greedy_action(state)
        return policy

    def get_state_value(self, state: BinaryObstacleState) -> float:
        self._ensure_state_exists(state)
        return max(self.q_table[state])

    # ------------------------------------------------------------------
    # Binary obstacle state creation
    # ------------------------------------------------------------------
    def _create_state(self, position: GridPoint) -> BinaryObstacleState:
        """Create binary obstacle state representation for given position."""
        row, col, _ = position
        # Get obstacle awareness (close_N, close_S, close_W, close_E)
        awareness = self.env.get_binary_obstacle_awareness(position)
        return (row, col, awareness[0], awareness[1], awareness[2], awareness[3])

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _greedy_action(self, state: BinaryObstacleState) -> Action:
        q_values = self.q_table[state]
        max_value = max(q_values)
        greedy_actions = [a for a, value in enumerate(q_values) if value == max_value]
        return self.rng.choice(greedy_actions)

    def _learn(
        self,
        state: BinaryObstacleState,
        action: Action,
        reward: float,
        next_state: BinaryObstacleState,
        done: bool,
    ) -> None:
        old_value = self.q_table[state][action]
        next_max = 0.0 if done else max(self.q_table[next_state])
        updated = (1 - self.settings.learning_rate) * old_value + self.settings.learning_rate * (
            reward + self.settings.discount_factor * next_max
        )
        self.q_table[state][action] = updated

    def _ensure_state_exists(self, state: BinaryObstacleState) -> None:
        if state not in self.q_table:
            self.q_table[state] = [0.0 for _ in self._available_actions()]

    def _available_actions(self) -> Sequence[Action]:
        return self.env.get_actions()


def share_binary_obstacle_q_tables(drones: Sequence[BinaryObstacleDrone]) -> None:
    """Average Q-tables across binary obstacle drones to encourage shared learning."""
    if not drones:
        return
    aggregate: Dict[BinaryObstacleState, List[float]] = {}
    for drone in drones:
        for state, values in drone.q_table.items():
            if state not in aggregate:
                aggregate[state] = [0.0] * len(values)
            for idx, value in enumerate(values):
                aggregate[state][idx] += value
    for state_values in aggregate.values():
        for idx in range(len(state_values)):
            state_values[idx] /= len(drones)
    for drone in drones:
        for state, values in aggregate.items():
            drone.q_table[state] = list(values)
