"""Configuration objects for the drone Q-learning simulation."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, List, Optional

GridPoint = Tuple[int, int, int]
SimpleState = Tuple[int, int]  # Just (row, col) for 2D
BinaryObstacleState = Tuple[int, int, bool, bool, bool, bool]  # (row, col, close_N, close_S, close_W, close_E)


@dataclass(frozen=True)
class ObstacleZone:
    """Defines a rectangular zone where obstacles can be generated."""
    min_row: int
    max_row: int
    min_col: int
    max_col: int
    obstacle_density: float = 1.0  # Fraction of obstacles to place in this zone (0.0-1.0)


@dataclass(frozen=True)
class GridSettings:
    """Static configuration for the 2D grid world."""

    rows: int = 20
    cols: int = 20
    layers: int = 1
    start: GridPoint = (0, 0, 0)
    goal: GridPoint = (0, 19, 0)
    obstacle_count: int = 7
    random_seed: int = 11
    obstacle_zones: Optional[List[ObstacleZone]] = None  # If None, use full grid


@dataclass(frozen=True)
class LearningSettings:
    """Q-learning hyperparameters optimized for complex obstacle navigation."""

    learning_rate: float = 0.3
    discount_factor: float = 0.995
    exploration_rate: float = 0.999
    exploration_decay: float = 0.99995
    min_exploration: float = 0.4
    reward_step: float = -0.005
    reward_goal: float = 2000.0
    reward_obstacle: float = -20.0
    action_failure_rate: float = 0.0


@dataclass(frozen=True)
class SimulationSettings:
    """Training simulation configuration."""

    max_steps_per_episode: int = 10000
    episodes_per_train_call: int = 1
    max_parallel_drones: int = 5
    playback_delay_ms: int = 40
    max_episodes: int = 50000


# Predefined obstacle zone configurations
OBSTACLE_ZONES_CENTER_CLUSTER = [
    ObstacleZone(min_row=8, max_row=12, min_col=8, max_col=12, obstacle_density=1.0)
]

OBSTACLE_ZONES_RIGHT_CORRIDOR = [
    ObstacleZone(min_row=5, max_row=15, min_col=12, max_col=18, obstacle_density=1.0)
]

OBSTACLE_ZONES_DUAL_BARRIERS = [
    ObstacleZone(min_row=0, max_row=10, min_col=5, max_col=8, obstacle_density=1.0),
    ObstacleZone(min_row=12, max_row=19, min_col=12, max_col=15, obstacle_density=1.0)
]

OBSTACLE_ZONES_HORIZONTAL_BANDS = [
    ObstacleZone(min_row=5, max_row=6, min_col=0, max_col=14, obstacle_density=0.5),
    ObstacleZone(min_row=13, max_row=14, min_col=10, max_col=19, obstacle_density=0.5)
]

OBSTACLE_ZONES_CORNER_SPARSE = [
    ObstacleZone(min_row=0, max_row=5, min_col=15, max_col=19, obstacle_density=0.5),
    ObstacleZone(min_row=15, max_row=19, min_col=0, max_col=5, obstacle_density=0.5)
]

# Default settings
DEFAULT_GRID = GridSettings()
DEFAULT_LEARNING = LearningSettings()
DEFAULT_SIMULATION = SimulationSettings()

# Obstacle configuration presets for GUI
OBSTACLE_CONFIGS = {
    "Center Cluster": GridSettings(obstacle_zones=OBSTACLE_ZONES_CENTER_CLUSTER, obstacle_count=10),
    "Right Corridor": GridSettings(obstacle_zones=OBSTACLE_ZONES_RIGHT_CORRIDOR, obstacle_count=12),
    "Dual Barriers": GridSettings(obstacle_zones=OBSTACLE_ZONES_DUAL_BARRIERS, obstacle_count=27),
    "Horizontal Bands": GridSettings(obstacle_zones=OBSTACLE_ZONES_HORIZONTAL_BANDS, obstacle_count=15),
    "Corner Sparse": GridSettings(obstacle_zones=OBSTACLE_ZONES_CORNER_SPARSE, obstacle_count=6),
    "Dense Random (12%)": GridSettings(obstacle_zones=None, obstacle_count=48)  # 48/400 = 12%
}
