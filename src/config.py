"""Configuration objects for the drone Q-learning simulation."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, List, Optional

GridPoint = Tuple[int, int, int]
SimpleState = Tuple[int, int]  # Just (row, col) for 2D
BinaryObstacleState = Tuple[int, int, bool, bool, bool, bool]  # (row, col, close_N, close_S, close_W, close_E)
EnhancedState = Tuple[int, int, int, int, int, int]  # (row, col, dist_N, dist_S, dist_W, dist_E)

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
    """Parameters governing the Q-learning agents."""

    learning_rate: float = 0.7
    discount_factor: float = 0.98
    exploration_rate: float = 0.95
    exploration_decay: float = 0.998
    min_exploration: float = 0.1
    reward_step: float = -0.05
    reward_goal: float = 1000.0
    reward_obstacle: float = -100.0
    action_failure_rate: float = 0.0  # Start without stochasticity


@dataclass(frozen=True)
class EnhancedLearningSettings:
    """Enhanced parameters for complex obstacle navigation in binary obstacle drones."""

    learning_rate: float = 0.5  # Slower learning for more stable convergence
    discount_factor: float = 0.99  # Higher future reward consideration
    exploration_rate: float = 0.99  # Much higher initial exploration
    exploration_decay: float = 0.9995  # Much slower exploration decay
    min_exploration: float = 0.25  # Higher minimum exploration to keep trying new paths
    reward_step: float = -0.02  # Reduced step penalty to encourage longer paths
    reward_goal: float = 1000.0
    reward_obstacle: float = -50.0  # Reduced obstacle penalty to encourage exploration near obstacles
    action_failure_rate: float = 0.0


@dataclass(frozen=True)
class SUPEREnhancedLearningSettings:
    """SUPER aggressive parameters to force learning complex navigation!"""

    learning_rate: float = 0.3  # Much slower learning for stability
    discount_factor: float = 0.995  # Even higher future reward consideration
    exploration_rate: float = 0.999  # MAXIMUM initial exploration
    exploration_decay: float = 0.99995  # EXTREMELY slow exploration decay
    min_exploration: float = 0.4  # VERY high minimum exploration - always trying new paths
    reward_step: float = -0.005  # MINIMAL step penalty to encourage very long exploration
    reward_goal: float = 2000.0  # HIGHER goal reward
    reward_obstacle: float = -20.0  # VERY reduced obstacle penalty - barely punish hitting obstacles
    action_failure_rate: float = 0.0


@dataclass(frozen=True)
class SimulationSettings:
    """High level configuration for training runs."""

    max_steps_per_episode: int = 2500
    episodes_per_train_call: int = 1
    max_parallel_drones: int = 5
    playback_delay_ms: int = 80
    max_episodes: int = 800  # Base episodes for 2D state space (rows * cols * 2)


@dataclass(frozen=True)
class BinaryObstacleSimulationSettings:
    """Training settings optimized for binary obstacle awareness (larger state space)."""

    max_steps_per_episode: int = 2500
    episodes_per_train_call: int = 1
    max_parallel_drones: int = 5
    playback_delay_ms: int = 80
    max_episodes: int = 12800  # Proportionate to state space: rows * cols * 2^4 * 2


@dataclass(frozen=True)
class EnhancedBinaryObstacleSimulationSettings:
    """Enhanced training settings for complex obstacle navigation."""

    max_steps_per_episode: int = 5000  # Much longer episodes to allow complex pathfinding
    episodes_per_train_call: int = 1
    max_parallel_drones: int = 5
    playback_delay_ms: int = 60  # Slightly faster for more responsive training
    max_episodes: int = 20000  # More episodes for complex learning


@dataclass(frozen=True)
class SUPEREnhancedBinaryObstacleSimulationSettings:
    """SUPER enhanced settings - MAXIMUM episode length and training time!"""

    max_steps_per_episode: int = 10000  # MASSIVE episode length - 10K steps!
    episodes_per_train_call: int = 1
    max_parallel_drones: int = 5
    playback_delay_ms: int = 40  # Faster updates for quicker training
    max_episodes: int = 50000  # LOTS of episodes to really learn


# Predefined obstacle zone configurations
OBSTACLE_ZONES_CENTER_CLUSTER = [
    ObstacleZone(min_row=8, max_row=12, min_col=8, max_col=12, obstacle_density=1.0)  # Center cluster
]

OBSTACLE_ZONES_RIGHT_SIDE = [
    ObstacleZone(min_row=5, max_row=15, min_col=12, max_col=18, obstacle_density=1.0)  # Right side danger zone
]

OBSTACLE_ZONES_PATH_BLOCKER = [
    ObstacleZone(min_row=0, max_row=10, min_col=5, max_col=8, obstacle_density=1.0),   # Upper middle
    ObstacleZone(min_row=12, max_row=19, min_col=12, max_col=15, obstacle_density=1.0)  # Lower right
]

OBSTACLE_ZONES_COOL_NEW = [
    ObstacleZone(min_row=5, max_row=6, min_col=0, max_col=14, obstacle_density=0.5),   # Top right (sparse)
    ObstacleZone(min_row=13, max_row=14, min_col=10, max_col=19, obstacle_density=0.5)    # Bottom left (sparse)
]

OBSTACLE_ZONES_SPARSE_CORNERS = [
    ObstacleZone(min_row=0, max_row=5, min_col=15, max_col=19, obstacle_density=0.5),   # Top right (sparse)
    ObstacleZone(min_row=15, max_row=19, min_col=0, max_col=5, obstacle_density=0.5)    # Bottom left (sparse)
]

DEFAULT_GRID = GridSettings()
DEFAULT_LEARNING = LearningSettings()
DEFAULT_SIMULATION = SimulationSettings()
DEFAULT_BINARY_OBSTACLE_SIMULATION = BinaryObstacleSimulationSettings()

# Enhanced settings for complex navigation
ENHANCED_LEARNING = EnhancedLearningSettings()
ENHANCED_BINARY_OBSTACLE_SIMULATION = EnhancedBinaryObstacleSimulationSettings()

# SUPER enhanced settings for AGGRESSIVE learning!
SUPER_ENHANCED_LEARNING = SUPEREnhancedLearningSettings()
SUPER_ENHANCED_BINARY_OBSTACLE_SIMULATION = SUPEREnhancedBinaryObstacleSimulationSettings()

# Obstacle configuration options for GUI
OBSTACLE_CONFIGS = {
    "Default (Random)": DEFAULT_GRID,
    "Center Cluster": GridSettings(obstacle_zones=OBSTACLE_ZONES_CENTER_CLUSTER, obstacle_count=10),
    "Right Side Danger": GridSettings(obstacle_zones=OBSTACLE_ZONES_RIGHT_SIDE, obstacle_count=12),
    "Path Blockers": GridSettings(obstacle_zones=OBSTACLE_ZONES_PATH_BLOCKER, obstacle_count=27),
    "Sparse Corners": GridSettings(obstacle_zones=OBSTACLE_ZONES_SPARSE_CORNERS, obstacle_count=6)
}
