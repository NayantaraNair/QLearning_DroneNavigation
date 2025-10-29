"""Grid world environment used by the drone simulation."""
from __future__ import annotations

from collections import deque
import random
from dataclasses import dataclass, field
from typing import Deque, Dict, FrozenSet, Iterable, Optional, Sequence, Set, Tuple

from .config import DEFAULT_GRID, DEFAULT_LEARNING, GridPoint, GridSettings, LearningSettings, ObstacleZone

Action = int

ACTION_VECTORS: Dict[Action, Tuple[int, int, int]] = {
    0: (-1, 0, 0),  # north
    1: (1, 0, 0),   # south
    2: (0, -1, 0),  # west
    3: (0, 1, 0),   # east
}

ACTION_NAMES = {
    0: "north",
    1: "south",
    2: "west",
    3: "east",
}


@dataclass(frozen=True)
class GridSnapshot:
    """Immutable view of the grid layout for rendering."""

    rows: int
    cols: int
    layers: int
    start: GridPoint
    goal: GridPoint
    obstacles: FrozenSet[GridPoint]


@dataclass
class GridWorld:
    """Simple 3D grid with obstacles, start, and goal positions."""

    settings: GridSettings = field(default_factory=lambda: DEFAULT_GRID)
    rewards: LearningSettings = field(default_factory=lambda: DEFAULT_LEARNING)
    obstacles: Optional[Set[GridPoint]] = None
    rng: random.Random = field(default_factory=random.Random)

    def __post_init__(self) -> None:
        if self.obstacles is None:
            self.obstacles = self._generate_obstacles()
        self._current_state: GridPoint = self.settings.start

    def reset(self) -> GridPoint:
        """Reset the environment to the start state."""
        self.obstacles = self._generate_obstacles()
        self._current_state = self.settings.start
        return self._current_state

    def step(self, action: Action) -> Tuple[GridPoint, float, bool]:
        """Apply an action, returning (next_state, reward, done)."""
        if action not in ACTION_VECTORS:
            raise ValueError(f"Invalid action: {action}")

        # Stochastic action: sometimes actions fail or drift
        actual_action = action
        if self.rng.random() < self.rewards.action_failure_rate:
            # Action fails: either no movement or random drift
            if self.rng.random() < 0.5:
                # No movement (action completely fails)
                return self._current_state, self.rewards.reward_step * 2, False  # Extra penalty
            else:
                # Random drift to adjacent action
                available_actions = list(ACTION_VECTORS.keys())
                actual_action = self.rng.choice(available_actions)

        proposed = self._apply_action(self._current_state, ACTION_VECTORS[actual_action])
        reward = 0.0
        done = False

        if proposed == self.settings.goal:
            reward = self.rewards.reward_goal
            done = True
            self._current_state = proposed
        elif not self._is_within_bounds(proposed) or proposed in self.obstacles:
            reward = self.rewards.reward_obstacle
        else:
            reward = self.rewards.reward_step
            self._current_state = proposed

        return self._current_state, reward, done

    def get_actions(self) -> Sequence[Action]:
        return tuple(ACTION_VECTORS.keys())

    def get_state(self) -> GridPoint:
        return self._current_state

    def iter_states(self) -> Iterable[GridPoint]:
        for layer in range(self.settings.layers):
            for row in range(self.settings.rows):
                for col in range(self.settings.cols):
                    yield (row, col, layer)

    def make_snapshot(self) -> GridSnapshot:
        return GridSnapshot(
            rows=self.settings.rows,
            cols=self.settings.cols,
            layers=self.settings.layers,
            start=self.settings.start,
            goal=self.settings.goal,
            obstacles=frozenset(self.obstacles or set()),
        )

    def clone(self) -> "GridWorld":
        return GridWorld(
            settings=self.settings,
            rewards=self.rewards,
            obstacles=set(self.obstacles) if self.obstacles else None,
            rng=random.Random(self.rng.random()),
        )

    def _generate_obstacles(self) -> Set[GridPoint]:
        """Generate obstacles either in specific zones or across the full grid."""
        obstacles: Set[GridPoint] = set()
        attempt = 0
        while True:
            attempt += 1
            obstacles.clear()
            import time
            self.rng.seed(int(time.time() * 1000) + attempt)

            if self.settings.obstacle_zones:
                # Generate obstacles in specified zones
                obstacles = self._generate_zoned_obstacles()
            else:
                # Original full-grid generation
                obstacles = self._generate_full_grid_obstacles()

            if self._has_path(obstacles):
                return set(obstacles)

    def _generate_zoned_obstacles(self) -> Set[GridPoint]:
        """Generate obstacles within specified zones."""
        obstacles: Set[GridPoint] = set()
        total_obstacles_to_place = self.settings.obstacle_count
        obstacles_placed = 0

        # Calculate zone weights based on their area and density
        zone_weights = []
        total_weight = 0
        for zone in self.settings.obstacle_zones:
            zone_area = (zone.max_row - zone.min_row + 1) * (zone.max_col - zone.min_col + 1)
            weight = zone_area * zone.obstacle_density
            zone_weights.append(weight)
            total_weight += weight

        # Distribute obstacles proportionally to zone weights
        for i, zone in enumerate(self.settings.obstacle_zones):
            if total_weight == 0:
                zone_target = total_obstacles_to_place // len(self.settings.obstacle_zones)
            else:
                zone_target = int((zone_weights[i] / total_weight) * total_obstacles_to_place)

            # Generate obstacles for this zone
            zone_obstacles_placed = 0
            zone_attempts = 0
            max_attempts = 1000  # Prevent infinite loops

            while zone_obstacles_placed < zone_target and zone_attempts < max_attempts and obstacles_placed < total_obstacles_to_place:
                zone_attempts += 1

                point = (
                    self.rng.randrange(zone.min_row, zone.max_row + 1),
                    self.rng.randrange(zone.min_col, zone.max_col + 1),
                    self.rng.randrange(self.settings.layers),
                )

                # Skip if it's start/goal or already exists
                if point in (self.settings.start, self.settings.goal) or point in obstacles:
                    continue

                obstacles.add(point)
                zone_obstacles_placed += 1
                obstacles_placed += 1

        # If we still need more obstacles and have remaining budget, distribute them randomly across zones
        while obstacles_placed < total_obstacles_to_place:
            # Pick a random zone
            zone = self.rng.choice(self.settings.obstacle_zones)

            point = (
                self.rng.randrange(zone.min_row, zone.max_row + 1),
                self.rng.randrange(zone.min_col, zone.max_col + 1),
                self.rng.randrange(self.settings.layers),
            )

            # Skip if it's start/goal or already exists
            if point in (self.settings.start, self.settings.goal) or point in obstacles:
                continue

            obstacles.add(point)
            obstacles_placed += 1

        return obstacles

    def _generate_full_grid_obstacles(self) -> Set[GridPoint]:
        """Original full-grid obstacle generation."""
        obstacles: Set[GridPoint] = set()
        while len(obstacles) < self.settings.obstacle_count:
            point = (
                self.rng.randrange(self.settings.rows),
                self.rng.randrange(self.settings.cols),
                self.rng.randrange(self.settings.layers),
            )
            if point in (self.settings.start, self.settings.goal):
                continue
            obstacles.add(point)
        return obstacles

    def _is_in_zone(self, point: GridPoint, zone: ObstacleZone) -> bool:
        """Check if a point is within the specified zone."""
        row, col, _ = point
        return (zone.min_row <= row <= zone.max_row and
                zone.min_col <= col <= zone.max_col)

    def _has_path(self, obstacles: Set[GridPoint]) -> bool:
        """Breadth-first search to ensure the goal is reachable."""
        frontier: Deque[GridPoint] = deque([self.settings.start])
        visited: Set[GridPoint] = {self.settings.start}
        while frontier:
            row, col, layer = frontier.popleft()
            if (row, col, layer) == self.settings.goal:
                return True
            for dr, dc, dl in ACTION_VECTORS.values():
                nxt = (row + dr, col + dc, layer + dl)
                if nxt in visited:
                    continue
                if not self._is_within_bounds(nxt):
                    continue
                if nxt in obstacles:
                    continue
                frontier.append(nxt)
                visited.add(nxt)
        return False

    def _is_within_bounds(self, point: GridPoint) -> bool:
        row, col, layer = point
        return (
            0 <= row < self.settings.rows
            and 0 <= col < self.settings.cols
            and 0 <= layer < self.settings.layers
        )

    @staticmethod
    def _apply_action(point: GridPoint, delta: Tuple[int, int, int]) -> GridPoint:
        return point[0] + delta[0], point[1] + delta[1], point[2] + delta[2]

    def get_obstacle_distances(self, position: GridPoint) -> Tuple[int, ...]:
        """Get distance to nearest obstacle in each direction (N, S, W, E)."""
        distances = []
        for action_vector in ACTION_VECTORS.values():
            distance = 0
            current = position
            while True:
                distance += 1
                next_pos = self._apply_action(current, action_vector)
                if not self._is_within_bounds(next_pos) or next_pos in self.obstacles:
                    break
                current = next_pos
            distances.append(distance)
        return tuple(distances)

    def get_goal_direction(self, position: GridPoint) -> Tuple[int, int, int]:
        """Get relative vector from current position to goal (2D only)."""
        return (
            self.settings.goal[0] - position[0],
            self.settings.goal[1] - position[1],
            0  # No Z-axis in 2D
        )

    def get_binary_obstacle_awareness(self, position: GridPoint, threshold: int = 2) -> Tuple[bool, bool, bool, bool]:
        """Get binary obstacle awareness: True if obstacle within threshold steps in each direction (N, S, W, E)."""
        awareness = []
        for action_vector in ACTION_VECTORS.values():
            distance = 0
            current = position
            while distance < threshold:
                distance += 1
                next_pos = self._apply_action(current, action_vector)
                if not self._is_within_bounds(next_pos) or next_pos in self.obstacles:
                    awareness.append(True)  # Obstacle is close
                    break
                current = next_pos
            else:
                awareness.append(False)  # No obstacle within threshold
        return tuple(awareness)



def create_default_world() -> GridWorld:
    """Factory that mirrors the defaults used across the project."""
    base_world = GridWorld(settings=DEFAULT_GRID, rewards=DEFAULT_LEARNING)
    base_world.reset()
    return base_world

def clone_world(world: GridWorld) -> GridWorld:
    """Create a lightweight copy with identical configuration and layout."""
    return GridWorld(
        settings=world.settings,
        rewards=world.rewards,
        obstacles=set(world.obstacles) if world.obstacles else None,
    )
