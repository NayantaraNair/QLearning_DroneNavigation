#!/usr/bin/env python3
"""Test script for sub-grid obstacle generation."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.config import (
    GridSettings, ObstacleZone,
    OBSTACLE_ZONES_CENTER_CLUSTER,
    OBSTACLE_ZONES_RIGHT_SIDE,
    OBSTACLE_ZONES_PATH_BLOCKER,
    OBSTACLE_ZONES_SPARSE_CORNERS
)
from src.grid import GridWorld

def print_grid_with_obstacles(world: GridWorld, title: str):
    """Print a visual representation of the grid with obstacles."""
    print(f"\n{title}")
    print("="*50)

    # Create visual grid
    grid_visual = [['.' for _ in range(world.settings.cols)] for _ in range(world.settings.rows)]

    # Mark obstacles
    for obstacle in world.obstacles:
        row, col, _ = obstacle
        grid_visual[row][col] = '#'

    # Mark start and goal
    start_row, start_col, _ = world.settings.start
    goal_row, goal_col, _ = world.settings.goal
    grid_visual[start_row][start_col] = 'S'
    grid_visual[goal_row][goal_col] = 'G'

    # Print grid with coordinates
    print("   ", end="")
    for col in range(world.settings.cols):
        print(f"{col:2}", end="")
    print()

    for row in range(world.settings.rows):
        print(f"{row:2}: ", end="")
        for col in range(world.settings.cols):
            print(grid_visual[row][col] + " ", end="")
        print()

    print(f"Legend: S=Start, G=Goal, #=Obstacle, .=Free")
    print(f"Total obstacles: {len(world.obstacles)}")

def test_obstacle_zone_config(config_name: str, grid_settings: GridSettings):
    """Test a specific obstacle zone configuration."""
    world = GridWorld(settings=grid_settings)
    world.reset()  # Generate obstacles
    print_grid_with_obstacles(world, f"{config_name} Configuration")

    # Show zone information
    if grid_settings.obstacle_zones:
        print(f"\nObstacle Zones:")
        for i, zone in enumerate(grid_settings.obstacle_zones):
            obstacles_in_zone = sum(1 for obs in world.obstacles
                                   if zone.min_row <= obs[0] <= zone.max_row
                                   and zone.min_col <= obs[1] <= zone.max_col)
            print(f"  Zone {i+1}: rows {zone.min_row}-{zone.max_row}, cols {zone.min_col}-{zone.max_col}")
            print(f"           Density: {zone.obstacle_density}, Obstacles placed: {obstacles_in_zone}")

def main():
    """Test different obstacle zone configurations."""

    print("TESTING SUB-GRID OBSTACLE GENERATION")
    print("This demonstrates how obstacles can be confined to specific zones")

    # Test 1: Default (full grid)
    default_config = GridSettings()
    test_obstacle_zone_config("DEFAULT (Full Grid)", default_config)

    # Test 2: Center cluster
    center_config = GridSettings(obstacle_zones=OBSTACLE_ZONES_CENTER_CLUSTER)
    test_obstacle_zone_config("CENTER CLUSTER", center_config)

    # Test 3: Right side danger zone
    right_config = GridSettings(obstacle_zones=OBSTACLE_ZONES_RIGHT_SIDE)
    test_obstacle_zone_config("RIGHT SIDE DANGER ZONE", right_config)

    # Test 4: Path blockers
    path_blocker_config = GridSettings(obstacle_zones=OBSTACLE_ZONES_PATH_BLOCKER)
    test_obstacle_zone_config("PATH BLOCKERS", path_blocker_config)

    # Test 5: Sparse corners
    sparse_config = GridSettings(obstacle_zones=OBSTACLE_ZONES_SPARSE_CORNERS)
    test_obstacle_zone_config("SPARSE CORNERS", sparse_config)

    # Test 6: Custom configuration - diagonal barrier
    diagonal_zones = [
        ObstacleZone(min_row=5, max_row=15, min_col=5, max_col=8, obstacle_density=1.0)
    ]
    diagonal_config = GridSettings(obstacle_zones=diagonal_zones)
    test_obstacle_zone_config("DIAGONAL BARRIER", diagonal_config)

    print(f"\nSUMMARY:")
    print(f"Sub-grid obstacle generation allows you to:")
    print(f"  - Confine obstacles to specific rectangular zones")
    print(f"  - Control obstacle density per zone (0.0 to 1.0)")
    print(f"  - Create obstacle-free safe areas")
    print(f"  - Force drones to learn area avoidance strategies")
    print(f"  - Test navigation around clustered vs distributed obstacles")

if __name__ == "__main__":
    main()