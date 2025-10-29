#!/usr/bin/env python3
"""Example of training drones with custom obstacle zones."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.config import GridSettings, ObstacleZone, DEFAULT_LEARNING
from src.trainer import DroneTrainer
from src.binary_trainer import BinaryObstacleDroneTrainer

def train_with_center_cluster():
    """Train a drone to navigate around center-clustered obstacles."""
    print("Training with CENTER CLUSTER obstacle configuration...")

    # Create configuration with obstacles clustered in center
    center_zones = [
        ObstacleZone(min_row=8, max_row=12, min_col=8, max_col=12, obstacle_density=1.0)
    ]

    grid_config = GridSettings(
        obstacle_zones=center_zones,
        obstacle_count=10  # More obstacles in the cluster
    )

    # Train simple drone
    trainer = DroneTrainer(grid_settings=grid_config)

    print("Training simple drone for 100 episodes...")
    for episode in range(100):
        trainer.train_step()
        if trainer.is_training_complete():
            break

    trainer.print_evaluation_metrics()

    # Show learned path
    path = trainer.greedy_path()
    print(f"Learned path length: {len(path)} steps")
    print("Path avoids center cluster:",
          all(not (8 <= pos[0] <= 12 and 8 <= pos[1] <= 12) for pos in path[1:-1]))

def train_with_right_side_danger():
    """Train a binary drone to avoid right-side danger zone."""
    print("\nTraining with RIGHT SIDE DANGER ZONE...")

    # Create right-side danger zone
    danger_zones = [
        ObstacleZone(min_row=5, max_row=15, min_col=12, max_col=18, obstacle_density=1.0)
    ]

    grid_config = GridSettings(
        obstacle_zones=danger_zones,
        obstacle_count=12  # Dense danger zone
    )

    # Train binary obstacle drone (better for obstacle avoidance)
    trainer = BinaryObstacleDroneTrainer(grid_settings=grid_config)

    print("Training binary drone for 200 episodes...")
    for episode in range(200):
        trainer.train_batch(1)
        if trainer.is_training_complete():
            break

    trainer.print_evaluation_metrics()

    # Show learned path
    path = trainer.greedy_path()
    print(f"Learned path length: {len(path)} steps")

    # Check if path avoids danger zone
    danger_zone_steps = sum(1 for pos in path if 5 <= pos[0] <= 15 and 12 <= pos[1] <= 18)
    print(f"Steps through danger zone: {danger_zone_steps}/{len(path)} ({danger_zone_steps/len(path)*100:.1f}%)")

if __name__ == "__main__":
    print("EXAMPLE: Training Drones with Custom Obstacle Zones")
    print("="*60)

    # Example 1: Center cluster
    train_with_center_cluster()

    # Example 2: Right side danger zone
    train_with_right_side_danger()

    print("\n" + "="*60)
    print("USAGE SUMMARY:")
    print("1. Define ObstacleZone(min_row, max_row, min_col, max_col, density)")
    print("2. Create GridSettings(obstacle_zones=[your_zones])")
    print("3. Pass grid_settings to DroneTrainer or BinaryObstacleDroneTrainer")
    print("4. Train and observe how drones learn to avoid dense obstacle areas")