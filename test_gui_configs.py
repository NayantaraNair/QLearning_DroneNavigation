#!/usr/bin/env python3
"""Test script to demonstrate the obstacle configuration switcher functionality."""

from config import OBSTACLE_CONFIGS
from binary_trainer import BinaryObstacleDroneTrainer

def test_all_configurations():
    """Test that all obstacle configurations can be loaded successfully."""
    print("Testing all obstacle configurations...\n")

    for config_name, grid_settings in OBSTACLE_CONFIGS.items():
        print(f"Testing configuration: {config_name}")
        print("-" * 50)

        try:
            # Create trainer with configuration
            trainer = BinaryObstacleDroneTrainer(grid_settings=grid_settings)

            # Get world snapshot
            snapshot = trainer.base_world.make_snapshot()

            # Display configuration info
            print(f"  Grid size: {snapshot.rows}x{snapshot.cols}")
            print(f"  Start: {snapshot.start}")
            print(f"  Goal: {snapshot.goal}")
            print(f"  Total obstacles: {len(snapshot.obstacles)}")

            if grid_settings.obstacle_zones:
                print(f"  Obstacle zones: {len(grid_settings.obstacle_zones)}")
                for i, zone in enumerate(grid_settings.obstacle_zones):
                    obstacles_in_zone = sum(1 for obs in snapshot.obstacles
                                           if zone.min_row <= obs[0] <= zone.max_row
                                           and zone.min_col <= obs[1] <= zone.max_col)
                    print(f"    Zone {i+1}: rows {zone.min_row}-{zone.max_row}, "
                          f"cols {zone.min_col}-{zone.max_col}, "
                          f"density {zone.obstacle_density}, "
                          f"obstacles: {obstacles_in_zone}")
            else:
                print(f"  Full grid random placement")

            # Test path finding
            path = trainer.greedy_path()
            print(f"  Initial path length: {len(path)} steps")

            print(f"  [SUCCESS] Configuration loaded successfully!")

        except Exception as e:
            print(f"  [ERROR] Error loading configuration: {e}")

        print()

def main():
    print("OBSTACLE CONFIGURATION SWITCHER TEST")
    print("="*60)
    print("This script tests all pre-built obstacle configurations")
    print("that are available in the binary GUI switcher.\n")

    test_all_configurations()

    print("GUI USAGE INSTRUCTIONS:")
    print("="*60)
    print("1. Run: python binary_gui.py")
    print("2. Use the 'Obstacle Configuration' dropdown to select different patterns")
    print("3. Click 'Apply Configuration' to reload the environment")
    print("4. Toggle 'Show Obstacle Awareness' to see different visualizations")
    print("5. Watch how the binary drone adapts to different obstacle patterns")
    print("\nAvailable configurations:")
    for config_name in OBSTACLE_CONFIGS.keys():
        print(f"  - {config_name}")

if __name__ == "__main__":
    main()