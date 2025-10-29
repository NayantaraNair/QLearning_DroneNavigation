#!/usr/bin/env python3
"""Launch script for Binary Obstacle Drone GUI."""

try:
    from binary_gui import launch_binary_gui
    print("Launching Binary Obstacle Drone GUI...")
    print("Features:")
    print("- Enhanced state representation with obstacle awareness")
    print("- Visual obstacle awareness indicators around the drone")
    print("- Toggle between Q-value heatmap and obstacle awareness heatmap")
    print("- Red dots around drone indicate nearby obstacles")
    print()
    launch_binary_gui()
except ImportError as e:
    print(f"Error importing GUI components: {e}")
    print("Make sure all required files are in the same directory.")
except Exception as e:
    print(f"Error launching GUI: {e}")
    import traceback
    traceback.print_exc()