#!/usr/bin/env python3
"""Side-by-side comparison GUI: Simple Drone vs Binary Obstacle Drone."""

import tkinter as tk
from tkinter import ttk
from typing import Dict, Tuple

from src.config import GridSettings, DEFAULT_LEARNING, DEFAULT_SIMULATION, OBSTACLE_CONFIGS
from src.trainer import DroneTrainer
from src.binary_trainer import BinaryObstacleDroneTrainer

CELL_SIZE = 20
CANVAS_MARGIN = 15
FLOOR_FILL = "#f0f4f8"
PATH_COLOR = "#ffa500"  # Orange path

class ComparisonApp(tk.Tk):
    """Side-by-side comparison of Simple vs Binary Obstacle drones."""

    def __init__(self) -> None:
        super().__init__()
        self.title("Drone Comparison: Simple vs Binary Obstacle Awareness")
        self.resizable(False, False)

        # Start with first obstacle configuration
        self.current_config_name = list(OBSTACLE_CONFIGS.keys())[0]
        grid_config = OBSTACLE_CONFIGS[self.current_config_name]

        # Create both trainers with same configuration
        self.simple_trainer = DroneTrainer(
            grid_settings=grid_config,
            learning_settings=DEFAULT_LEARNING,
            simulation_settings=DEFAULT_SIMULATION
        )

        self.binary_trainer = BinaryObstacleDroneTrainer(
            grid_settings=grid_config,
            learning_settings=DEFAULT_LEARNING,
            simulation_settings=DEFAULT_SIMULATION
        )

        # Get grid dimensions
        self.snapshot = self.simple_trainer.base_world.make_snapshot()
        self.canvas_width = self.snapshot.cols * CELL_SIZE + 2 * CANVAS_MARGIN
        self.canvas_height = self.snapshot.rows * CELL_SIZE + 2 * CANVAS_MARGIN

        # Training state
        self.running = False
        self.training_speed = 10
        self.regenerate_obstacles = False

        # Build UI
        self._build_ui()
        self._refresh_all()

        # Start update loop
        self.after(100, self._update_loop)

    def _build_ui(self) -> None:
        """Build the side-by-side comparison interface."""

        # Main container
        main_frame = ttk.Frame(self, padding=10)
        main_frame.grid(row=0, column=0, sticky="nsew")

        # Title
        title = ttk.Label(main_frame, text="Q-Learning Comparison: Simple vs Binary Obstacle Drone",
                         font=("Arial", 14, "bold"))
        title.grid(row=0, column=0, columnspan=2, pady=(0, 10))

        # Left side - Simple Drone
        left_frame = ttk.LabelFrame(main_frame, text="Simple Drone (Position Only)", padding=10)
        left_frame.grid(row=1, column=0, padx=5, sticky="nsew")

        self.simple_canvas = tk.Canvas(
            left_frame, width=self.canvas_width, height=self.canvas_height,
            bg="#f0f4f8", highlightthickness=1, highlightbackground="#ddd"
        )
        self.simple_canvas.pack()

        self.simple_stats = tk.StringVar()
        simple_stats_label = ttk.Label(left_frame, textvariable=self.simple_stats,
                                      font=("Courier", 9), justify="left")
        simple_stats_label.pack(pady=(5, 0))

        # Right side - Binary Drone
        right_frame = ttk.LabelFrame(main_frame, text="Binary Drone (Position + Obstacle Awareness)", padding=10)
        right_frame.grid(row=1, column=1, padx=5, sticky="nsew")

        self.binary_canvas = tk.Canvas(
            right_frame, width=self.canvas_width, height=self.canvas_height,
            bg="#f0f4f8", highlightthickness=1, highlightbackground="#ddd"
        )
        self.binary_canvas.pack()

        self.binary_stats = tk.StringVar()
        binary_stats_label = ttk.Label(right_frame, textvariable=self.binary_stats,
                                      font=("Courier", 9), justify="left")
        binary_stats_label.pack(pady=(5, 0))

        # Control panel at bottom
        controls = ttk.Frame(main_frame, padding=10)
        controls.grid(row=2, column=0, columnspan=2, pady=10, sticky="ew")

        # Training controls
        self.train_button = ttk.Button(controls, text="Start Training", command=self._toggle_training)
        self.train_button.grid(row=0, column=0, padx=5)

        ttk.Button(controls, text="Reset Both", command=self._reset).grid(row=0, column=1, padx=5)

        # Speed control
        ttk.Label(controls, text="Speed:").grid(row=0, column=2, padx=(20, 5))
        self.speed_var = tk.IntVar(value=10)
        speed_slider = ttk.Scale(controls, from_=1, to=50, variable=self.speed_var,
                                orient="horizontal", length=150)
        speed_slider.grid(row=0, column=3, padx=5)

        # Obstacle configuration selector
        ttk.Label(controls, text="Configuration:").grid(row=0, column=4, padx=(20, 5))
        self.config_var = tk.StringVar(value=self.current_config_name)
        config_dropdown = ttk.Combobox(controls, textvariable=self.config_var,
                                      values=list(OBSTACLE_CONFIGS.keys()),
                                      state="readonly", width=20)
        config_dropdown.grid(row=0, column=5, padx=5)

        ttk.Button(controls, text="Apply Config",
                  command=self._apply_config).grid(row=0, column=6, padx=5)

        # Goal position controls (second row)
        ttk.Label(controls, text="Goal Position:").grid(row=1, column=0, padx=(0, 5), pady=(10, 0), sticky="e")

        goal_frame = ttk.Frame(controls)
        goal_frame.grid(row=1, column=1, columnspan=2, pady=(10, 0), sticky="w")

        ttk.Label(goal_frame, text="Row:").pack(side="left", padx=(0, 5))
        self.goal_row_var = tk.IntVar(value=self.snapshot.goal[0])
        goal_row_spin = ttk.Spinbox(goal_frame, from_=0, to=self.snapshot.rows-1,
                                    textvariable=self.goal_row_var, width=5)
        goal_row_spin.pack(side="left", padx=(0, 10))

        ttk.Label(goal_frame, text="Col:").pack(side="left", padx=(0, 5))
        self.goal_col_var = tk.IntVar(value=self.snapshot.goal[1])
        goal_col_spin = ttk.Spinbox(goal_frame, from_=0, to=self.snapshot.cols-1,
                                    textvariable=self.goal_col_var, width=5)
        goal_col_spin.pack(side="left")

        ttk.Button(controls, text="Set Goal", command=self._set_goal).grid(row=1, column=3, padx=5, pady=(10, 0))

        # Regenerate obstacles checkbox
        self.regen_var = tk.BooleanVar(value=False)
        regen_check = ttk.Checkbutton(
            controls,
            text="Regenerate Obstacles Every Episode",
            variable=self.regen_var,
            command=self._toggle_regeneration
        )
        regen_check.grid(row=1, column=4, columnspan=3, pady=(10, 0), sticky="w", padx=(20, 0))

        # Comparison stats at bottom
        comparison_frame = ttk.LabelFrame(main_frame, text="Performance Comparison", padding=10)
        comparison_frame.grid(row=3, column=0, columnspan=2, sticky="ew", pady=(10, 0))

        self.comparison_stats = tk.StringVar()
        comparison_label = ttk.Label(comparison_frame, textvariable=self.comparison_stats,
                                    font=("Courier", 10, "bold"), justify="center")
        comparison_label.pack()

    def _toggle_training(self) -> None:
        """Toggle training on/off."""
        self.running = not self.running
        self.train_button.config(text="Pause Training" if self.running else "Start Training")

    def _reset(self) -> None:
        """Reset both trainers."""
        self.running = False
        self.train_button.config(text="Start Training")

        grid_config = OBSTACLE_CONFIGS[self.current_config_name]

        self.simple_trainer = DroneTrainer(
            grid_settings=grid_config,
            learning_settings=DEFAULT_LEARNING,
            simulation_settings=DEFAULT_SIMULATION
        )

        self.binary_trainer = BinaryObstacleDroneTrainer(
            grid_settings=grid_config,
            learning_settings=DEFAULT_LEARNING,
            simulation_settings=DEFAULT_SIMULATION
        )

        self.snapshot = self.simple_trainer.base_world.make_snapshot()
        self._refresh_all()

    def _apply_config(self) -> None:
        """Apply new obstacle configuration to both trainers."""
        self.current_config_name = self.config_var.get()
        self._reset()

    def _set_goal(self) -> None:
        """Update the goal position for both trainers."""
        new_row = self.goal_row_var.get()
        new_col = self.goal_col_var.get()

        # Validate goal position
        if not (0 <= new_row < self.snapshot.rows and 0 <= new_col < self.snapshot.cols):
            print(f"Invalid goal position: ({new_row}, {new_col})")
            return

        new_goal = (new_row, new_col, 0)

        # Check if goal is on an obstacle
        if new_goal in self.snapshot.obstacles:
            print(f"Cannot set goal on obstacle at ({new_row}, {new_col})")
            return

        # Update both trainers
        self.simple_trainer.set_target_position(new_goal)
        self.binary_trainer.set_target_position(new_goal)

        # Update snapshot
        self.snapshot = self.simple_trainer.base_world.make_snapshot()

        # Refresh display
        self._refresh_all()
        print(f"Goal position updated to ({new_row}, {new_col})")

    def _toggle_regeneration(self) -> None:
        """Toggle obstacle regeneration on/off."""
        self.regenerate_obstacles = self.regen_var.get()
        status = "ENABLED" if self.regenerate_obstacles else "DISABLED"
        print(f"Obstacle regeneration {status}")
        if self.regenerate_obstacles:
            print("Obstacles will regenerate every episode - Binary drone should excel!")

    def _refresh_all(self) -> None:
        """Refresh both canvases and all stats."""
        self._draw_grid(self.simple_canvas, self.simple_trainer)
        self._draw_grid(self.binary_canvas, self.binary_trainer)
        self._update_stats()

    def _draw_grid(self, canvas: tk.Canvas, trainer) -> None:
        """Draw the grid for one trainer with Q-value heatmap."""
        canvas.delete("all")

        snapshot = trainer.base_world.make_snapshot()
        drone = trainer.get_primary_drone()

        # Calculate Q-value range for heatmap coloring
        all_values = []
        if drone.q_table:
            for state_values in drone.q_table.values():
                all_values.extend(state_values)

        if all_values:
            min_val = min(all_values)
            max_val = max(all_values)
            val_range = max_val - min_val if max_val != min_val else 1.0
        else:
            min_val = max_val = val_range = 0

        # Draw grid cells with Q-value heatmap
        for row in range(snapshot.rows):
            for col in range(snapshot.cols):
                x0 = CANVAS_MARGIN + col * CELL_SIZE
                y0 = CANVAS_MARGIN + row * CELL_SIZE
                x1 = x0 + CELL_SIZE
                y1 = y0 + CELL_SIZE

                # Determine cell color
                if (row, col, 0) in snapshot.obstacles:
                    color = "#333333"
                elif (row, col, 0) == snapshot.start:
                    color = "#4CAF50"  # Green for start
                elif (row, col, 0) == snapshot.goal:
                    color = "#2196F3"  # Blue for goal
                elif all_values:
                    # Q-value heatmap: red (bad) to green (good)
                    # For binary drone, need to find best Q across all obstacle states at this position
                    best_q_for_cell = None

                    for state in drone.q_table:
                        # Check if this state matches the current cell position
                        if state[0] == row and state[1] == col:
                            cell_best = max(drone.q_table[state])
                            if best_q_for_cell is None or cell_best > best_q_for_cell:
                                best_q_for_cell = cell_best

                    if best_q_for_cell is not None:
                        normalized = (best_q_for_cell - min_val) / val_range
                        red = int(255 * (1 - normalized))
                        green = int(255 * normalized)
                        blue = 100
                        color = f"#{red:02x}{green:02x}{blue:02x}"
                    else:
                        color = FLOOR_FILL  # Unvisited
                else:
                    color = "#ffffff"

                canvas.create_rectangle(x0, y0, x1, y1, fill=color, outline="#ddd")

        # Draw orange path (greedy policy)
        try:
            path = trainer.greedy_path()
            if len(path) >= 2:
                line_coords = []
                for cell in path:
                    row, col, _ = cell
                    cx = CANVAS_MARGIN + col * CELL_SIZE + CELL_SIZE // 2
                    cy = CANVAS_MARGIN + row * CELL_SIZE + CELL_SIZE // 2
                    line_coords.extend([cx, cy])
                if len(line_coords) >= 4:
                    canvas.create_line(line_coords, fill=PATH_COLOR, width=3, capstyle="round")
        except:
            pass  # No path available yet

        # Draw drones
        for drone in trainer.drones:
            state = drone.state
            row, col = state[0], state[1]
            x = CANVAS_MARGIN + col * CELL_SIZE + CELL_SIZE // 2
            y = CANVAS_MARGIN + row * CELL_SIZE + CELL_SIZE // 2
            radius = CELL_SIZE // 3

            canvas.create_oval(
                x - radius, y - radius, x + radius, y + radius,
                fill="#FF5722", outline="#D32F2F", width=2
            )

    def _update_stats(self) -> None:
        """Update statistics for both drones and comparison."""

        # Simple drone stats
        simple_episodes = self.simple_trainer.stats.episodes_completed
        simple_states = len(self.simple_trainer.get_primary_drone().q_table)
        simple_success = (self.simple_trainer.stats.successful_episodes / simple_episodes
                         if simple_episodes > 0 else 0.0)

        recent_simple = self.simple_trainer.stats.episode_rewards[-10:] if self.simple_trainer.stats.episode_rewards else [0]
        simple_avg_reward = sum(recent_simple) / len(recent_simple)

        self.simple_stats.set(
            f"Episodes: {simple_episodes}\n"
            f"States: {simple_states}\n"
            f"Success: {simple_success:.1%}\n"
            f"Avg Reward: {simple_avg_reward:.1f}"
        )

        # Binary drone stats
        binary_episodes = self.binary_trainer.stats.episodes_completed
        binary_states = len(self.binary_trainer.get_primary_drone().q_table)
        binary_success = (self.binary_trainer.stats.successful_episodes / binary_episodes
                         if binary_episodes > 0 else 0.0)

        recent_binary = self.binary_trainer.stats.episode_rewards[-10:] if self.binary_trainer.stats.episode_rewards else [0]
        binary_avg_reward = sum(recent_binary) / len(recent_binary)

        self.binary_stats.set(
            f"Episodes: {binary_episodes}\n"
            f"States: {binary_states}\n"
            f"Success: {binary_success:.1%}\n"
            f"Avg Reward: {binary_avg_reward:.1f}"
        )

        # Comparison stats
        success_diff = binary_success - simple_success
        state_ratio = binary_states / max(simple_states, 1)

        winner = "Binary" if binary_success > simple_success else "Simple" if simple_success > binary_success else "Tied"

        self.comparison_stats.set(
            f"Configuration: {self.current_config_name} | "
            f"Success Diff: {success_diff:+.1%} | "
            f"State Space Ratio: {state_ratio:.1f}x | "
            f"Leader: {winner}"
        )

    def _update_loop(self) -> None:
        """Main training loop."""
        if self.running:
            steps = self.speed_var.get()

            # Track if we need to regenerate obstacles
            simple_prev_episodes = self.simple_trainer.stats.episodes_completed
            binary_prev_episodes = self.binary_trainer.stats.episodes_completed

            # Train both drones simultaneously
            for _ in range(steps):
                if not self.simple_trainer.is_training_complete():
                    self.simple_trainer.train_step()

                if not self.binary_trainer.is_training_complete():
                    self.binary_trainer.train_step()

            # If regenerate is enabled and episode completed, regenerate obstacles
            if self.regenerate_obstacles:
                simple_new_episodes = self.simple_trainer.stats.episodes_completed
                binary_new_episodes = self.binary_trainer.stats.episodes_completed

                # Check if simple trainer completed an episode
                if simple_new_episodes > simple_prev_episodes:
                    # Regenerate obstacles for simple trainer
                    self.simple_trainer.base_world.reset()
                    # Update main snapshot to reflect new obstacles
                    self.snapshot = self.simple_trainer.base_world.make_snapshot()
                    # Update all drone environments to use the same new obstacles
                    new_obstacles = self.simple_trainer.base_world.obstacles or set()
                    for drone in self.simple_trainer.drones:
                        drone.env.obstacles = set(new_obstacles)
                        drone.env._current_state = drone.env.settings.start
                        drone.state = (drone.env.settings.start[0], drone.env.settings.start[1])

                # Check if binary trainer completed an episode
                if binary_new_episodes > binary_prev_episodes:
                    # Regenerate obstacles for binary trainer
                    self.binary_trainer.base_world.reset()
                    # Update main snapshot to reflect new obstacles
                    self.snapshot = self.binary_trainer.base_world.make_snapshot()
                    # Update all drone environments to use the same new obstacles
                    new_obstacles = self.binary_trainer.base_world.obstacles or set()
                    for drone in self.binary_trainer.drones:
                        drone.env.obstacles = set(new_obstacles)
                        drone.env._current_state = drone.env.settings.start
                        drone.state = drone._create_state(drone.env.settings.start)

            self._refresh_all()

            # Stop if both are complete
            if (self.simple_trainer.is_training_complete() and
                self.binary_trainer.is_training_complete()):
                self.running = False
                self.train_button.config(text="Start Training")
                print("Both drones completed training!")

        self.after(50, self._update_loop)


def launch_comparison_gui() -> None:
    """Launch the comparison GUI."""
    app = ComparisonApp()
    app.mainloop()


if __name__ == "__main__":
    launch_comparison_gui()
