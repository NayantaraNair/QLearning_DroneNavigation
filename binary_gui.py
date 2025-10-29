#!/usr/bin/env python3
"""Enhanced GUI for Binary Obstacle Drone with obstacle awareness visualization."""

import tkinter as tk
from tkinter import ttk
from typing import Dict, Iterable, List, Tuple

from src.config import (
    GridPoint, BinaryObstacleState, GridSettings,
    DEFAULT_LEARNING, DEFAULT_SIMULATION, OBSTACLE_CONFIGS
)
from src.simulation import SimulationController
from src.binary_trainer import BinaryObstacleDroneTrainer

CELL_SIZE = 25
CANVAS_MARGIN = 20
FLOOR_FILL = "#f0f4f8"
FLOOR_OUTLINE = "#cdd5dd"
OBSTACLE_FILL = "#666666"
PATH_COLOR = "#ffa500"
DRONE_FILL = "#f76e6e"
DRONE_OUTLINE = "#232323"
START_FILL = "#c8f7c5"
GOAL_FILL = "#c5d8f7"

# New colors for obstacle awareness visualization
AWARENESS_COLORS = {
    "safe": "#90EE90",      # Light green for safe directions
    "danger": "#FFB6C1",    # Light pink for dangerous directions
    "neutral": FLOOR_FILL   # Default floor color
}

# Obstacle configs imported from src.config


class BinaryObstacleDroneApp(tk.Tk):
    """Enhanced GUI for Binary Obstacle Drone with obstacle awareness indicators."""

    def __init__(self) -> None:
        super().__init__()
        self.title("Binary Obstacle Drone - Q-Learning with Obstacle Awareness")
        self.resizable(False, False)

        # Initialize with first available configuration
        self.current_config_name = list(OBSTACLE_CONFIGS.keys())[0]

        # Initialize trainer with optimized settings
        self.trainer = BinaryObstacleDroneTrainer(
            grid_settings=OBSTACLE_CONFIGS[self.current_config_name],
            learning_settings=DEFAULT_LEARNING,
            simulation_settings=DEFAULT_SIMULATION
        )
        self.controller = SimulationController(self.trainer, settings=DEFAULT_SIMULATION)
        self.snapshot = self.trainer.base_world.make_snapshot()
        self.canvas_width = self.snapshot.cols * CELL_SIZE + 2 * CANVAS_MARGIN
        self.canvas_height = self.snapshot.rows * CELL_SIZE + 2 * CANVAS_MARGIN
        self.stats_var = tk.StringVar()
        self.speed_var = tk.IntVar(value=10)
        self.drone_count_var = tk.IntVar(value=1)
        self.show_awareness_var = tk.BooleanVar(value=True)
        self.obstacle_config_var = tk.StringVar(value=self.current_config_name)
        self.target_row_var = tk.IntVar(value=self.snapshot.goal[0])
        self.target_col_var = tk.IntVar(value=self.snapshot.goal[1])
        self._build_ui()
        self._refresh_stats()
        self._refresh_canvas()
        delay = self.controller.settings.playback_delay_ms
        self.after(delay, self._update_loop)

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------
    def _build_ui(self) -> None:
        outer = ttk.Frame(self, padding=12)
        outer.grid(row=0, column=0, sticky="nsew")

        self.canvas = tk.Canvas(
            outer,
            width=self.canvas_width,
            height=self.canvas_height,
            bg="white",
            highlightthickness=1,
            highlightbackground="#b0b0b0",
        )
        self.canvas.grid(row=0, column=0, rowspan=3, padx=(0, 12))

        controls = ttk.LabelFrame(outer, text="Controls", padding=8)
        controls.grid(row=0, column=1, sticky="new")

        play_button = ttk.Button(controls, text="Play/Pause", command=self._toggle)
        play_button.grid(row=0, column=0, sticky="ew", pady=2)

        step_button = ttk.Button(controls, text="Step Once", command=self._step_once)
        step_button.grid(row=1, column=0, sticky="ew", pady=2)

        reset_button = ttk.Button(controls, text="Reset", command=self._reset)
        reset_button.grid(row=2, column=0, sticky="ew", pady=2)

        # Add obstacle awareness toggle
        awareness_check = ttk.Checkbutton(
            controls,
            text="Show Obstacle Awareness",
            variable=self.show_awareness_var,
            command=self._refresh_canvas
        )
        awareness_check.grid(row=3, column=0, sticky="ew", pady=2)

        # Add obstacle configuration selector
        config_frame = ttk.LabelFrame(outer, text="Obstacle Configuration", padding=8)
        config_frame.grid(row=3, column=1, sticky="new", pady=(12, 0))

        config_combo = ttk.Combobox(
            config_frame,
            textvariable=self.obstacle_config_var,
            values=list(OBSTACLE_CONFIGS.keys()),
            state="readonly",
            width=18
        )
        config_combo.grid(row=0, column=0, sticky="ew", pady=2)
        config_combo.bind("<<ComboboxSelected>>", self._on_config_change)

        reload_button = ttk.Button(
            config_frame,
            text="Apply Configuration",
            command=self._reload_environment
        )
        reload_button.grid(row=1, column=0, sticky="ew", pady=2)

        speed_frame = ttk.LabelFrame(outer, text="Speed (steps/tick)", padding=8)
        speed_frame.grid(row=1, column=1, sticky="new", pady=(12, 0))
        speed_scale = ttk.Scale(
            speed_frame,
            from_=1,
            to=60,
            orient="horizontal",
            command=self._on_speed_change,
            length=180,
        )
        speed_scale.set(self.speed_var.get())
        speed_scale.grid(row=0, column=0, sticky="ew")

        drone_frame = ttk.LabelFrame(outer, text="Parallel drones", padding=8)
        drone_frame.grid(row=2, column=1, sticky="new", pady=(12, 0))
        drone_spin = tk.Spinbox(
            drone_frame,
            from_=1,
            to=self.controller.settings.max_parallel_drones,
            textvariable=self.drone_count_var,
            command=self._on_drone_count_change,
            width=5,
        )
        drone_spin.grid(row=0, column=0, sticky="ew")
        self.drone_count_var.trace_add("write", lambda *_: self._on_drone_count_change())

        # Target position controls
        target_frame = ttk.LabelFrame(outer, text="Target Position", padding=8)
        target_frame.grid(row=4, column=1, sticky="new", pady=(12, 0))

        ttk.Label(target_frame, text="Row:").grid(row=0, column=0, sticky="w", padx=(0, 5))
        target_row_spin = tk.Spinbox(
            target_frame,
            from_=0,
            to=self.snapshot.rows-1,
            textvariable=self.target_row_var,
            width=5,
        )
        target_row_spin.grid(row=0, column=1, sticky="ew", padx=(0, 10))

        ttk.Label(target_frame, text="Col:").grid(row=0, column=2, sticky="w", padx=(0, 5))
        target_col_spin = tk.Spinbox(
            target_frame,
            from_=0,
            to=self.snapshot.cols-1,
            textvariable=self.target_col_var,
            width=5,
        )
        target_col_spin.grid(row=0, column=3, sticky="ew")

        set_target_button = ttk.Button(target_frame, text="Set Target", command=self._set_target_position)
        set_target_button.grid(row=1, column=0, columnspan=4, sticky="ew", pady=(8, 0))

        # Enhanced stats for binary obstacle drone
        stats_frame = ttk.LabelFrame(outer, text="Stats", padding=8)
        stats_frame.grid(row=5, column=0, columnspan=2, sticky="ew", pady=(12, 0))
        stats_label = ttk.Label(stats_frame, textvariable=self.stats_var, justify="left")
        stats_label.grid(row=0, column=0, sticky="w")

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------
    def _toggle(self) -> None:
        self.controller.toggle()

    def _on_speed_change(self, value: str) -> None:
        try:
            speed = float(value)
        except ValueError:
            return
        self.speed_var.set(int(speed))
        self.controller.set_speed(speed)

    def _on_drone_count_change(self) -> None:
        try:
            count = int(self.drone_count_var.get())
        except (tk.TclError, ValueError):
            return
        self.controller.set_parallel_drones(count)
        self._refresh_stats()
        self._refresh_canvas()

    def _step_once(self) -> None:
        self.controller.step_once()
        self._refresh_canvas()
        self._refresh_stats()

    def _reset(self) -> None:
        self.controller.reset()
        self._refresh_canvas()
        self._refresh_stats()

    def _on_config_change(self, event=None) -> None:
        """Handle obstacle configuration change selection."""
        self.current_config_name = self.obstacle_config_var.get()
        print(f"Selected configuration: {self.current_config_name}")

    def _reload_environment(self) -> None:
        """Reload the training environment with the selected obstacle configuration."""
        print(f"Reloading environment with configuration: {self.current_config_name}")

        # Get the selected configuration
        grid_config = OBSTACLE_CONFIGS[self.current_config_name]

        # Pause training if running
        was_running = getattr(self.controller, 'is_running', False)
        if was_running:
            try:
                self.controller.pause()
            except AttributeError:
                pass  # Controller doesn't have pause method

        # Create new trainer with selected configuration
        self.trainer = BinaryObstacleDroneTrainer(grid_settings=grid_config)

        # Update controller with new trainer
        self.controller.trainer = self.trainer

        # Reset controller state
        self.controller.reset()

        # Update snapshot
        self.snapshot = self.trainer.base_world.make_snapshot()

        # Refresh display
        self._refresh_canvas()
        self._refresh_stats()

        # Resume if it was running before
        if was_running:
            try:
                self.controller.resume()
            except AttributeError:
                pass  # Controller doesn't have resume method

        print(f"Environment reloaded successfully!")

    def _set_target_position(self) -> None:
        """Set a new target position for the binary obstacle drone."""
        try:
            new_row = int(self.target_row_var.get())
            new_col = int(self.target_col_var.get())
        except (tk.TclError, ValueError):
            return

        # Clamp values to valid range
        new_row = max(0, min(new_row, self.snapshot.rows - 1))
        new_col = max(0, min(new_col, self.snapshot.cols - 1))

        # Update the variables to reflect clamped values
        self.target_row_var.set(new_row)
        self.target_col_var.set(new_col)

        # Set the new target position
        new_goal = (new_row, new_col, 0)
        self.controller.set_target_position(new_goal)

        # Update the snapshot to reflect the new goal
        self.snapshot = self.trainer.base_world.make_snapshot()

        self._refresh_canvas()
        self._refresh_stats()
        print(f"Target position updated to ({new_row}, {new_col})")

    # ------------------------------------------------------------------
    # Enhanced rendering with obstacle awareness
    # ------------------------------------------------------------------
    def _refresh_canvas(self) -> None:
        self.canvas.delete("all")
        self.snapshot = self.trainer.get_primary_drone().env.make_snapshot()

        if self.show_awareness_var.get():
            self._draw_obstacle_awareness_heatmap()
        else:
            self._draw_q_value_heatmap()

        self._draw_obstacles()
        self._draw_marker(self.snapshot.start, fill="#c8f7c5", outline="#5f9e5f")
        self._draw_marker(self.snapshot.goal, fill="#c5d8f7", outline="#5f6e9e")
        self._draw_path(self.trainer.greedy_path())
        self._draw_drone_with_awareness(self.trainer.current_primary_state())

    def _draw_obstacle_awareness_heatmap(self) -> None:
        """Draw cells colored by obstacle awareness from current drone position."""
        drone = self.trainer.get_primary_drone()
        current_pos = (drone.state[0], drone.state[1], 0)  # Convert to GridPoint

        for row in range(self.snapshot.rows):
            for col in range(self.snapshot.cols):
                position = (row, col, 0)

                # Get obstacle awareness for this position
                awareness = drone.env.get_binary_obstacle_awareness(position)
                # awareness is (close_N, close_S, close_W, close_E)

                # Color based on danger level
                danger_count = sum(awareness)
                if danger_count == 0:
                    color = AWARENESS_COLORS["safe"]  # Safe in all directions
                elif danger_count >= 3:
                    color = AWARENESS_COLORS["danger"]  # Dangerous (trapped)
                else:
                    # Intermediate danger - blend colors
                    intensity = danger_count / 4.0
                    red = int(255 * (0.96 + 0.04 * intensity))  # Light pink base
                    green = int(255 * (0.94 - 0.3 * intensity))
                    blue = int(255 * (0.89 - 0.2 * intensity))
                    color = f"#{red:02x}{green:02x}{blue:02x}"

                # Highlight current position
                if position[:2] == current_pos[:2]:
                    color = "#FFD700"  # Gold for current position

                x1 = CANVAS_MARGIN + col * CELL_SIZE
                y1 = CANVAS_MARGIN + row * CELL_SIZE
                x2 = x1 + CELL_SIZE
                y2 = y1 + CELL_SIZE

                self.canvas.create_rectangle(x1, y1, x2, y2, fill=color, outline=FLOOR_OUTLINE)

    def _draw_q_value_heatmap(self) -> None:
        """Draw traditional Q-value heatmap (adapted for binary obstacle states)."""
        drone = self.trainer.get_primary_drone()

        # Calculate value range for color mapping
        if not drone.q_table:
            self._draw_floor_tiles()
            return

        all_values = []
        for state_values in drone.q_table.values():
            all_values.extend(state_values)

        if not all_values:
            self._draw_floor_tiles()
            return

        min_val = min(all_values)
        max_val = max(all_values)
        val_range = max_val - min_val if max_val != min_val else 1.0

        # Draw cells with color based on best Q-value for position
        position_q_values = {}  # Map (row, col) to best Q-value

        for state, q_values in drone.q_table.items():
            row, col = state[0], state[1]  # Extract position from binary obstacle state
            best_q = max(q_values)

            if (row, col) not in position_q_values or best_q > position_q_values[(row, col)]:
                position_q_values[(row, col)] = best_q

        for row in range(self.snapshot.rows):
            for col in range(self.snapshot.cols):
                if (row, col) in position_q_values:
                    best_q = position_q_values[(row, col)]
                    normalized = (best_q - min_val) / val_range
                    # Color from red (bad) to green (good)
                    red = int(255 * (1 - normalized))
                    green = int(255 * normalized)
                    blue = 100
                    color = f"#{red:02x}{green:02x}{blue:02x}"
                else:
                    color = FLOOR_FILL  # Unvisited

                x1 = CANVAS_MARGIN + col * CELL_SIZE
                y1 = CANVAS_MARGIN + row * CELL_SIZE
                x2 = x1 + CELL_SIZE
                y2 = y1 + CELL_SIZE

                self.canvas.create_rectangle(x1, y1, x2, y2, fill=color, outline=FLOOR_OUTLINE)

    def _draw_floor_tiles(self) -> None:
        for row in range(self.snapshot.rows):
            for col in range(self.snapshot.cols):
                x1 = CANVAS_MARGIN + col * CELL_SIZE
                y1 = CANVAS_MARGIN + row * CELL_SIZE
                x2 = x1 + CELL_SIZE
                y2 = y1 + CELL_SIZE
                self.canvas.create_rectangle(x1, y1, x2, y2, fill=FLOOR_FILL, outline=FLOOR_OUTLINE)

    def _draw_obstacles(self) -> None:
        for obstacle in self.snapshot.obstacles:
            row, col, _ = obstacle
            x1 = CANVAS_MARGIN + col * CELL_SIZE
            y1 = CANVAS_MARGIN + row * CELL_SIZE
            x2 = x1 + CELL_SIZE
            y2 = y1 + CELL_SIZE
            self.canvas.create_rectangle(x1, y1, x2, y2, fill=OBSTACLE_FILL, outline="black", width=2)

    def _draw_marker(self, cell: GridPoint, *, fill: str, outline: str) -> None:
        row, col, _ = cell
        x1 = CANVAS_MARGIN + col * CELL_SIZE + 5
        y1 = CANVAS_MARGIN + row * CELL_SIZE + 5
        x2 = x1 + CELL_SIZE - 10
        y2 = y1 + CELL_SIZE - 10
        self.canvas.create_rectangle(x1, y1, x2, y2, fill=fill, outline=outline, width=3)

    def _draw_path(self, path: List[GridPoint]) -> None:
        if len(path) < 2:
            return
        line_coords = []
        for cell in path:
            row, col, _ = cell
            cx = CANVAS_MARGIN + col * CELL_SIZE + CELL_SIZE // 2
            cy = CANVAS_MARGIN + row * CELL_SIZE + CELL_SIZE // 2
            line_coords.extend([cx, cy])
        if len(line_coords) >= 4:
            self.canvas.create_line(line_coords, fill=PATH_COLOR, width=4, capstyle="round")

    def _draw_drone_with_awareness(self, cell: GridPoint) -> None:
        """Draw drone with obstacle awareness indicators."""
        row, col, _ = cell
        cx = CANVAS_MARGIN + col * CELL_SIZE + CELL_SIZE // 2
        cy = CANVAS_MARGIN + row * CELL_SIZE + CELL_SIZE // 2

        # Draw main drone
        size = 12
        self.canvas.create_oval(
            cx - size, cy - size, cx + size, cy + size,
            fill=DRONE_FILL, outline=DRONE_OUTLINE, width=3
        )

        # Draw obstacle awareness indicators if enabled
        if self.show_awareness_var.get():
            drone = self.trainer.get_primary_drone()
            awareness = drone.env.get_binary_obstacle_awareness((row, col, 0))
            # awareness is (close_N, close_S, close_W, close_E)

            indicator_size = 4
            indicator_distance = 18

            # North indicator
            if awareness[0]:  # Obstacle to north
                self.canvas.create_oval(
                    cx - indicator_size, cy - indicator_distance - indicator_size,
                    cx + indicator_size, cy - indicator_distance + indicator_size,
                    fill="red", outline="darkred"
                )

            # South indicator
            if awareness[1]:  # Obstacle to south
                self.canvas.create_oval(
                    cx - indicator_size, cy + indicator_distance - indicator_size,
                    cx + indicator_size, cy + indicator_distance + indicator_size,
                    fill="red", outline="darkred"
                )

            # West indicator
            if awareness[2]:  # Obstacle to west
                self.canvas.create_oval(
                    cx - indicator_distance - indicator_size, cy - indicator_size,
                    cx - indicator_distance + indicator_size, cy + indicator_size,
                    fill="red", outline="darkred"
                )

            # East indicator
            if awareness[3]:  # Obstacle to east
                self.canvas.create_oval(
                    cx + indicator_distance - indicator_size, cy - indicator_size,
                    cx + indicator_distance + indicator_size, cy + indicator_size,
                    fill="red", outline="darkred"
                )

    # ------------------------------------------------------------------
    # Stats display
    # ------------------------------------------------------------------
    def _refresh_stats(self) -> None:
        epsilons = ", ".join(f"{e:.2f}" for e in self.trainer.exploration_rates())
        rewards = ", ".join(f"{r:.2f}" for r in self.trainer.stats.last_rewards)

        # Learning progress metrics
        recent_episodes = self.trainer.stats.episode_rewards[-10:] if self.trainer.stats.episode_rewards else []
        avg_recent_reward = sum(recent_episodes) / len(recent_episodes) if recent_episodes else 0.0

        recent_lengths = self.trainer.stats.episode_lengths[-10:] if self.trainer.stats.episode_lengths else []
        avg_recent_length = sum(recent_lengths) / len(recent_lengths) if recent_lengths else 0.0

        q_table_size = len(self.trainer.get_primary_drone().q_table)

        # Success and collision rates
        total_episodes = len(self.trainer.stats.episode_rewards)
        success_rate = (self.trainer.stats.successful_episodes / total_episodes) if total_episodes > 0 else 0.0
        collision_rate = (self.trainer.stats.collision_episodes / total_episodes) if total_episodes > 0 else 0.0

        # Learning trend
        trend = ""
        if len(self.trainer.stats.episode_rewards) >= 2:
            recent_10 = self.trainer.stats.episode_rewards[-10:]
            older_10 = self.trainer.stats.episode_rewards[-20:-10] if len(self.trainer.stats.episode_rewards) >= 20 else []

            if older_10:
                recent_avg = sum(recent_10) / len(recent_10)
                older_avg = sum(older_10) / len(older_10)
                if recent_avg > older_avg:
                    trend = "Improving"
                elif recent_avg < older_avg:
                    trend = "Declining"
                else:
                    trend = "Stable"

        # Current drone state details
        drone = self.trainer.get_primary_drone()
        current_state = drone.state
        obstacle_info = f"N={current_state[2]}, S={current_state[3]}, W={current_state[4]}, E={current_state[5]}"

        # Get obstacle configuration info
        config_info = self.current_config_name
        if self.trainer.grid_settings.obstacle_zones:
            zone_count = len(self.trainer.grid_settings.obstacle_zones)
            config_info += f" ({zone_count} zones)"

        stats = (
            f"Configuration: {config_info}\n"
            f"Obstacles: {len(self.snapshot.obstacles)}\n"
            f"Grid: {self.snapshot.rows}x{self.snapshot.cols}\n"
            f"\nTraining Progress:\n"
            f"Episodes: {self.trainer.stats.episodes_completed}/{self.trainer.simulation_settings.max_episodes}\n"
            f"Total Steps: {self.trainer.stats.total_steps}\n"
            f"States Learned: {q_table_size}\n"
            f"\nPerformance (last 10 episodes):\n"
            f"Success Rate: {success_rate:.1%}\n"
            f"Collision Rate: {collision_rate:.1%}\n"
            f"Avg Reward: {avg_recent_reward:.1f}\n"
            f"Avg Length: {avg_recent_length:.1f} steps\n"
            f"Trend: {trend}\n"
            f"\nCurrent State:\n"
            f"Obstacle Awareness: {obstacle_info}\n"
            f"Exploration Rate: {epsilons}\n"
            f"Last Reward: {rewards}"
        )
        self.stats_var.set(stats)

    def _update_loop(self) -> None:
        if not self.trainer.is_training_complete():
            self.controller.tick()
            self._refresh_canvas()
            self._refresh_stats()
            delay = self.controller.settings.playback_delay_ms
            self.after(delay, self._update_loop)
        else:
            print(f"Training complete! Reached {self.trainer.simulation_settings.max_episodes} episodes.")
            self.trainer.print_evaluation_metrics()
            self._refresh_stats()


def launch_binary_gui() -> None:
    """Launch the Binary Obstacle Drone GUI."""
    app = BinaryObstacleDroneApp()
    app.mainloop()


if __name__ == "__main__":
    launch_binary_gui()