"""Tkinter based GUI for visualising the drone training process (2D grid view)."""
from __future__ import annotations

import tkinter as tk
from tkinter import ttk
from typing import Dict, Iterable, List, Tuple

from src.config import DEFAULT_SIMULATION, GridPoint
from src.simulation import SimulationController
from src.trainer import DroneTrainer

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


class DroneLearningApp(tk.Tk):
    """Tk application that presents the 2D grid with Q-learning visualization."""

    def __init__(self) -> None:
        super().__init__()
        self.title("Q-Learning: Watch AI Learn to Navigate!")
        self.resizable(False, False)
        self.trainer = DroneTrainer()
        self.controller = SimulationController(self.trainer, settings=DEFAULT_SIMULATION)
        self.snapshot = self.trainer.base_world.make_snapshot()
        self.canvas_width = self.snapshot.cols * CELL_SIZE + 2 * CANVAS_MARGIN
        self.canvas_height = self.snapshot.rows * CELL_SIZE + 2 * CANVAS_MARGIN
        self.stats_var = tk.StringVar()
        self.speed_var = tk.IntVar(value=10)
        self.drone_count_var = tk.IntVar(value=1)
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

        target_frame = ttk.LabelFrame(outer, text="Target Position", padding=8)
        target_frame.grid(row=3, column=1, sticky="new", pady=(12, 0))

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

        stats_frame = ttk.LabelFrame(outer, text="Stats", padding=8)
        stats_frame.grid(row=4, column=0, columnspan=2, sticky="ew", pady=(12, 0))
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

    def _set_target_position(self) -> None:
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

    # ------------------------------------------------------------------
    # Rendering helpers
    # ------------------------------------------------------------------
    def _refresh_canvas(self) -> None:
        self.canvas.delete("all")
        self.snapshot = self.trainer.get_primary_drone().env.make_snapshot()
        self._draw_q_value_heatmap()
        self._draw_obstacles()
        self._draw_marker(self.snapshot.start, fill="#c8f7c5", outline="#5f9e5f")
        self._draw_marker(self.snapshot.goal, fill="#c5d8f7", outline="#5f6e9e")
        self._draw_path(self.trainer.greedy_path())
        self._draw_drone(self.trainer.current_primary_state())

    def _draw_q_value_heatmap(self) -> None:
        """Draw cells colored by their Q-values to show learning progress."""
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

        # Draw cells with color based on best Q-value
        for row in range(self.snapshot.rows):
            for col in range(self.snapshot.cols):
                simple_state = (row, col)

                if simple_state in drone.q_table:
                    best_q = max(drone.q_table[simple_state])
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

    def _draw_drone(self, cell: GridPoint) -> None:
        row, col, _ = cell
        cx = CANVAS_MARGIN + col * CELL_SIZE + CELL_SIZE // 2
        cy = CANVAS_MARGIN + row * CELL_SIZE + CELL_SIZE // 2
        size = 12
        self.canvas.create_oval(
            cx - size, cy - size, cx + size, cy + size,
            fill=DRONE_FILL, outline=DRONE_OUTLINE, width=3
        )

    # ------------------------------------------------------------------
    # Stats and loop
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

        # Learning trend
        trend = ""
        if len(self.trainer.stats.episode_rewards) >= 2:
            recent_10 = self.trainer.stats.episode_rewards[-10:]
            older_10 = self.trainer.stats.episode_rewards[-20:-10] if len(self.trainer.stats.episode_rewards) >= 20 else []

            if older_10:
                recent_avg = sum(recent_10) / len(recent_10)
                older_avg = sum(older_10) / len(older_10)
                if recent_avg > older_avg:
                    trend = "IMPROVING!"
                elif recent_avg < older_avg:
                    trend = "Declining..."
                else:
                    trend = "Stable"

        stats = (
            f"LEARNING PROGRESS:\n"
            f"Episodes: {self.trainer.stats.episodes_completed}\n"
            f"Steps: {self.trainer.stats.total_steps}\n"
            f"States Learned: {q_table_size}\n"
            f"\nPERFORMANCE (last 10):\n"
            f"Avg Reward: {avg_recent_reward:.1f}\n"
            f"Avg Length: {avg_recent_length:.1f}\n"
            f"{trend}\n"
            f"\nExploration: {epsilons}\n"
            f"Current Reward: {rewards}\n"
            f"\n2D Grid: {self.snapshot.rows}x{self.snapshot.cols}\n"
            f"Watch colors change as AI learns!"
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


def launch_gui() -> None:
    app = DroneLearningApp()
    app.mainloop()


if __name__ == "__main__":
    launch_gui()
