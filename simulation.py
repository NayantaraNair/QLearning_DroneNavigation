"""High level controller for coordinating training and playback."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

from config import DEFAULT_SIMULATION, SimulationSettings, GridPoint
from trainer import DroneTrainer


@dataclass
class SimulationController:
    """Wraps a trainer instance with playback friendly controls."""

    trainer: DroneTrainer
    settings: SimulationSettings = DEFAULT_SIMULATION
    steps_per_tick: int = 1
    running: bool = False

    def toggle(self) -> None:
        self.running = not self.running

    def pause(self) -> None:
        self.running = False

    def play(self) -> None:
        self.running = True

    def set_speed(self, speed_multiplier: float) -> None:
        self.steps_per_tick = max(1, int(speed_multiplier))

    def set_parallel_drones(self, count: int) -> None:
        self.trainer.configure_drones(count)
        self.trainer.reset()

    def step_once(self) -> List[Dict[str, object]]:
        return self.trainer.train_step()

    def tick(self) -> List[Dict[str, object]]:
        logs: List[Dict[str, object]] = []
        if not self.running:
            return logs
        for _ in range(self.steps_per_tick):
            logs = self.trainer.train_step()
        return logs

    def reset(self) -> None:
        self.trainer.reset()
        self.running = False
        self.steps_per_tick = 1

    def set_target_position(self, new_goal: GridPoint) -> None:
        """Update the target position for all drones."""
        self.trainer.set_target_position(new_goal)
