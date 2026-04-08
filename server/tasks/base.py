"""Abstract base class for all PipelineRx tasks."""

from __future__ import annotations

import abc
from typing import Any, Dict

import pandas as pd


class BaseTask(abc.ABC):
    """Every task must implement these methods."""

    name: str = ""
    description: str = ""
    difficulty: str = "easy"
    max_steps: int = 8

    @abc.abstractmethod
    def generate_data(self) -> pd.DataFrame:
        """Return the *corrupted* DataFrame for this task (deterministic via seed)."""

    @abc.abstractmethod
    def generate_ground_truth(self) -> pd.DataFrame:
        """Return the *ideal / expected* DataFrame after all fixes are applied."""

    @abc.abstractmethod
    def compute_quality(self, df: pd.DataFrame) -> float:
        """Return a quality score in [0.0, 1.0] for the current DataFrame state."""

    @abc.abstractmethod
    def step_reward(
        self,
        action_type: str,
        action_column: str | None,
        action_params: Dict[str, Any] | None,
        old_quality: float,
        new_quality: float,
        df_before: pd.DataFrame,
        df_after: pd.DataFrame,
    ) -> Dict[str, float]:
        """Return a breakdown dict of reward components for a single step."""

    def step_penalty(self) -> float:
        """Per-step time penalty (override for harder tasks)."""
        return -0.01
