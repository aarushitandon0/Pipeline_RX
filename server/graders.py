"""Grading logic for each PipelineRx task.

The grader is invoked when the agent sends the ``finish`` action or the step
budget is exhausted.  It returns a float in [0.0, 1.0].
"""

from __future__ import annotations

from typing import Any, Dict, List

import pandas as pd

from server.tasks.base import BaseTask
from server.tasks.task5_pipeline_cascade import PipelineCascadeTask


def grade(
    task: BaseTask,
    df: pd.DataFrame,
    action_history: List[Dict[str, Any]],
) -> float:
    """Return final score in [0.0, 1.0]."""
    base_score = task.compute_quality(df)

    # Extra bonuses for pipeline_cascade
    if isinstance(task, PipelineCascadeTask):
        bonus = PipelineCascadeTask.order_bonus(action_history)
        bonus += PipelineCascadeTask.efficiency_bonus(len(action_history), base_score)
        return round(min(max(base_score + bonus, 0.0), 1.0), 4)

    return round(min(max(base_score, 0.0), 1.0), 4)
