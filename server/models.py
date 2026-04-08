"""Pydantic v2 models for PipelineRx environment."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field


class ColumnStats(BaseModel):
    null_count: int
    null_pct: float
    dtype: str
    mean: Optional[float] = None
    std: Optional[float] = None
    min: Optional[float] = None
    max: Optional[float] = None
    unique_count: int
    sample_values: List[Any]


class PipelineObservation(BaseModel):
    step: int
    max_steps: int
    shape: Tuple[int, int]
    columns: List[str]
    column_stats: Dict[str, ColumnStats]
    sample_rows: List[Dict[str, Any]]
    last_action_result: str
    last_action_error: Optional[str] = None
    quality_score: float
    task_name: str
    task_description: str


class PipelineAction(BaseModel):
    action_type: str
    column: Optional[str] = None
    params: Optional[Dict[str, Any]] = None


class PipelineReward(BaseModel):
    value: float
    breakdown: Dict[str, float]
    cumulative: float


class PipelineState(BaseModel):
    task_name: str
    step: int
    done: bool
    episode_id: str
    current_quality: float
    action_history: List[Dict[str, Any]]


class ResetRequest(BaseModel):
    task_name: str = "null_sweep"


class StepResponse(BaseModel):
    observation: PipelineObservation
    reward: Optional[float] = None
    done: bool
    info: Dict[str, Any] = Field(default_factory=dict)


class ResetResponse(BaseModel):
    observation: PipelineObservation
    done: bool = False
    reward: Optional[float] = None
    info: Dict[str, Any] = Field(default_factory=dict)


class TaskInfo(BaseModel):
    name: str
    description: str
    difficulty: str
    max_steps: int
