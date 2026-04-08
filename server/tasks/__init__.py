"""Task registry for PipelineRx."""

from server.tasks.task1_null_sweep import NullSweepTask
from server.tasks.task2_type_drift import TypeDriftTask
from server.tasks.task3_duplicate_drift import DuplicateDriftTask
from server.tasks.task4_unit_mismatch import UnitMismatchTask
from server.tasks.task5_pipeline_cascade import PipelineCascadeTask

TASK_REGISTRY = {
    "null_sweep": NullSweepTask,
    "type_drift": TypeDriftTask,
    "duplicate_drift": DuplicateDriftTask,
    "unit_mismatch": UnitMismatchTask,
    "pipeline_cascade": PipelineCascadeTask,
}

__all__ = [
    "TASK_REGISTRY",
    "NullSweepTask",
    "TypeDriftTask",
    "DuplicateDriftTask",
    "UnitMismatchTask",
    "PipelineCascadeTask",
]
