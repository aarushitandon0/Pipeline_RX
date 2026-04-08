"""Core reinforcement-learning environment for PipelineRx."""

from __future__ import annotations

import threading
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4

import numpy as np
import pandas as pd

from server.graders import grade
from server.models import (
    ColumnStats,
    PipelineAction,
    PipelineObservation,
    PipelineState,
)
from server.tasks import TASK_REGISTRY
from server.tasks.base import BaseTask

# Conversion look-up used by convert_units action
_CONV_FNS = {
    "F_to_C": lambda v: (v - 32) / 1.8,
    "C_to_F": lambda v: v * 1.8 + 32,
    "miles_to_km": lambda v: v * 1.60934,
    "km_to_miles": lambda v: v / 1.60934,
    "EUR_to_USD": lambda v: v * 1.08,
    "USD_to_EUR": lambda v: v / 1.08,
    "lbs_to_kg": lambda v: v * 0.453592,
    "kg_to_lbs": lambda v: v / 0.453592,
}

_BOOL_MAP = {
    "yes": 1, "no": 0, "true": 1, "false": 0,
    "1": 1, "0": 0, "Yes": 1, "No": 0,
    "YES": 1, "NO": 0, "TRUE": 1, "FALSE": 0,
}


class PipelineEnv:
    """Thread-safe RL environment managing a single episode at a time."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.current_task: Optional[BaseTask] = None
        self.df: Optional[pd.DataFrame] = None
        self.original_df: Optional[pd.DataFrame] = None
        self.step_count: int = 0
        self.episode_id: str = str(uuid4())
        self.cumulative_reward: float = 0.0
        self.action_history: List[Dict[str, Any]] = []
        self.done: bool = False
        self._last_action_result: str = "none"
        self._last_action_error: Optional[str] = None

    # ------------------------------------------------------------------
    # reset
    # ------------------------------------------------------------------

    def reset(self, task_name: str = "null_sweep") -> Dict[str, Any]:
        with self._lock:
            if task_name not in TASK_REGISTRY:
                raise ValueError(
                    f"Unknown task '{task_name}'. "
                    f"Available: {list(TASK_REGISTRY.keys())}"
                )
            task_cls = TASK_REGISTRY[task_name]
            self.current_task = task_cls()
            self.df = self.current_task.generate_data()
            self.original_df = self.df.copy()
            self.step_count = 0
            self.episode_id = str(uuid4())
            self.cumulative_reward = 0.0
            self.action_history = []
            self.done = False
            self._last_action_result = "none"
            self._last_action_error = None

            obs = self._compute_observation()
            return {
                "observation": obs.model_dump(),
                "done": False,
                "reward": None,
                "info": {},
            }

    # ------------------------------------------------------------------
    # step
    # ------------------------------------------------------------------

    def step(self, action: PipelineAction) -> Dict[str, Any]:
        with self._lock:
            if self.current_task is None or self.df is None:
                raise RuntimeError("Call /reset before /step.")
            if self.done:
                obs = self._compute_observation()
                return {
                    "observation": obs.model_dump(),
                    "reward": 0.0,
                    "done": True,
                    "info": {"message": "Episode already done."},
                }

            old_quality = self.current_task.compute_quality(self.df)
            df_before = self.df.copy()

            # Apply action
            result, error = self._apply_action(action)
            self._last_action_result = result
            self._last_action_error = error

            self.step_count += 1
            self.action_history.append(
                {
                    "step": self.step_count,
                    "action_type": action.action_type,
                    "column": action.column,
                    "params": action.params,
                    "result": result,
                    "error": error,
                }
            )

            new_quality = self.current_task.compute_quality(self.df)

            # Reward
            reward_breakdown = self.current_task.step_reward(
                action_type=action.action_type,
                action_column=action.column,
                action_params=action.params,
                old_quality=old_quality,
                new_quality=new_quality,
                df_before=df_before,
                df_after=self.df,
            )
            step_reward = sum(reward_breakdown.values())
            step_reward = max(-1.0, min(1.0, step_reward))
            self.cumulative_reward += step_reward

            # Done check
            if action.action_type == "finish":
                self.done = True
            if self.step_count >= self.current_task.max_steps:
                self.done = True

            # Final grading on done
            info: Dict[str, Any] = {"reward_breakdown": reward_breakdown}
            if self.done:
                final_score = grade(
                    self.current_task, self.df, self.action_history
                )
                info["final_score"] = final_score

            obs = self._compute_observation()
            return {
                "observation": obs.model_dump(),
                "reward": round(step_reward, 4),
                "done": self.done,
                "info": info,
            }

    # ------------------------------------------------------------------
    # state
    # ------------------------------------------------------------------

    def get_state(self) -> PipelineState:
        with self._lock:
            return PipelineState(
                task_name=self.current_task.name if self.current_task else "",
                step=self.step_count,
                done=self.done,
                episode_id=self.episode_id,
                current_quality=(
                    self.current_task.compute_quality(self.df)
                    if self.current_task and self.df is not None
                    else 0.0
                ),
                action_history=self.action_history,
            )

    # ------------------------------------------------------------------
    # observation builder
    # ------------------------------------------------------------------

    def _compute_observation(self) -> PipelineObservation:
        assert self.current_task is not None
        assert self.df is not None

        df = self.df
        col_stats: Dict[str, ColumnStats] = {}
        for col in df.columns:
            series = df[col]
            null_count = int(series.isnull().sum())
            null_pct = round(null_count / max(len(df), 1), 4)
            dtype_str = str(series.dtype)

            mean_val = std_val = min_val = max_val = None
            if pd.api.types.is_numeric_dtype(series):
                desc = series.describe()
                mean_val = round(float(desc.get("mean", 0)), 4) if "mean" in desc.index else None
                std_val = round(float(desc.get("std", 0)), 4) if "std" in desc.index else None
                min_val = round(float(desc.get("min", 0)), 4) if "min" in desc.index else None
                max_val = round(float(desc.get("max", 0)), 4) if "max" in desc.index else None

            unique_count = int(series.nunique(dropna=True))
            sample_vals = (
                series.dropna().head(5).tolist() if not series.dropna().empty else []
            )

            col_stats[col] = ColumnStats(
                null_count=null_count,
                null_pct=null_pct,
                dtype=dtype_str,
                mean=mean_val,
                std=std_val,
                min=min_val,
                max=max_val,
                unique_count=unique_count,
                sample_values=sample_vals,
            )

        # sample rows (5 rows as list of dicts, handle NaN → None)
        sample_df = df.head(5).copy()
        sample_rows = []
        for _, row in sample_df.iterrows():
            d = {}
            for k, v in row.items():
                if pd.isna(v):
                    d[k] = None
                elif isinstance(v, (np.integer,)):
                    d[k] = int(v)
                elif isinstance(v, (np.floating,)):
                    d[k] = float(v)
                elif isinstance(v, np.bool_):
                    d[k] = bool(v)
                else:
                    d[k] = v
                    
            sample_rows.append(d)

        quality = self.current_task.compute_quality(df)

        return PipelineObservation(
            step=self.step_count,
            max_steps=self.current_task.max_steps,
            shape=(len(df), len(df.columns)),
            columns=list(df.columns),
            column_stats=col_stats,
            sample_rows=sample_rows,
            last_action_result=self._last_action_result,
            last_action_error=self._last_action_error,
            quality_score=round(quality, 4),
            task_name=self.current_task.name,
            task_description=self.current_task.description,
        )

    # ------------------------------------------------------------------
    # action dispatcher
    # ------------------------------------------------------------------

    def _apply_action(
        self, action: PipelineAction
    ) -> Tuple[str, Optional[str]]:
        assert self.df is not None
        at = action.action_type
        col = action.column
        params = action.params or {}

        try:
            if at == "fill_nulls":
                return self._action_fill_nulls(col, params)
            elif at == "cast_column":
                return self._action_cast_column(col, params)
            elif at == "drop_column":
                return self._action_drop_column(col)
            elif at == "deduplicate":
                return self._action_deduplicate(params)
            elif at == "convert_units":
                return self._action_convert_units(col, params)
            elif at == "inspect":
                return ("inspect_ok", None)
            elif at == "finish":
                return ("finish", None)
            else:
                return ("invalid_action", f"Unknown action_type '{at}'")
        except Exception as exc:  # noqa: BLE001
            return ("error", str(exc))

    # ---------- individual action implementations -----------------------

    def _action_fill_nulls(
        self, col: Optional[str], params: Dict[str, Any]
    ) -> Tuple[str, Optional[str]]:
        if not col or col not in self.df.columns:
            return ("invalid_column", f"Column '{col}' not found.")
        strategy = params.get("strategy", "median")
        series = self.df[col]

        if strategy == "median":
            if pd.api.types.is_numeric_dtype(series):
                self.df[col] = series.fillna(series.median())
            else:
                return ("invalid_strategy", "median requires numeric column")
        elif strategy == "mode":
            mode_vals = series.mode()
            if len(mode_vals) > 0:
                self.df[col] = series.fillna(mode_vals.iloc[0])
            else:
                return ("no_op", "mode is empty")
        elif strategy == "forward_fill":
            self.df[col] = series.ffill()
        elif strategy == "zero":
            self.df[col] = series.fillna(0)
        else:
            return ("invalid_strategy", f"Unknown strategy '{strategy}'")

        return ("success", None)

    def _action_cast_column(
        self, col: Optional[str], params: Dict[str, Any]
    ) -> Tuple[str, Optional[str]]:
        if not col or col not in self.df.columns:
            return ("invalid_column", f"Column '{col}' not found.")
        target_dtype = params.get("dtype", "float64")

        if target_dtype == "float64":
            self.df[col] = pd.to_numeric(self.df[col], errors="coerce")
        elif target_dtype == "int64":
            self.df[col] = pd.to_numeric(self.df[col], errors="coerce")
            # drop rows with NaN before int cast would lose data; use nullable int
            self.df[col] = self.df[col].astype("Int64")
        elif target_dtype == "bool":
            self.df[col] = (
                self.df[col]
                .astype(str)
                .str.strip()
                .str.lower()
                .map(_BOOL_MAP)
                .astype("Int64")
            )
        elif target_dtype == "datetime":
            self.df[col] = pd.to_datetime(self.df[col], errors="coerce")
            self.df[col] = (
                self.df[col]
                .astype("int64", errors="ignore") // 10**9
            )
            # fallback: try converting to int64
            try:
                self.df[col] = self.df[col].astype("Int64")
            except Exception:
                pass
        else:
            return ("invalid_dtype", f"Unsupported dtype '{target_dtype}'")

        return ("success", None)

    def _action_drop_column(
        self, col: Optional[str]
    ) -> Tuple[str, Optional[str]]:
        if not col or col not in self.df.columns:
            return ("invalid_column", f"Column '{col}' not found.")
        self.df = self.df.drop(columns=[col])
        return ("success", None)

    def _action_deduplicate(
        self, params: Dict[str, Any]
    ) -> Tuple[str, Optional[str]]:
        key_columns: List[str] = params.get("key_columns", [])
        keep: str = params.get("keep", "last")

        if not key_columns:
            return ("invalid_params", "key_columns must be a non-empty list.")

        missing = [c for c in key_columns if c not in self.df.columns]
        if missing:
            return ("invalid_column", f"Columns not found: {missing}")

        if keep not in ("first", "last"):
            return ("invalid_params", f"keep must be 'first' or 'last', got '{keep}'")

        self.df = self.df.sort_values(
            key_columns, kind="mergesort"
        )
        before_len = len(self.df)
        self.df = self.df.drop_duplicates(subset=key_columns, keep=keep)
        self.df = self.df.reset_index(drop=True)
        removed = before_len - len(self.df)
        return ("success", f"Removed {removed} duplicate rows.")

    def _action_convert_units(
        self, col: Optional[str], params: Dict[str, Any]
    ) -> Tuple[str, Optional[str]]:
        if not col or col not in self.df.columns:
            return ("invalid_column", f"Column '{col}' not found.")

        conversion = params.get("conversion")
        source_region = params.get("source_region")

        if conversion not in _CONV_FNS:
            return (
                "invalid_params",
                f"Unknown conversion '{conversion}'. "
                f"Available: {list(_CONV_FNS.keys())}",
            )

        fn = _CONV_FNS[conversion]

        if source_region and "source_region" in self.df.columns:
            mask = self.df["source_region"] == source_region
            if not pd.api.types.is_numeric_dtype(self.df[col]):
                return ("invalid_column", f"Column '{col}' is not numeric.")
            self.df.loc[mask, col] = fn(self.df.loc[mask, col]).round(4)
        else:
            if not pd.api.types.is_numeric_dtype(self.df[col]):
                return ("invalid_column", f"Column '{col}' is not numeric.")
            self.df[col] = fn(self.df[col]).round(4)

        return ("success", None)
