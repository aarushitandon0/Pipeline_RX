"""Task 2 – Type Drift (Medium, max_steps=10).

The agent must cast five columns that were stored as object/str to their correct
dtypes while leaving customer_id (str) and revenue (float64) untouched.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from server.tasks.base import BaseTask


class TypeDriftTask(BaseTask):
    name = "type_drift"
    description = "Correct type errors in a mixed-format dataset"
    difficulty = "medium"
    max_steps = 10

    SEED = 123
    N_ROWS = 800

    TARGET_DTYPES = {
        "temperature": "float64",
        "distance_km": "float64",
        "event_count": "int64",
        "timestamp": "int64",
        "is_valid": "int64",
    }
    CORRECT_COLUMNS = {"customer_id", "revenue"}

    # ---- data generation -----------------------------------------------

    def generate_data(self) -> pd.DataFrame:
        rng = np.random.RandomState(self.SEED)
        n = self.N_ROWS

        # temperature as str with dirty values
        temp_vals = rng.uniform(10.0, 45.0, n).round(1).astype(str).tolist()
        for i in rng.choice(n, 40, replace=False):
            temp_vals[i] = rng.choice(["N/A", "--", ""])
        temperature = temp_vals

        # distance_km as str
        dist_vals = rng.uniform(1.0, 500.0, n).round(1).astype(str).tolist()
        for i in rng.choice(n, 30, replace=False):
            dist_vals[i] = "NaN"
        distance_km = dist_vals

        # event_count as str
        ec_vals = rng.randint(0, 100, n).astype(str).tolist()
        for i in rng.choice(n, 25, replace=False):
            ec_vals[i] = "na"
        event_count = ec_vals

        # timestamp mixed: some ISO strings, some unix ints stored as str
        base_ts = 1_704_067_200  # 2024-01-01 UTC
        ts_vals = []
        for i in range(n):
            if rng.rand() < 0.4:
                # ISO date string
                day_offset = int(rng.randint(0, 365))
                dt = pd.Timestamp("2024-01-01") + pd.Timedelta(days=day_offset)
                ts_vals.append(dt.strftime("%Y-%m-%d"))
            else:
                ts_vals.append(str(base_ts + int(rng.randint(0, 30_000_000))))
        timestamp = ts_vals

        # is_valid: "yes"/"no"/"true"/"false"/"1"/"0"/"Yes"/"No"
        bool_pool = ["yes", "no", "true", "false", "1", "0", "Yes", "No"]
        is_valid = rng.choice(bool_pool, n).tolist()

        # correct columns
        customer_id = [f"C{str(i).zfill(4)}" for i in range(1, n + 1)]
        revenue = rng.uniform(10.0, 10_000.0, n).round(2)

        df = pd.DataFrame(
            {
                "temperature": temperature,
                "distance_km": distance_km,
                "event_count": event_count,
                "timestamp": timestamp,
                "is_valid": is_valid,
                "customer_id": customer_id,
                "revenue": revenue,
            }
        )
        return df

    def generate_ground_truth(self) -> pd.DataFrame:
        df = self.generate_data()
        # cast temperature
        df["temperature"] = pd.to_numeric(df["temperature"], errors="coerce")
        # cast distance_km
        df["distance_km"] = pd.to_numeric(df["distance_km"], errors="coerce")
        # cast event_count
        df["event_count"] = pd.to_numeric(df["event_count"], errors="coerce")
        df["event_count"] = df["event_count"].astype("Int64")
        # cast timestamp
        def _parse_ts(val: Any) -> Optional[int]:
            s = str(val).strip()
            try:
                return int(float(s))
            except (ValueError, TypeError):
                pass
            try:
                return int(pd.Timestamp(s).timestamp())
            except Exception:
                return None

        df["timestamp"] = df["timestamp"].apply(_parse_ts).astype("Int64")
        # cast is_valid
        bool_map = {
            "yes": 1, "no": 0, "true": 1, "false": 0,
            "1": 1, "0": 0, "Yes": 1, "No": 0,
        }
        df["is_valid"] = df["is_valid"].map(bool_map).astype("Int64")
        return df

    # ---- quality -------------------------------------------------------

    def compute_quality(self, df: pd.DataFrame) -> float:
        # dtype_score
        dtype_hits = 0
        for col, expected in self.TARGET_DTYPES.items():
            if col in df.columns:
                actual = str(df[col].dtype)
                if expected in actual or actual == expected:
                    dtype_hits += 1
        dtype_score = dtype_hits / len(self.TARGET_DTYPES)

        # nan_score after cast
        nan_scores = []
        for col in self.TARGET_DTYPES:
            if col in df.columns and df[col].dtype != object:
                nan_rate = df[col].isnull().mean()
                nan_scores.append(max(0.0, 1.0 - nan_rate / 0.2))
            else:
                nan_scores.append(0.0)
        nan_score = sum(nan_scores) / max(len(nan_scores), 1)

        # untouched_score
        untouched = 1.0
        for col in self.CORRECT_COLUMNS:
            if col not in df.columns:
                untouched = 0.0
                break
        if "customer_id" in df.columns and df["customer_id"].dtype != object:
            untouched = 0.0
        if "revenue" in df.columns and not pd.api.types.is_float_dtype(df["revenue"]):
            untouched = 0.0

        return round(0.5 * dtype_score + 0.3 * nan_score + 0.2 * untouched, 4)

    # ---- per-step reward -----------------------------------------------

    def step_reward(
        self,
        action_type: str,
        action_column: Optional[str],
        action_params: Optional[Dict[str, Any]],
        old_quality: float,
        new_quality: float,
        df_before: pd.DataFrame,
        df_after: pd.DataFrame,
    ) -> Dict[str, float]:
        breakdown: Dict[str, float] = {}
        delta = new_quality - old_quality
        breakdown["quality_delta"] = round(delta * 2.0, 4)

        if action_type == "cast_column" and action_column:
            if action_column in self.TARGET_DTYPES:
                expected = self.TARGET_DTYPES[action_column]
                actual = str(df_after[action_column].dtype) if action_column in df_after.columns else ""
                if expected in actual or actual == expected:
                    breakdown["correct_cast"] = 0.15
                    # nan bonus
                    if action_column in df_after.columns:
                        nan_rate = df_after[action_column].isnull().mean()
                        if nan_rate < 0.10:
                            breakdown["low_nan_bonus"] = 0.05
                else:
                    breakdown["incorrect_cast"] = 0.0
            elif action_column in self.CORRECT_COLUMNS:
                breakdown["wrong_column_penalty"] = -0.25

        if action_type == "inspect":
            breakdown["inspect_penalty"] = -0.005

        breakdown["step_penalty"] = self.step_penalty()
        return breakdown
