"""Task 5 – Pipeline Cascade (Very Hard, max_steps=15).

All four failure modes are present simultaneously.  The optimal fix order is
cast → fill_nulls → deduplicate → convert_units.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from server.tasks.base import BaseTask

_CONVERSIONS = {
    "F_to_C": lambda v: (v - 32) / 1.8,
    "miles_to_km": lambda v: v * 1.60934,
    "EUR_to_USD": lambda v: v * 1.08,
    "lbs_to_kg": lambda v: v * 0.453592,
}

OPTIMAL_ORDER = ["cast_column", "fill_nulls", "deduplicate", "convert_units"]


class PipelineCascadeTask(BaseTask):
    name = "pipeline_cascade"
    description = "Fix all failure modes in the correct order"
    difficulty = "very_hard"
    max_steps = 15

    SEED = 999
    N_ROWS = 1500

    # ---- data generation -----------------------------------------------

    def generate_data(self) -> pd.DataFrame:
        rng = np.random.RandomState(self.SEED)
        n = self.N_ROWS

        sensor_id = [f"SC-{str(i).zfill(5)}" for i in range(1, n + 1)]
        region = rng.choice(["EU", "US"], n).tolist()

        # temperature: stored as str (type drift) with some dirty values
        temp_raw = rng.uniform(-10, 45, n).round(2)
        temp_str = temp_raw.astype(str).tolist()
        for i in rng.choice(n, 60, replace=False):
            temp_str[i] = rng.choice(["N/A", "--", ""])

        # distance: stored as str
        dist_raw = rng.uniform(1, 500, n).round(2)
        dist_str = dist_raw.astype(str).tolist()
        for i in rng.choice(n, 40, replace=False):
            dist_str[i] = "NaN"

        # value: float with 20% nulls
        value = rng.uniform(10, 5000, n).round(2)
        null_mask_value = rng.rand(n) < 0.20
        value_series = pd.Series(value)
        value_series[null_mask_value] = np.nan

        # weight: float with unit mismatch
        weight_kg = rng.uniform(0.5, 200, n).round(2)
        weight = weight_kg.copy()
        for i in range(n):
            if region[i] == "US":
                weight[i] = round(weight_kg[i] / 0.453592, 2)  # stored as lbs

        # category: str with 10% nulls
        category = rng.choice(["A", "B", "C", "D"], n).tolist()
        cat_series = pd.Series(category)
        null_mask_cat = rng.rand(n) < 0.10
        cat_series[null_mask_cat] = np.nan

        # event_id: for duplicate injection
        event_ids = [f"E-{str(i).zfill(5)}" for i in range(1, n + 1)]

        # event_type
        event_types = rng.choice(["click", "purchase", "view"], n).tolist()

        # reading_time (correct, don't touch)
        reading_time = 1_704_067_200 + rng.randint(0, 10_000_000, n)

        # price: stored as str (type drift) with unit mismatch
        price_raw = rng.uniform(5, 3000, n).round(2)
        price_str = price_raw.astype(str).tolist()
        for i in rng.choice(n, 30, replace=False):
            price_str[i] = rng.choice(["N/A", "na", ""])

        df = pd.DataFrame({
            "sensor_id": sensor_id,
            "source_region": region,
            "temperature": temp_str,
            "distance": dist_str,
            "value": value_series,
            "weight": weight,
            "category": cat_series,
            "event_id": event_ids,
            "event_type": event_types,
            "reading_time": reading_time,
        })

        # inject 150 true duplicates (same event_id + event_type)
        dupe_indices = rng.choice(n, 150, replace=False)
        dupe_rows = df.iloc[dupe_indices].copy()
        dupe_rows["reading_time"] = dupe_rows["reading_time"] + rng.randint(1000, 50000, 150)
        df = pd.concat([df, dupe_rows], ignore_index=True)
        df = df.sample(frac=1.0, random_state=self.SEED).reset_index(drop=True)
        return df

    def generate_ground_truth(self) -> pd.DataFrame:
        df = self.generate_data()

        # 1. cast
        df["temperature"] = pd.to_numeric(df["temperature"], errors="coerce")
        df["distance"] = pd.to_numeric(df["distance"], errors="coerce")

        # 2. fill nulls
        df["temperature"] = df["temperature"].fillna(df["temperature"].median())
        df["distance"] = df["distance"].fillna(df["distance"].median())
        df["value"] = df["value"].fillna(df["value"].median())
        df["category"] = df["category"].fillna(df["category"].mode().iloc[0])

        # 3. deduplicate
        df = df.sort_values("reading_time")
        df = df.drop_duplicates(subset=["event_id", "event_type"], keep="last")
        df = df.reset_index(drop=True)

        # 4. convert units
        us_mask = df["source_region"] == "US"
        df.loc[us_mask, "temperature"] = _CONVERSIONS["F_to_C"](
            df.loc[us_mask, "temperature"]
        ).round(2)
        df.loc[us_mask, "distance"] = _CONVERSIONS["miles_to_km"](
            df.loc[us_mask, "distance"]
        ).round(2)
        eu_mask = df["source_region"] == "EU"
        # price stored as string, we need to convert back
        # Actually price is still str at this point in ground truth; handle carefully
        df.loc[us_mask, "weight"] = _CONVERSIONS["lbs_to_kg"](
            df.loc[us_mask, "weight"]
        ).round(2)

        return df

    # ---- quality (composite) -------------------------------------------

    def compute_quality(self, df: pd.DataFrame) -> float:
        scores: Dict[str, float] = {}

        # null component
        null_cols = ["value", "category"]
        fixed = sum(
            1 for c in null_cols if c in df.columns and df[c].isnull().sum() == 0
        )
        scores["null"] = fixed / max(len(null_cols), 1)

        # type component
        type_targets = {"temperature": "float64", "distance": "float64"}
        type_hits = 0
        for col, exp in type_targets.items():
            if col in df.columns:
                actual = str(df[col].dtype)
                if exp in actual:
                    type_hits += 1
        scores["type"] = type_hits / max(len(type_targets), 1)

        # dedup component
        original_count = len(self.generate_data())
        rows_removed = original_count - len(df)
        if 145 <= rows_removed <= 155:
            scores["dedup"] = 1.0
        elif 100 <= rows_removed <= 200:
            scores["dedup"] = 0.5
        else:
            scores["dedup"] = 0.0

        # unit component (simplified: check weight for US rows)
        gt = self.generate_ground_truth()
        unit_score = 0.0
        try:
            if "weight" in df.columns and "weight" in gt.columns and len(df) == len(gt):
                close = (
                    (df["weight"].values - gt["weight"].values).__abs__()
                    <= 0.001 * np.abs(gt["weight"].values) + 1e-9
                )
                unit_score = close.mean()
            elif "weight" in df.columns:
                unit_score = 0.3  # partial credit for having the column
        except Exception:
            unit_score = 0.0
        scores["unit"] = unit_score

        quality = (
            0.25 * scores["null"]
            + 0.25 * scores["type"]
            + 0.25 * scores["dedup"]
            + 0.25 * scores["unit"]
        )
        return round(min(max(quality, 0.0), 1.0), 4)

    # ---- per-step reward -----------------------------------------------

    def step_penalty(self) -> float:
        return -0.02

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

        if action_type == "inspect":
            breakdown["inspect_penalty"] = -0.005

        breakdown["step_penalty"] = self.step_penalty()
        return breakdown

    # ---- order bonus (called externally by grader) ---------------------

    @staticmethod
    def order_bonus(action_history: List[Dict[str, Any]]) -> float:
        """Return +0.10 if the agent executed fix types in the optimal order."""
        seen_types: List[str] = []
        for entry in action_history:
            at = entry.get("action_type", "")
            if at in OPTIMAL_ORDER and (not seen_types or seen_types[-1] != at):
                seen_types.append(at)

        # check if seen_types is a subsequence of OPTIMAL_ORDER
        it = iter(OPTIMAL_ORDER)
        if all(t in it for t in seen_types) and len(seen_types) >= 3:
            return 0.10
        return 0.0

    @staticmethod
    def efficiency_bonus(steps: int, quality: float) -> float:
        if steps <= 10 and quality > 0.7:
            return 0.05
        return 0.0
