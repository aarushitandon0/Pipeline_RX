"""Task 1 – Null Sweep (Easy, max_steps=8).

The agent must fill nulls in age/income/score/product and drop the region column
(which has >30 % nulls and cannot be reliably imputed).
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from server.tasks.base import BaseTask


class NullSweepTask(BaseTask):
    name = "null_sweep"
    description = "Fix null values in a sensor dataset"
    difficulty = "easy"
    max_steps = 8

    SEED = 42
    N_ROWS = 1000
    NULL_SPEC: Dict[str, float] = {
        "age": 0.25,
        "income": 0.35,
        "score": 0.15,
        "region": 0.40,
        "product": 0.08,
    }
    DROP_THRESHOLD = 0.30  # columns above this should be dropped

    # ---- data generation -----------------------------------------------

    def generate_data(self) -> pd.DataFrame:
        rng = np.random.RandomState(self.SEED)
        n = self.N_ROWS

        df = pd.DataFrame(
            {
                "user_id": np.arange(1, n + 1),
                "age": rng.uniform(18, 80, n).round(1),
                "income": rng.uniform(20_000, 200_000, n).round(2),
                "score": rng.uniform(0, 100, n).round(2),
                "region": rng.choice(
                    ["North", "South", "East", "West"], n
                ),
                "product": rng.choice(
                    ["Widget", "Gadget", "Doohickey", "Thingamajig"], n
                ),
                "signup_date": pd.date_range(
                    "2020-01-01", periods=n, freq="h"
                )
                .strftime("%Y-%m-%d")
                .tolist(),
                "is_active": rng.choice([0, 1], n),
            }
        )

        # inject nulls
        for col, pct in self.NULL_SPEC.items():
            mask = rng.rand(n) < pct
            df.loc[mask, col] = np.nan

        return df

    def generate_ground_truth(self) -> pd.DataFrame:
        rng = np.random.RandomState(self.SEED)
        n = self.N_ROWS

        df = pd.DataFrame(
            {
                "user_id": np.arange(1, n + 1),
                "age": rng.uniform(18, 80, n).round(1),
                "income": rng.uniform(20_000, 200_000, n).round(2),
                "score": rng.uniform(0, 100, n).round(2),
                "region": rng.choice(
                    ["North", "South", "East", "West"], n
                ),
                "product": rng.choice(
                    ["Widget", "Gadget", "Doohickey", "Thingamajig"], n
                ),
                "signup_date": pd.date_range(
                    "2020-01-01", periods=n, freq="h"
                )
                .strftime("%Y-%m-%d")
                .tolist(),
                "is_active": rng.choice([0, 1], n),
            }
        )
        # ground truth: region dropped, no nulls elsewhere
        df = df.drop(columns=["region"])
        return df

    # ---- quality / grading --------------------------------------------

    def compute_quality(self, df: pd.DataFrame) -> float:
        # null_score: fraction of fixable columns that are fully fixed
        fixable = ["age", "income", "score", "product"]
        fixed = 0
        total = 0
        for col in fixable:
            if col in df.columns:
                total += 1
                if df[col].isnull().sum() == 0:
                    fixed += 1
            else:
                # column was dropped instead of fixed – count as NOT fixed
                total += 1

        null_score = fixed / max(total, 1)

        # drop_score: 1.0 iff region is dropped AND no other original column is
        # also dropped (except region)
        original_keep = {"user_id", "age", "income", "score", "product",
                         "signup_date", "is_active"}
        region_dropped = "region" not in df.columns
        others_kept = original_keep.issubset(set(df.columns))
        drop_score = 1.0 if (region_dropped and others_kept) else 0.0

        return round(0.6 * null_score + 0.4 * drop_score, 4)

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

        # quality delta amplified
        delta = new_quality - old_quality
        breakdown["quality_delta"] = round(delta * 2.0, 4)

        if action_type == "fill_nulls" and action_column:
            # check if this actually removed nulls
            if action_column in df_before.columns and action_column in df_after.columns:
                before_nulls = int(df_before[action_column].isnull().sum())
                after_nulls = int(df_after[action_column].isnull().sum())
                if after_nulls < before_nulls:
                    breakdown["null_fix_bonus"] = 0.15
                else:
                    breakdown["null_fix_bonus"] = 0.0
            else:
                breakdown["null_fix_bonus"] = 0.0

        if action_type == "drop_column" and action_column:
            if action_column == "region":
                breakdown["correct_drop"] = 0.10
            else:
                breakdown["wrong_drop_penalty"] = -0.20

        if action_type == "inspect":
            breakdown["inspect_penalty"] = -0.005

        if action_type == "finish":
            if new_quality >= 0.9:
                breakdown["early_finish_bonus"] = 0.05

        breakdown["step_penalty"] = self.step_penalty()
        return breakdown
