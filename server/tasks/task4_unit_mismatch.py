"""Task 4 – Unit Mismatch (Hard, max_steps=12).

The agent must detect that EU and US rows use different measurement systems and
apply the correct conversions so every row ends up in the canonical unit system
(Celsius, km, USD, kg).
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from server.tasks.base import BaseTask

# Conversion helpers
_CONVERSIONS = {
    "F_to_C": lambda v: (v - 32) / 1.8,
    "C_to_F": lambda v: v * 1.8 + 32,
    "miles_to_km": lambda v: v * 1.60934,
    "km_to_miles": lambda v: v / 1.60934,
    "EUR_to_USD": lambda v: v * 1.08,
    "USD_to_EUR": lambda v: v / 1.08,
    "lbs_to_kg": lambda v: v * 0.453592,
    "kg_to_lbs": lambda v: v / 0.453592,
}


class UnitMismatchTask(BaseTask):
    name = "unit_mismatch"
    description = "Convert units across EU/US regional data merge"
    difficulty = "hard"
    max_steps = 12

    SEED = 789
    N_ROWS = 600
    N_PER_REGION = 300
    N_OUTLIERS = 20  # 10 per region

    # ---- data generation -----------------------------------------------

    def generate_data(self) -> pd.DataFrame:
        rng = np.random.RandomState(self.SEED)
        n = self.N_PER_REGION

        # EU rows: Celsius, km, EUR, kg
        eu_sensor = [f"S-EU-{str(i).zfill(4)}" for i in range(1, n + 1)]
        eu_temp = rng.uniform(-10.0, 40.0, n).round(2)
        eu_dist = rng.uniform(1.0, 500.0, n).round(2)
        eu_price = rng.uniform(5.0, 5000.0, n).round(2)
        eu_weight = rng.uniform(0.5, 200.0, n).round(2)
        eu_time = 1_704_067_200 + rng.randint(0, 10_000_000, n)

        # US rows: Fahrenheit, miles, USD, lbs
        us_sensor = [f"S-US-{str(i).zfill(4)}" for i in range(1, n + 1)]
        us_temp_c = rng.uniform(-10.0, 40.0, n).round(2)
        us_temp = (us_temp_c * 1.8 + 32).round(2)  # store as F
        us_dist_km = rng.uniform(1.0, 500.0, n).round(2)
        us_dist = (us_dist_km / 1.60934).round(2)  # store as miles
        us_price = rng.uniform(5.0, 5000.0, n).round(2)  # already USD
        us_weight_kg = rng.uniform(0.5, 200.0, n).round(2)
        us_weight = (us_weight_kg / 0.453592).round(2)  # store as lbs
        us_time = 1_704_067_200 + rng.randint(0, 10_000_000, n)

        # outliers (extreme but valid)
        for idx in rng.choice(n, 10, replace=False):
            eu_temp[idx] = rng.choice([-25.0, 48.0])
        for idx in rng.choice(n, 10, replace=False):
            us_temp[idx] = rng.choice([-15.0, 125.0])  # extreme F

        eu_df = pd.DataFrame({
            "sensor_id": eu_sensor,
            "source_region": "EU",
            "temperature": eu_temp,
            "distance": eu_dist,
            "price": eu_price,
            "weight": eu_weight,
            "reading_time": eu_time,
        })
        us_df = pd.DataFrame({
            "sensor_id": us_sensor,
            "source_region": "US",
            "temperature": us_temp,
            "distance": us_dist,
            "price": us_price,
            "weight": us_weight,
            "reading_time": us_time,
        })

        df = pd.concat([eu_df, us_df], ignore_index=True)
        df = df.sample(frac=1.0, random_state=self.SEED).reset_index(drop=True)
        return df

    def generate_ground_truth(self) -> pd.DataFrame:
        df = self.generate_data()
        us_mask = df["source_region"] == "US"

        df.loc[us_mask, "temperature"] = _CONVERSIONS["F_to_C"](
            df.loc[us_mask, "temperature"]
        ).round(2)
        df.loc[us_mask, "distance"] = _CONVERSIONS["miles_to_km"](
            df.loc[us_mask, "distance"]
        ).round(2)
        # price: EU→USD
        eu_mask = df["source_region"] == "EU"
        df.loc[eu_mask, "price"] = _CONVERSIONS["EUR_to_USD"](
            df.loc[eu_mask, "price"]
        ).round(2)
        df.loc[us_mask, "weight"] = _CONVERSIONS["lbs_to_kg"](
            df.loc[us_mask, "weight"]
        ).round(2)
        return df

    # ---- quality -------------------------------------------------------

    def compute_quality(self, df: pd.DataFrame) -> float:
        gt = self.generate_ground_truth()
        unit_cols = ["temperature", "distance", "price", "weight"]
        col_scores = []
        for col in unit_cols:
            if col not in df.columns or col not in gt.columns:
                col_scores.append(0.0)
                continue
            # align on index
            try:
                merged = pd.DataFrame({"cur": df[col].values, "exp": gt[col].values})
                close = (
                    (merged["cur"] - merged["exp"]).abs()
                    <= 0.001 * merged["exp"].abs() + 1e-9
                )
                col_scores.append(close.mean())
            except Exception:
                col_scores.append(0.0)

        # untouched score
        untouched = 1.0
        if "reading_time" in df.columns and "reading_time" in gt.columns:
            if not (df["reading_time"].values == gt["reading_time"].values).all():
                untouched = 0.0
        else:
            untouched = 0.0

        return round(
            0.8 * (sum(col_scores) / max(len(col_scores), 1)) + 0.2 * untouched, 4
        )

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

        if action_type == "convert_units" and action_column:
            if action_column == "reading_time":
                breakdown["reading_time_penalty"] = -0.30
            elif delta > 0.001:
                breakdown["correct_conversion"] = 0.12
            elif delta < -0.001:
                breakdown["wrong_region_penalty"] = -0.20

        if action_type == "inspect":
            breakdown["inspect_penalty"] = -0.005

        breakdown["step_penalty"] = self.step_penalty()
        return breakdown
