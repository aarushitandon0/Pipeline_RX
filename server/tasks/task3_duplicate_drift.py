"""Task 3 – Duplicate Drift (Medium, max_steps=10).

The agent must deduplicate an event log that has 200 true duplicate pairs
(same event_id + event_type with later timestamp) while preserving 50
legitimate multi-event entries (same event_id but different event_type).
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from server.tasks.base import BaseTask


class DuplicateDriftTask(BaseTask):
    name = "duplicate_drift"
    description = "Deduplicate a CDC-replayed event log"
    difficulty = "medium"
    max_steps = 10

    SEED = 456
    N_BASE = 950  # base unique rows
    N_TRUE_DUPES = 200
    N_MULTI_EVENT = 50
    EXPECTED_FINAL_ROWS = 1000  # 1200 - 200

    # ---- data generation -----------------------------------------------

    def generate_data(self) -> pd.DataFrame:
        rng = np.random.RandomState(self.SEED)

        # base rows: 950 unique events
        event_ids = [f"EVT-{str(i).zfill(5)}" for i in range(1, self.N_BASE + 1)]
        user_ids = rng.randint(1, 500, self.N_BASE)
        event_types = rng.choice(["click", "purchase", "view", "signup"], self.N_BASE).tolist()
        values = rng.uniform(1.0, 500.0, self.N_BASE).round(2)
        base_ts = 1_704_067_200
        timestamps = base_ts + rng.randint(0, 10_000_000, self.N_BASE)
        processed = rng.choice([0, 1], self.N_BASE)

        rows = list(
            zip(event_ids, user_ids, event_types, values, timestamps, processed)
        )

        # 200 true duplicates: copy first 200, same event_id + event_type,
        # later timestamp, updated value
        dupe_rows = []
        for i in range(self.N_TRUE_DUPES):
            eid, uid, etype, val, ts, proc = rows[i]
            new_ts = ts + rng.randint(1000, 100_000)
            new_val = round(val + rng.uniform(-10, 10), 2)
            dupe_rows.append((eid, uid, etype, new_val, new_ts, proc))

        # 50 legitimate multi-events: same event_id, DIFFERENT event_type
        multi_rows = []
        for i in range(self.N_MULTI_EVENT):
            idx = self.N_TRUE_DUPES + i  # pick rows 200..249
            eid, uid, etype, val, ts, proc = rows[idx]
            # flip event type to something different
            alt_type = "purchase" if etype != "purchase" else "click"
            new_ts = ts + rng.randint(1000, 50_000)
            new_val = round(rng.uniform(10.0, 200.0), 2)
            multi_rows.append((eid, uid, alt_type, new_val, new_ts, proc))

        all_rows = rows + dupe_rows + multi_rows  # 950 + 200 + 50 = 1200

        df = pd.DataFrame(
            all_rows,
            columns=[
                "event_id", "user_id", "event_type", "value", "timestamp", "processed"
            ],
        )
        # shuffle deterministically
        df = df.sample(frac=1.0, random_state=self.SEED).reset_index(drop=True)
        return df

    def generate_ground_truth(self) -> pd.DataFrame:
        df = self.generate_data()
        df = df.sort_values("timestamp")
        df = df.drop_duplicates(subset=["event_id", "event_type"], keep="last")
        df = df.reset_index(drop=True)
        return df

    # ---- quality -------------------------------------------------------

    def compute_quality(self, df: pd.DataFrame) -> float:
        original_count = 1200
        rows_removed = original_count - len(df)

        # dedup_score
        if 195 <= rows_removed <= 205:
            dedup_score = 1.0
        elif 150 <= rows_removed <= 250:
            dedup_score = 0.5
        else:
            dedup_score = 0.0

        # legit_events_score: check that multi-event ids still have both types
        gt = self.generate_data()
        # multi-event ids are the first 50 after the dupe block in the original generation
        rng = np.random.RandomState(self.SEED)
        base_ids = [f"EVT-{str(i).zfill(5)}" for i in range(1, self.N_BASE + 1)]
        multi_event_ids = base_ids[self.N_TRUE_DUPES: self.N_TRUE_DUPES + self.N_MULTI_EVENT]

        preserved = 0
        for eid in multi_event_ids:
            sub = df[df["event_id"] == eid]
            if sub["event_type"].nunique() >= 2:
                preserved += 1
        legit_score = preserved / max(len(multi_event_ids), 1)

        return round(0.6 * dedup_score + 0.4 * legit_score, 4)

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

        if action_type == "deduplicate":
            rows_removed = len(df_before) - len(df_after)
            if 195 <= rows_removed <= 205:
                breakdown["correct_dedup"] = 0.50
            elif rows_removed > 205:
                breakdown["over_dedup_penalty"] = -0.30
            elif rows_removed > 0:
                proportion = rows_removed / 200.0
                breakdown["partial_dedup"] = round(0.20 * proportion, 4)

            # check key columns
            params = action_params or {}
            key_cols = params.get("key_columns", [])
            if sorted(key_cols) != ["event_id", "event_type"]:
                breakdown["wrong_keys_penalty"] = -0.20

        if action_type == "inspect":
            breakdown["inspect_penalty"] = -0.005

        breakdown["step_penalty"] = self.step_penalty()
        return breakdown
