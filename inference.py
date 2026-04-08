#!/usr/bin/env python3
"""PipelineRx inference script.

Runs all 5 tasks sequentially, using an LLM via the OpenAI-compatible API to
decide which fix actions to apply at each step.

STDOUT FORMAT (per task):
    [START] task=<task_name> env=pipelinerx model=<model_name>
    [STEP]  step=<n> action=<json> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> score=<0.00> rewards=<r1,r2,...,rn>

Required env vars:
    API_BASE_URL   – OpenAI-compatible endpoint (injected by validator)
    API_KEY        – API key for the endpoint (injected by validator)
    MODEL_NAME     – model id (default: Qwen/Qwen2.5-72B-Instruct)
"""

from __future__ import annotations

import json
import os
import re
import sys
import textwrap
import traceback
from typing import Any, Dict, List, Optional

import requests
from openai import OpenAI

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
MODEL_NAME: str = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
ENV_URL: str = os.getenv("ENV_URL", "http://localhost:8000")
BENCHMARK: str = "pipelinerx"

TASKS: List[str] = [
    "null_sweep",
    "type_drift",
    "duplicate_drift",
    "unit_mismatch",
    "pipeline_cascade",
]
MAX_STEPS: Dict[str, int] = {
    "null_sweep": 8,
    "type_drift": 10,
    "duplicate_drift": 10,
    "unit_mismatch": 12,
    "pipeline_cascade": 15,
}
TEMPERATURE: float = 0.2
MAX_TOKENS: int = 300

# Client is lazily initialised in call_llm() so env vars are read at runtime
_client: Optional[OpenAI] = None

# ---------------------------------------------------------------------------
# STDOUT protocol helpers  (exactly matching the mandatory format)
# ---------------------------------------------------------------------------


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(
    step: int,
    action: str,
    reward: float,
    done: bool,
    error: Optional[str],
) -> None:
    done_val = str(done).lower()
    error_val = error if error else "null"
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} "
        f"done={done_val} error={error_val}",
        flush=True,
    )


def log_end(
    success: bool, steps: int, score: float, rewards: List[float]
) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} "
        f"score={score:.2f} rewards={rewards_str}",
        flush=True,
    )


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = textwrap.dedent("""\
You are an expert data engineer agent operating inside the PipelineRx environment.
You are given an observation of a corrupted pandas DataFrame and must choose a single repair action.

Available action_types and their required fields:
1. fill_nulls   – column (str), params.strategy ("median"|"mode"|"forward_fill"|"zero")
2. cast_column  – column (str), params.dtype ("float64"|"int64"|"bool"|"datetime")
3. drop_column  – column (str)
4. deduplicate  – params.key_columns (list[str]), params.keep ("first"|"last")
5. convert_units – column (str), params.source_region (str), params.conversion
     Conversions: "F_to_C","C_to_F","miles_to_km","km_to_miles",
                  "EUR_to_USD","USD_to_EUR","lbs_to_kg","kg_to_lbs"
6. inspect      – no extra fields (just observe again; costs a step)
7. finish       – no extra fields (end the episode, trigger grading)

JSON schema for your reply:
{"action_type": "<type>", "column": "<col_or_null>", "params": {<key>: <value>}}

Rules:
- Fix the highest-impact issue first.
- Drop columns only if >30% null and not otherwise fixable.
- For deduplication use key_columns that define a unique event.
- For unit conversions, only convert the region that needs converting.
- Call "finish" when you believe all fixable issues are resolved.

IMPORTANT: Reply with ONLY a valid JSON object. No explanation. No markdown. Just JSON.
""").strip()


# ---------------------------------------------------------------------------
# User prompt builder
# ---------------------------------------------------------------------------


def build_user_prompt(
    obs: Dict[str, Any], step_num: int, max_steps: int
) -> str:
    parts = [
        f"Task: {obs['task_name']} — {obs['task_description']}",
        f"Step {step_num}/{max_steps}  |  Quality: {obs['quality_score']:.4f}",
        f"Shape: {obs['shape']}  |  Last action result: {obs['last_action_result']}",
    ]
    if obs.get("last_action_error"):
        parts.append(f"Last error: {obs['last_action_error']}")

    parts.append("\nColumn statistics:")
    for col, stats in obs["column_stats"].items():
        parts.append(
            f"  {col}: dtype={stats['dtype']} "
            f"nulls={stats['null_count']}({stats['null_pct']:.1%}) "
            f"unique={stats['unique_count']}"
            + (f" mean={stats['mean']}" if stats.get("mean") is not None else "")
            + f" sample={stats['sample_values'][:3]}"
        )

    parts.append("\nSample rows (first 5):")
    for i, row in enumerate(obs.get("sample_rows", [])[:5]):
        parts.append(f"  [{i}] {json.dumps(row, default=str)}")

    parts.append(
        "\nIMPORTANT: Reply with ONLY a valid JSON object. "
        "No explanation. No markdown. Just JSON."
    )
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# JSON action parser
# ---------------------------------------------------------------------------

_JSON_BLOCK = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)


def parse_action(text: str) -> Dict[str, Any]:
    """Best-effort extraction of a JSON action dict from LLM output."""
    text = text.strip()
    # 1. direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # 2. markdown code fence
    m = _JSON_BLOCK.search(text)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass
    # 3. first { … } substring
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            pass
    # 4. safe fallback
    return {"action_type": "inspect"}


# ---------------------------------------------------------------------------
# LLM call
# ---------------------------------------------------------------------------


def call_llm(user_prompt: str) -> str:
    """Send a chat-completion request and return the assistant content."""
    global _client
    if _client is None:
        # Read the proxy URL and key from whichever env var the validator sets.
        # Precedence: API_BASE_URL > OPENAI_BASE_URL  (same for key).
        base_url = (
            os.environ.get("API_BASE_URL")
            or os.environ.get("OPENAI_BASE_URL")
            or ""
        )
        api_key = (
            os.environ.get("API_KEY")
            or os.environ.get("OPENAI_API_KEY")
            or "no-key"
        )
        print(
            f"[DEBUG] ENV DUMP: API_BASE_URL={os.environ.get('API_BASE_URL')!r} "
            f"API_KEY={os.environ.get('API_KEY', '')[:8]}*** "
            f"OPENAI_BASE_URL={os.environ.get('OPENAI_BASE_URL')!r} "
            f"OPENAI_API_KEY={os.environ.get('OPENAI_API_KEY', '')[:8]}***",
            file=sys.stderr,
            flush=True,
        )
        print(
            f"[DEBUG] Resolved: base_url={base_url!r} api_key={api_key[:8]}***",
            file=sys.stderr,
            flush=True,
        )
        # Also set the OPENAI_ env vars so the SDK picks them up internally.
        if base_url:
            os.environ["OPENAI_BASE_URL"] = base_url
        if api_key and api_key != "no-key":
            os.environ["OPENAI_API_KEY"] = api_key
        _client = OpenAI(base_url=base_url, api_key=api_key)
    try:
        print(
            f"[DEBUG] Calling LLM model={MODEL_NAME}",
            file=sys.stderr,
            flush=True,
        )
        response = _client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            stream=False,
        )
        print("[DEBUG] LLM call succeeded", file=sys.stderr, flush=True)
        return response.choices[0].message.content or ""
    except Exception as exc:
        print(f"[DEBUG] LLM request FAILED: {exc}", file=sys.stderr, flush=True)
        traceback.print_exc(file=sys.stderr)
        return '{"action_type": "inspect"}'


# ---------------------------------------------------------------------------
# Environment HTTP helpers
# ---------------------------------------------------------------------------


def env_reset(task_name: str) -> Dict[str, Any]:
    resp = requests.post(
        f"{ENV_URL}/reset", json={"task_name": task_name}, timeout=30
    )
    resp.raise_for_status()
    return resp.json()


def env_step(action: Dict[str, Any]) -> Dict[str, Any]:
    resp = requests.post(f"{ENV_URL}/step", json=action, timeout=30)
    resp.raise_for_status()
    return resp.json()


def env_state() -> Dict[str, Any]:
    resp = requests.get(f"{ENV_URL}/state", timeout=10)
    resp.raise_for_status()
    return resp.json()


# ---------------------------------------------------------------------------
# Run a single task
# ---------------------------------------------------------------------------


def run_task(task_name: str) -> None:
    """Run one task end-to-end, emitting [START], [STEP]…, [END]."""
    max_steps = MAX_STEPS.get(task_name, 10)
    rewards: List[float] = []
    steps_taken: int = 0
    score: float = 0.0
    success: bool = False

    log_start(task=task_name, env=BENCHMARK, model=MODEL_NAME)

    try:
        reset_data = env_reset(task_name)
        obs = reset_data["observation"]
        done: bool = False

        while not done and steps_taken < max_steps:
            user_prompt = build_user_prompt(obs, steps_taken + 1, max_steps)
            raw_response = call_llm(user_prompt)
            action = parse_action(raw_response)
            action_str = json.dumps(action, separators=(",", ":"))

            step_result = env_step(action)
            obs = step_result["observation"]
            reward: float = step_result.get("reward", 0.0) or 0.0
            done = step_result.get("done", False)
            error: Optional[str] = obs.get("last_action_error")
            steps_taken += 1
            rewards.append(reward)

            log_step(
                step=steps_taken,
                action=action_str,
                reward=reward,
                done=done,
                error=error,
            )

            quality: float = obs.get("quality_score", 0.0)
            score = quality

            # Auto-finish when quality is nearly perfect
            if quality >= 0.95 and not done:
                finish_action: Dict[str, Any] = {"action_type": "finish"}
                finish_str = json.dumps(finish_action, separators=(",", ":"))
                step_result = env_step(finish_action)
                obs = step_result["observation"]
                reward = step_result.get("reward", 0.0) or 0.0
                done = True
                error = obs.get("last_action_error")
                steps_taken += 1
                rewards.append(reward)
                score = obs.get("quality_score", quality)

                log_step(
                    step=steps_taken,
                    action=finish_str,
                    reward=reward,
                    done=done,
                    error=error,
                )

        # Final score from /state (authoritative)
        try:
            state = env_state()
            score = state.get("current_quality", score)
        except Exception:
            pass

        score = min(max(score, 0.0), 1.0)
        success = score > 0.0

    except Exception as exc:
        print(
            f"[DEBUG] task {task_name} failed: {exc}",
            file=sys.stderr,
            flush=True,
        )
        traceback.print_exc(file=sys.stderr)

    finally:
        log_end(
            success=success,
            steps=steps_taken,
            score=score,
            rewards=rewards,
        )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    for task_name in TASKS:
        run_task(task_name)


if __name__ == "__main__":
    main()
