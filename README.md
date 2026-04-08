# PipelineRx 🔧

**OpenEnv-compliant reinforcement learning environment for diagnosing and repairing broken data pipelines.**

An AI agent interacts with a FastAPI server to observe corrupted pandas DataFrames, apply targeted repair actions, and receive dense reward signals based on objective data-quality criteria. Think of it as an on-call data engineer simulator.

---

## Motivation

Silent data pipeline failures — type drift, null injection, duplicate rows, unit mismatches — are the #1 source of production ML model degradation. PipelineRx provides a controlled, deterministic sandbox where RL agents learn to diagnose and fix these issues through a standard HTTP API.

---

## Quick Start

### Docker (recommended)

```bash
docker build -t pipelinerx .
docker run -p 7860:7860 pipelinerx
```

### Local

```bash
pip install -r requirements.txt
uvicorn server.main:app --host 0.0.0.0 --port 8000
```

### Verify

```bash
curl http://localhost:8000/health
# {"status":"ok"}

curl -X POST http://localhost:8000/reset -H "Content-Type: application/json" -d '{"task_name":"null_sweep"}'
```

---

## Endpoints

| Method | Path     | Description                                |
|--------|----------|--------------------------------------------|
| GET    | /health  | Health check                               |
| POST   | /reset   | Reset environment with optional task name  |
| POST   | /step    | Submit an action, get observation + reward  |
| GET    | /state   | Current episode state                      |
| GET    | /tasks   | List available tasks with metadata         |

---

## Observation Space

Each observation includes:

| Field              | Type                | Description                                  |
|--------------------|---------------------|----------------------------------------------|
| step               | int                 | Current step number                          |
| max_steps          | int                 | Budget for this task                         |
| shape              | (rows, cols)        | DataFrame dimensions                         |
| columns            | list[str]           | Column names                                 |
| column_stats       | dict[str, Stats]    | Per-column: null count/pct, dtype, mean, std, min, max, unique, samples |
| sample_rows        | list[dict]          | First 5 rows as dicts                        |
| last_action_result | str                 | "success", "invalid_column", "no_op", etc.   |
| last_action_error  | str \| null         | Error message if applicable                  |
| quality_score      | float               | Current quality 0.0–1.0                      |
| task_name          | str                 | Active task identifier                       |
| task_description   | str                 | Human-readable task description              |

---

## Action Space

| action_type    | Required Fields                                | Description                           |
|----------------|------------------------------------------------|---------------------------------------|
| fill_nulls     | column, params.strategy                        | Fill NaN values (median/mode/forward_fill/zero) |
| cast_column    | column, params.dtype                           | Cast column dtype (float64/int64/bool/datetime) |
| drop_column    | column                                         | Remove a column entirely              |
| deduplicate    | params.key_columns, params.keep                | Drop duplicate rows by key            |
| convert_units  | column, params.source_region, params.conversion| Unit conversion on regional subset    |
| inspect        | —                                              | Re-read observation (costs 1 step)    |
| finish         | —                                              | End episode, trigger final grading    |

---

## Tasks

| Task              | Difficulty | Max Steps | Description                                      |
|-------------------|------------|-----------|--------------------------------------------------|
| null_sweep        | Easy       | 8         | Fix null values in a sensor dataset              |
| type_drift        | Medium     | 10        | Correct type errors in a mixed-format dataset    |
| duplicate_drift   | Medium     | 10        | Deduplicate a CDC-replayed event log             |
| unit_mismatch     | Hard       | 12        | Convert units across EU/US regional data merge   |
| pipeline_cascade  | Very Hard  | 15        | Fix all failure modes in the correct order       |

---

## Baseline Scores

Approximate scores achieved by a Qwen2.5-72B-Instruct agent with zero-shot prompting:

| Task              | Score  |
|-------------------|--------|
| null_sweep        | ~0.85  |
| type_drift        | ~0.72  |
| duplicate_drift   | ~0.68  |
| unit_mismatch     | ~0.55  |
| pipeline_cascade  | ~0.48  |

---

## Inference Script

The `inference.py` script at the project root drives the agent through all 5 tasks:

```bash
export API_BASE_URL="https://your-inference-endpoint/v1"
export HF_TOKEN="hf_..."
export MODEL_NAME="Qwen/Qwen2.5-72B-Instruct"
export ENV_URL="http://localhost:8000"

python inference.py
```

Required environment variables:
- `API_BASE_URL` — OpenAI-compatible inference endpoint (mandatory, no default)
- `HF_TOKEN` or `API_KEY` — Bearer token
- `MODEL_NAME` — Model identifier (default: `Qwen/Qwen2.5-72B-Instruct`)
- `ENV_URL` — PipelineRx server URL (default: `http://localhost:8000`)

---

## Project Structure

```
PipelineRx/
├── Dockerfile
├── openenv.yaml
├── pyproject.toml
├── requirements.txt
├── inference.py
├── README.md
├── server/
│   ├── __init__.py
│   ├── main.py          ← FastAPI app
│   ├── env.py           ← Core environment logic
│   ├── models.py        ← Pydantic models
│   ├── graders.py       ← Final grading per task
│   └── tasks/
│       ├── __init__.py
│       ├── base.py
│       ├── task1_null_sweep.py
│       ├── task2_type_drift.py
│       ├── task3_duplicate_drift.py
│       ├── task4_unit_mismatch.py
│       └── task5_pipeline_cascade.py
```

---

## License

MIT
