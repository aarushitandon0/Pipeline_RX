---
title: PipelineRx
emoji: 🔧
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
---

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
# PipelineRx

PipelineRx is an OpenEnv-compatible reinforcement learning environment for diagnosing and repairing data pipeline failures. An agent interacts with a FastAPI environment over HTTP, observes corrupted pandas DataFrames, issues repair actions, and receives dense rewards based on objective data quality metrics.

## Key features

- Five curated tasks that reflect common pipeline failure modes: null injection, type drift, duplicate records, unit mismatches, and multi-fault cascades
- HTTP API compatible with OpenEnv runtime expectations and tooling
- Deterministic, reproducible episodes that are suitable for reinforcement learning and evaluation
- A reference `inference.py` script that demonstrates how to drive the environment with an LLM

## Quick start

Prerequisites
- Python 3.10 or 3.11
- Docker (optional, recommended for consistent runs)

Using Docker (recommended)

```bash
docker build -t pipelinerx .
docker run -p 7860:7860 pipelinerx
```

Local development

```bash
pip install -r requirements.txt
uvicorn server.main:app --host 0.0.0.0 --port 7860
```

Verify the server

```bash
curl http://localhost:7860/health
```

The command above should return a small JSON object indicating the server is healthy.

Reset an episode

```bash
curl -X POST http://localhost:7860/reset -H "Content-Type: application/json" -d '{"task_name":"null_sweep"}'
```

## HTTP endpoints

The server exposes the following endpoints relevant to agents and evaluators:

- GET `/health` - health check
- POST `/reset` - reset the environment and optionally select a task
- POST `/step` - submit an action, receive observation and reward
- GET `/state` - retrieve current episode state
- GET `/tasks` - list available tasks and metadata
- GET `/metadata` - runtime metadata used by OpenEnv validators
- GET `/schema` - OpenAPI schema
- POST `/mcp` - minimal MCP endpoint expected by some validators

Consult the OpenAPI schema for full request and response shapes.

## Observations

Each observation object contains structured diagnostics of the current DataFrame and episode state. Key fields include:

- `step`: current step index
- `max_steps`: episode budget for the active task
- `shape`: tuple with number of rows and columns
- `columns`: list of column names
- `column_stats`: per-column summary (null counts, dtype, basic numeric stats, unique counts, sample values)
- `sample_rows`: a small list of row dicts for quick inspection
- `last_action_result`: string status for the previous action
- `last_action_error`: optional error message
- `quality_score`: float quality metric between 0.0 and 1.0
- `task_name` and `task_description`: identifiers and human-readable descriptions

## Actions

Supported action types include:

- `fill_nulls` - fill missing values using a strategy (median, mode, forward_fill, zero)
- `cast_column` - cast a column to a target dtype
- `drop_column` - remove a column
- `deduplicate` - drop duplicate rows using key columns
- `convert_units` - perform unit conversion on numeric columns
- `inspect` - re-read the observation without modifying data (consumes a step)
- `finish` - end the episode and trigger final grading

Action payloads follow a simple JSON shape with `action_type`, `column` (when applicable), and a `params` object for additional options.

## Included tasks

The environment contains five tasks with increasing difficulty. Each task defines an initial corruption, an objective quality metric, and a step budget.

- `null_sweep` - fix null values in a dataset (easy, max_steps 8)
- `type_drift` - correct mixed or incorrect dtypes (medium, max_steps 10)
- `duplicate_drift` - deduplicate a CDC-replayed event log (medium, max_steps 10)
- `unit_mismatch` - resolve regional unit mismatches (hard, max_steps 12)
- `pipeline_cascade` - repair multiple failure modes in sequence (very hard, max_steps 15)

## Running the reference agent

The repository includes a reference `inference.py` script that demonstrates how to run all tasks sequentially with an LLM-backed policy. Example usage for bash:

```bash
export API_BASE_URL="https://your-inference-endpoint/v1"
export HF_TOKEN="hf_..."
export MODEL_NAME="Qwen/Qwen2.5-72B-Instruct"
export ENV_URL="http://localhost:7860"

python inference.py
```

PowerShell example

```powershell
$env:API_BASE_URL = 'https://your-inference-endpoint/v1'
$env:HF_TOKEN = 'hf_...'
$env:MODEL_NAME = 'Qwen/Qwen2.5-72B-Instruct'
$env:ENV_URL = 'http://localhost:7860'

python inference.py
```

Required environment variables

- `API_BASE_URL` - OpenAI-compatible inference endpoint. If not provided, a common default router URL may be used by the client library.
- `HF_TOKEN` or `API_KEY` - bearer token for the inference endpoint
- `MODEL_NAME` - model identifier (defaults to `Qwen/Qwen2.5-72B-Instruct` in the reference script)
- `ENV_URL` - URL for the PipelineRx server (default `http://localhost:7860`)

Notes on validator compatibility

The reference `inference.py` script is structured to accept environment-injected values at runtime so that external validators and proxy layers can observe API calls. When running in hosted evaluation environments, do not hardcode credentials. Use the environment variables described above.

## Project layout

```
PipelineRx/
├── Dockerfile
├── openenv.yaml
├── pyproject.toml
├── requirements.txt
├── inference.py
├── README.md
└── server/
	├── main.py
	├── env.py
	├── models.py
	├── graders.py
	└── tasks/
		├── base.py
		├── task1_null_sweep.py
		├── task2_type_drift.py
		├── task3_duplicate_drift.py
		├── task4_unit_mismatch.py
		└── task5_pipeline_cascade.py
```

## License

This project is released under the MIT License.
