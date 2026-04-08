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

The environment contains five tasks with increasing difficulty. Each task includes an initial corruption pattern, a clear objective, observable diagnostics, and a recommended set of repair actions. The reference grader for each task computes a quality score in the range 0.0 to 1.0. Below are detailed descriptions intended to help agents and human evaluators understand the intent and evaluation criteria for each task.


### null_sweep

- Difficulty: Easy
- Max steps: 8

Initial corruption
- Random and targeted null injections across numeric and categorical columns. Some columns use sentinel values that should be interpreted as missing (for example -999 or 'unknown').

Objective
- Restore meaningful values where possible so that per-column null rates fall below task thresholds and a downstream quality metric (imputation-aware accuracy) improves.

Observation clues
- High null counts in specific columns shown in `column_stats`
- Presence of sentinel values in `sample_rows`
- Quality score initially low due to null-related penalties

Recommended actions
- `fill_nulls` with strategy selected per column (median for continuous data, mode for categorical, forward_fill for time series)
- `drop_column` when a column is mostly missing and not required by the task
- `inspect` to re-check before expensive operations

Reward summary and evaluation
- The grader rewards reductions in null fraction and penalizes imputations that increase distributional mismatch (for example, filling all values with global mean when a column is multimodal).
- Final quality is evaluated by a composite metric that considers null rate, distributional similarity to a clean reference, and whether sentinel values remain.


### type_drift

- Difficulty: Medium
- Max steps: 10

Initial corruption
- Columns contain mixed types or have been coerced incorrectly during ingestion. Examples include numeric-looking strings, timestamps as integers, and categorical labels encoded as floats.

Objective
- Convert columns to semantically correct dtypes while minimizing information loss and preserving value semantics.

Observation clues
- `column_stats` shows dtype inconsistencies and counts of parseable values
- `sample_rows` reveals formatting issues such as trailing characters or multiple date formats

Recommended actions
- `cast_column` to appropriate dtype after validation (for dates use datetime parse with multiple format guesses)
- `fill_nulls` may be needed after casting if parsing fails for some rows
- `inspect` to verify downstream numeric statistics after a cast

Reward summary and evaluation
- The grader rewards correct dtype conversion and penalizes casts that produce many nulls or nonsensical values.
- Evaluation considers downstream metrics such as numeric range plausibility, number of parsing errors introduced, and whether categorical value sets are preserved.


### duplicate_drift

- Difficulty: Medium
- Max steps: 10

Initial corruption
- The dataset contains duplicate records from CDC-style replays, partial duplicate rows, and near-duplicates where only a timestamp or minor field differs.

Objective
- Remove duplicate records while preserving legitimate repeated events and maintaining causal order when required by the task.

Observation clues
- Identical or near-identical `sample_rows`
- `column_stats` may show repeated keys or anomalously high counts for identifiers

Recommended actions
- `deduplicate` with appropriate `key_columns` to identify true duplicates
- Use `inspect` to confirm deduplication reduced duplicates without dropping unique events
- If duplicates are time-delayed repairs, consider `cast_column` on timestamp fields to align formats before deduplication

Reward summary and evaluation
- The grader rewards precision and recall of duplicate elimination: removing duplicates without dropping unique events.
- Final quality considers the reduction in duplicate ratio and the preservation of expected event counts or aggregates.


### unit_mismatch

- Difficulty: Hard
- Max steps: 12

Initial corruption
- Numeric columns hold values from different unit systems (for example, a mix of centimeters and inches or Celsius and Fahrenheit) introduced during regional merges. Units are not explicitly labeled.

Objective
- Detect unit inconsistencies and convert values to a consistent target unit for downstream correctness.

Observation clues
- Statistical anomalies in `column_stats` such as bimodal distributions and outlier clusters
- `sample_rows` may reveal context columns like `region` or `units` that help infer units

Recommended actions
- `convert_units` for the affected column, specifying `source_region` or using inferred heuristics
- `inspect` to validate that summary statistics align after conversion
- `drop_column` if a column is irrecoverably ambiguous and not required

Reward summary and evaluation
- The grader measures how well conversions align the column distribution with a clean reference. Rewards increase when aggregates and value ranges match expected units.
- Partial credit is given for reducing variance introduced by mixed units even if exact mapping is imperfect.


### pipeline_cascade

- Difficulty: Very Hard
- Max steps: 15

Initial corruption
- A cascade of multiple failure modes occurring in one dataset. Examples include null injections, type drift, duplicate events, and unit mismatches combined. The correct repair sequence matters: fixing the wrong issue first can mask or worsen other problems.

Objective
- Identify the full set of faults and apply a sequence of repairs that leads to a high overall quality score. The agent must reason about ordering and trade-offs between actions.

Observation clues
- Mixed indicators from the other tasks: high null rates, dtype inconsistencies, duplicate keys, and multimodal numeric distributions
- `task_description` and `column_stats` contain the richest clues for diagnosis

Recommended actions
- Use `inspect` frequently to re-evaluate the state after each repair
- Combine `fill_nulls`, `cast_column`, `deduplicate`, and `convert_units` as appropriate
- Prefer conservative operations early (for example, `inspect` and light imputation) then apply stronger transformations once the fault sources are isolated

Reward summary and evaluation
- The grader awards a composite quality score that aggregates the per-fault metrics. High scores require both correct repairs and an efficient action sequence.
- The environment penalizes unnecessary destructive actions such as dropping many columns or overly aggressive fills that reduce information content.


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
		PipelineRx

		PipelineRx is an OpenEnv-compatible reinforcement learning environment designed to simulate common data pipeline failures and the repairs that restore data quality. Agents interact with a FastAPI server over HTTP to inspect corrupted pandas DataFrames, apply repair actions, and receive dense rewards based on objective data-quality metrics.

		Environment description

		- Interface: HTTP API (FastAPI) exposing endpoints used by agents and validators. Typical workflow: POST /reset to start an episode, then loop POST /step to apply actions and receive observations and rewards.
		- Episodes: Deterministic and reproducible for a chosen task seed. Each episode ends when the agent issues the `finish` action or the episode reaches the task-specific step budget.
		- Grader: Each task includes a reference grader that computes a final quality score in the range 0.0 to 1.0. The grader evaluates repaired data against task-specific objectives such as null reduction, correct dtype conversions, duplicate elimination, and unit consistency.

		Action space

		Actions are submitted as JSON to POST /step. Every action must include an `action_type` field. Optional fields depend on the action type.

		Common fields

		- `action_type` (string): one of `fill_nulls`, `cast_column`, `drop_column`, `deduplicate`, `convert_units`, `inspect`, `finish`.
		- `column` (string, optional): name of the column the action targets when applicable.
		- `params` (object, optional): additional parameters required by the action.

		Action descriptions

		- `fill_nulls`
			- Required: `column`, `params.strategy` (one of `median`, `mode`, `forward_fill`, `zero`).
			- Behavior: Impute missing values in the specified column according to the strategy.

		- `cast_column`
			- Required: `column`, `params.dtype` (one of `float64`, `int64`, `bool`, `datetime`).
			- Behavior: Attempt to cast the column to the target dtype. Parsing failures may create nulls and affect reward.

		- `drop_column`
			- Required: `column`.
			- Behavior: Remove the column from the dataset. Destructive operation; penalized if critical information is lost.

		- `deduplicate`
			- Required: `params.key_columns` (list of column names). Optional `params.keep` (`first` or `last`).
			- Behavior: Drop duplicate rows determined by the key columns.

		- `convert_units`
			- Required: `column`, `params.conversion` or `params.source_region` to guide conversion.
			- Behavior: Convert numeric values from one unit system to a consistent target unit.

		- `inspect`
			- No additional fields required.
			- Behavior: Re-compute and return the current observation without changing data. Consumes one step.

		- `finish`
			- No additional fields required.
			- Behavior: End the episode and trigger final grading.

		Observation space

		The environment returns a JSON observation after each step. Key fields are:

		- `step` (int): Current step index starting at 0.
		- `max_steps` (int): Episode budget for the active task.
		- `task_name` (string): Active task identifier.
		- `task_description` (string): Human-readable description of the task.
		- `shape` (array): [rows, cols] of the current DataFrame.
		- `columns` (array[string]): Column names present in the DataFrame.
		- `column_stats` (object): Mapping from column name to a stats object. Typical stats include `null_count`, `null_pct`, `dtype`, `mean`, `std`, `min`, `max`, `unique_count`, and `examples` (a few sample values).
		- `sample_rows` (array[object]): Small sample of row dictionaries to aid diagnosis.
		- `last_action_result` (string): One of `success`, `invalid_column`, `no_op`, `error`.
		- `last_action_error` (string|null): Error message when applicable.
		- `quality_score` (float): Current task-specific quality metric between 0.0 and 1.0.

		Setup and usage

		Docker (recommended)

		```bash
		docker build -t pipelinerx .
		docker run -p 7860:7860 pipelinerx
		```

		Local (developer)

		```bash
		pip install -r requirements.txt
		uvicorn server.main:app --host 0.0.0.0 --port 7860
		```

		Verify the server

		```bash
		curl http://localhost:7860/health
		```

		Start an episode

		```bash
		curl -X POST http://localhost:7860/reset -H "Content-Type: application/json" -d '{"task_name":"null_sweep"}'
		```

		Apply an action example

		```bash
		curl -X POST http://localhost:7860/step -H "Content-Type: application/json" -d '{"action_type":"fill_nulls","column":"sensor_value","params":{"strategy":"median"}}'
		```

		Running the reference agent

		The repository includes `inference.py`, a reference script that runs all tasks with an LLM-driven policy. The script expects environment variables for the inference client.

		Bash example

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

		Validator compatibility

		- Do not hardcode API credentials. The reference `inference.py` is designed to use environment-injected credentials so hosted validators can observe requests.
		- The server exposes `/metadata` and `/schema` endpoints to support OpenEnv runtime checks.

		Project layout

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

		License

		This project is licensed under the MIT License.
