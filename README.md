---
title: PipelineRx
emoji: 🔧
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
---

# PipelineRx

PipelineRx is an OpenEnv-compatible diagnostic environment for identifying and repairing common data-pipeline faults in tabular data. The environment exposes an HTTP API (FastAPI) that agents use to reset episodes, inspect data, apply repair actions, and receive graded rewards. The objective for agents is to maximize a task-specific quality score while avoiding destructive or incorrect repairs.

## Key features

- HTTP API for programmatic agent interaction
- Multiple diagnostic tasks with deterministic episodes and task-specific seeds
- Task-specific graders that compute final quality in the range 0.0 to 1.0
- Action space covering common data repair operations
- Observations with per-column statistics, sample rows, and quality indicators

## Environment interface

Base URL: typically http://localhost:7860 during development.

Primary endpoints

- `POST /reset`: Start a new episode. JSON body should include `task_name` and optional `seed`.
- `POST /step`: Submit an action. Returns observation, reward, and episode status.
- `GET /health`: Health probe for the service.
- `GET /metadata` and `GET /schema`: Metadata to support validators and runtime checks.

Example reset request

```json
POST /reset
Content-Type: application/json

{ "task_name": "null_sweep" }
```

Example inspect step

```json
POST /step
Content-Type: application/json

{ "action_type": "inspect" }
```

## Action space

Actions are sent as JSON to `POST /step`. All actions must include `action_type`. The environment validates fields and returns informative `last_action_result` and `last_action_error` in the observation when applicable.

Common fields

- `action_type` (string): `fill_nulls`, `cast_column`, `drop_column`, `deduplicate`, `convert_units`, `inspect`, `finish`.
- `column` (string, optional): Target column when applicable.
- `params` (object, optional): Action-specific parameters.

Supported actions and required parameters

- `fill_nulls`
  - Required: `column`, `params.strategy` (one of `median`, `mode`, `forward_fill`, `zero`).
  - Purpose: Impute missing values in `column` using the specified strategy.

- `cast_column`
  - Required: `column`, `params.dtype` (one of `float64`, `int64`, `bool`, `datetime`).
  - Purpose: Convert a column to a specific dtype. Parsing failures may create nulls.

- `drop_column`
  - Required: `column`.
  - Purpose: Remove the column from the dataset. This is destructive and may be penalized by graders.

- `deduplicate`
  - Required: `params.key_columns` (array of column names).
  - Optional: `params.keep` (`first` or `last`).
  - Purpose: Drop rows that are duplicates with respect to the provided key columns.

- `convert_units`
  - Required: `column` and either `params.conversion` or `params.source_region`.
  - Purpose: Convert numeric values in `column` to a consistent unit system across the dataset.

- `inspect`
  - No additional fields.
  - Purpose: Recompute and return the current observation without mutating the dataset. Consumes one step.

- `finish`
  - No additional fields.
  - Purpose: Terminate the episode and compute the final grader score.

Action examples

Fill nulls by median

```json
{ "action_type": "fill_nulls", "column": "sensor_value", "params": { "strategy": "median" } }
```

Cast a column to datetime

```json
{ "action_type": "cast_column", "column": "timestamp", "params": { "dtype": "datetime" } }
```

Deduplicate on event id

```json
{ "action_type": "deduplicate", "params": { "key_columns": ["event_id"], "keep": "last" } }
```

## Observation space

After every step the environment returns a JSON observation describing the dataset and the agent status. The observation is designed to be compact but diagnostic.

Principal fields

- `step` (int): Current step index starting at 0.
- `max_steps` (int): Episode budget.
- `task_name` (string): Active task identifier.
- `task_description` (string): Brief description of the task and expected goals.
- `shape` (array): [rows, cols].
- `columns` (array[string]): Column names present in the DataFrame.
- `column_stats` (object): Per-column statistics. Typical entries: `null_count`, `null_pct`, `dtype`, `mean`, `std`, `min`, `max`, `unique_count`, and `examples` (a small set of values).
- `sample_rows` (array[object]): A small set of rows for manual inspection.
- `last_action_result` (string): One of `success`, `invalid_column`, `no_op`, `error`.
- `last_action_error` (string|null): Error message for the last action when applicable.
- `quality_score` (float): Task-specific quality score in [0.0, 1.0]. Updated after `finish` or at episode end.

Example (truncated)

```json
{
  "step": 2,
  "max_steps": 8,
  "task_name": "null_sweep",
  "shape": [100, 8],
  "columns": ["ts", "sensor_value", "region"],
  "column_stats": {
    "sensor_value": { "null_count": 24, "null_pct": 0.24, "dtype": "float64", "mean": 12.3 }
  },
  "sample_rows": [{ "ts": "2024-01-01T00:00:00Z", "sensor_value": null }],
  "last_action_result": "success",
  "quality_score": 0.62
}
```

## Tasks

Each task seeds a deterministic corruption into a clean reference dataset. A task provides a `task_description`, a step budget, and a grader that computes a quality score between 0.0 and 1.0. Below are concise but detailed descriptions for each task including objective, typical corruptions, what the grader checks, recommended strategies, and example actions.

### Task 1: null_sweep

Objective

- Reduce missing values and replace sentinel values while preserving the original distribution.

Typical corruptions

- Random null injections across rows and columns.
- Column-specific sentinel values such as `-999`, `unknown`, or empty strings used to represent missing data.

Grader and success criteria

- Measures reduction in overall null fraction and correctness of imputations relative to the clean reference.
- Penalizes imputations that introduce large distributional shifts or unrealistic values.

Recommended strategies

- Use `inspect` to identify columns with high `null_pct` and sentinel patterns.
- Prefer conservative imputations: `median` for numeric columns and `mode` for categorical columns.
- Use `forward_fill` only when temporal continuity is expected and supported by column semantics.
- If a column is irrecoverable and not critical, `drop_column` may be acceptable but will be penalized if the column is important to downstream metrics.

Example actions

```json
{ "action_type": "inspect" }
{ "action_type": "fill_nulls", "column": "sensor_value", "params": { "strategy": "median" } }
{ "action_type": "finish" }
```

Edge cases

- Sentinel strings mixed with valid values require cautious cleaning to avoid removing valid categories.
- Low-cardinality numeric columns may be better treated as categorical for imputation.

### Task 2: type_drift

Objective

- Restore semantically correct dtypes for columns so downstream analysis and conversions succeed.

Typical corruptions

- Numeric values encoded as strings (for example "12.3").
- Timestamps encoded as integers or strings with inconsistent formats.
- Categorical values appearing as floats due to earlier processing.

Grader and success criteria

- Rewards correct casts and penalizes casts that produce many parsing failures or loss of information.
- Partial credit is given for correct parsing of the majority of values.

Recommended strategies

- Use `inspect` to sample and review example values before casting.
- For timestamps, attempt parse with robust formats, then use `fill_nulls` on parsing failures if appropriate.
- Avoid global destructive casts when only a subset of rows require conversion; consider column-specific parsing logic.

Example actions

```json
{ "action_type": "inspect" }
{ "action_type": "cast_column", "column": "timestamp", "params": { "dtype": "datetime" } }
{ "action_type": "fill_nulls", "column": "timestamp", "params": { "strategy": "median" } }
```

Edge cases

- Very heterogeneous timestamp formats may require splitting into multiple parse attempts.
- Converting numerics to integers can truncate or round values; consider `float64` when unsure.

### Task 3: duplicate_drift

Objective

- Remove true duplicate rows while preserving legitimate repeated events and correct chronological order.

Typical corruptions

- Exact duplicates inserted due to retries or ingestion errors.
- Partial duplicates where only a subset of columns matches.
- Near-duplicates with small timestamp shifts.

Grader and success criteria

- Evaluates precision and recall of duplicate removal against a reference. Penalizes over-removal of legitimate rows and under-removal of duplicated rows.

Recommended strategies

- Use `inspect` to identify stable key columns (for example `event_id`) and to verify timestamp formats.
- Use `deduplicate` with conservative `key_columns` first, validate results with `inspect`, and iterate.
- When timestamps are messy, consider casting the timestamp column first to align formats before deduplication.

Example actions

```json
{ "action_type": "inspect" }
{ "action_type": "deduplicate", "params": { "key_columns": ["event_id"], "keep": "last" } }
{ "action_type": "finish" }
```

Edge cases

- Near-duplicates that differ by a small floating timestamp or by non-key fields require custom heuristics.

### Task 4: unit_mismatch

Objective

- Detect mixed unit systems within numeric columns and convert values to a consistent target unit.

Typical corruptions

- Numeric columns that contain values recorded in different unit conventions for subsets of rows (for example centimeters mixed with inches).
- Temperature columns mixing Celsius and Fahrenheit.

Grader and success criteria

- Measures alignment of aggregates and distribution against the clean reference after conversion.
- Partial credit for reducing variance caused by unit inconsistencies.

Recommended strategies

- Use `inspect` to analyze distributions and outliers that may indicate a unit mismatch.
- If available, use `params.source_region` to guide automatic conversion rules, otherwise infer conversion by clustering or threshold heuristics.
- Apply `convert_units` on the identified column and validate the result with `inspect` before finishing.

Example actions

```json
{ "action_type": "inspect" }
{ "action_type": "convert_units", "column": "length", "params": { "conversion": "in_to_cm" } }
{ "action_type": "finish" }
```

Edge cases

- Mixed units and legitimate outliers may be hard to disambiguate automatically; prefer conservative conversions with validation.

### Task 5: pipeline_cascade

Objective

- Recover datasets that contain multiple simultaneous faults. Correct ordering of repairs is critical to reach a high final score.

Typical corruptions

- Combinations of nulls, type drift, duplicates, and unit mismatches across multiple columns.
- Fault interactions where a repair in one column affects the validity of another repair.

Grader and success criteria

- Composite grader that aggregates per-fault metrics. Rewards minimal destructive changes and high final alignment with the clean reference.

Recommended strategies

- Start with `inspect` to build a diagnosis plan.
- Prioritize non-destructive actions that enable safer downstream repairs (for example, cast timestamps before deduplication).
- Use small, verifiable steps and `inspect` between operations to reduce error propagation.

Example action sequence

```json
{ "action_type": "inspect" }
{ "action_type": "cast_column", "column": "timestamp", "params": { "dtype": "datetime" } }
{ "action_type": "deduplicate", "params": { "key_columns": ["event_id"], "keep": "last" } }
{ "action_type": "convert_units", "column": "length", "params": { "conversion": "in_to_cm" } }
{ "action_type": "fill_nulls", "column": "sensor_value", "params": { "strategy": "median" } }
{ "action_type": "finish" }
```

Edge cases

- Order-dependent faults where an early destructive action blocks later correct repairs. Prefer inspection and conservative fixes.

## Setup and usage

Prerequisites

- Python 3.10 or 3.11
- Docker (recommended for reproducible runs)

Install Python dependencies

PowerShell

```powershell
python -m pip install -r requirements.txt
```

Docker

Build the image

```powershell
docker build -t pipelinerx .
```

Run the container (expose port 7860)

```powershell
docker run -p 7860:7860 pipelinerx
```

Local development

Start the server locally with Uvicorn

```powershell
uvicorn server.main:app --host 0.0.0.0 --port 7860
```

Verify server health

```powershell
curl http://localhost:7860/health
```

Start an episode

```powershell
curl -X POST http://localhost:7860/reset -H "Content-Type: application/json" -d '{"task_name":"null_sweep"}'
```

Apply an action

```powershell
curl -X POST http://localhost:7860/step -H "Content-Type: application/json" -d '{"action_type":"fill_nulls","column":"sensor_value","params":{"strategy":"median"}}'
```

## Running the reference agent

The repository includes `inference.py`, a reference script that runs episodes with an LLM-backed or scripted policy. The script reads inference credentials from environment variables.

PowerShell example

```powershell
$env:API_BASE_URL = 'https://your-inference-endpoint/v1'
$env:HF_TOKEN = 'hf_...'
$env:MODEL_NAME = 'your-model-name'
$env:ENV_URL = 'http://localhost:7860'
python inference.py
```

Do not store API credentials in repository files. Use environment variables for all secrets.

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

This project is licensed under the MIT License.
