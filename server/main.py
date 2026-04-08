"""FastAPI application for PipelineRx RL environment."""

from __future__ import annotations

from typing import Any, Dict, List

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from server.env import PipelineEnv
from server.models import (
    PipelineAction,
    PipelineObservation,
    PipelineState,
    ResetRequest,
    ResetResponse,
    StepResponse,
    TaskInfo,
)
from server.tasks import TASK_REGISTRY

app = FastAPI(
    title="PipelineRx",
    version="1.0.0",
    description="OpenEnv-compliant RL environment for diagnosing and repairing broken data pipelines",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global environment instance (thread-safe via internal lock)
env = PipelineEnv()


# ------------------------------------------------------------------
# Endpoints
# ------------------------------------------------------------------


@app.get("/health")
async def health() -> Dict[str, str]:
    return {"status": "healthy"}


@app.get("/metadata")
async def metadata() -> Dict[str, Any]:
    return {
        "name": "pipelinerx",
        "description": "RL environment for diagnosing and repairing broken data pipelines",
        "version": "1.0.0",
        "author": "PipelineRx Team",
        "license": "MIT",
        "tags": ["data-engineering", "tabular", "multi-task", "deterministic"],
    }


@app.get("/schema")
async def schema() -> Dict[str, Any]:
    return {
        "action": PipelineAction.model_json_schema(),
        "observation": PipelineObservation.model_json_schema(),
        "state": PipelineState.model_json_schema(),
    }


@app.post("/reset", response_model=ResetResponse)
async def reset(body: ResetRequest | None = None) -> Dict[str, Any]:
    task_name = body.task_name if body else "null_sweep"
    try:
        result = env.reset(task_name=task_name)
        return result
    except ValueError as exc:
        # still return 200 with error info per spec
        return {
            "observation": None,
            "done": False,
            "reward": None,
            "info": {"error": str(exc)},
        }


@app.post("/step", response_model=StepResponse)
async def step(action: PipelineAction) -> Dict[str, Any]:
    try:
        result = env.step(action)
        return result
    except RuntimeError as exc:
        return {
            "observation": None,
            "reward": 0.0,
            "done": True,
            "info": {"error": str(exc)},
        }


@app.get("/state", response_model=PipelineState)
async def state() -> PipelineState:
    return env.get_state()


@app.get("/tasks", response_model=List[TaskInfo])
async def tasks() -> List[TaskInfo]:
    result = []
    for name, cls in TASK_REGISTRY.items():
        t = cls()
        result.append(
            TaskInfo(
                name=t.name,
                description=t.description,
                difficulty=t.difficulty,
                max_steps=t.max_steps,
            )
        )
    return result


@app.post("/mcp")
async def mcp(body: Dict[str, Any] | None = None) -> Dict[str, Any]:
    """Minimal JSON-RPC 2.0 MCP endpoint for OpenEnv compatibility."""
    body = body or {}
    req_id = body.get("id", 1)
    method = body.get("method", "")

    if method == "tools/list":
        tools = [
            {
                "name": "reset",
                "description": "Reset the environment with a task name",
                "inputSchema": ResetRequest.model_json_schema(),
            },
            {
                "name": "step",
                "description": "Take an action in the environment",
                "inputSchema": PipelineAction.model_json_schema(),
            },
            {
                "name": "state",
                "description": "Get current episode state",
                "inputSchema": {"type": "object", "properties": {}},
            },
        ]
        return {"jsonrpc": "2.0", "id": req_id, "result": {"tools": tools}}

    if method == "tools/call":
        params = body.get("params", {})
        tool_name = params.get("name", "")
        tool_args = params.get("arguments", {})

        if tool_name == "reset":
            task_name = tool_args.get("task_name", "null_sweep")
            result = env.reset(task_name=task_name)
            return {"jsonrpc": "2.0", "id": req_id, "result": result}
        if tool_name == "step":
            action = PipelineAction(**tool_args)
            result = env.step(action)
            return {"jsonrpc": "2.0", "id": req_id, "result": result}
        if tool_name == "state":
            result = env.get_state().model_dump()
            return {"jsonrpc": "2.0", "id": req_id, "result": result}

        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "error": {"code": -32601, "message": f"Unknown tool: {tool_name}"},
        }

    # Default: return server info
    return {
        "jsonrpc": "2.0",
        "id": req_id,
        "result": {
            "name": "pipelinerx",
            "version": "1.0.0",
            "protocolVersion": "2024-11-05",
        },
    }
