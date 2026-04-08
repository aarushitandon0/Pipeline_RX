"""OpenEnv-compliant application entry point for PipelineRx.

This module provides the ``main()`` function expected by ``openenv validate``
and by the ``[project.scripts] server`` entry point in ``pyproject.toml``.

It starts the Uvicorn server hosting the FastAPI application defined in
``server.main``.
"""

from __future__ import annotations

import os


def main(host: str = "0.0.0.0", port: int | None = None) -> None:
    """Launch the PipelineRx FastAPI server via Uvicorn."""
    import uvicorn

    if port is None:
        port = int(os.environ.get("PORT", "7860"))

    uvicorn.run(
        "server.main:app",
        host=host,
        port=port,
        log_level="info",
    )


if __name__ == "__main__":
    main()
