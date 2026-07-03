"""HTTP server and API routes for llama-deck."""

import asyncio
import importlib.resources
import json
import logging
from pathlib import Path

from llama_deck._vendor.httpserver import App, JSONResponse, StreamingResponse, abort
from llama_deck.config import DeckConfig, save_config
from llama_deck.llama_manager import LlamaManager
from llama_deck.model_manager import ModelManager
from llama_deck.system_monitor import SystemMonitor

logger = logging.getLogger(__name__)

# Cache the index.html content
_index_html: str | None = None


def _load_index_html() -> str:
    """Load index.html from package data."""
    return (
        importlib.resources.files("llama_deck")
        .joinpath("static/index.html")
        .read_text("utf-8")
    )


def create_app(config: DeckConfig, config_path: Path) -> App:
    """Create and configure the llama-deck HTTP application.

    Args:
        config: The deck configuration.
        config_path: Path where config is saved.

    Returns:
        Configured App instance ready to run.
    """
    app = App(max_body_size=10_000_000)

    # Attach managers to app
    app.config = config
    app.config_path = config_path
    app.llama = LlamaManager(config)
    app.sysmon = SystemMonitor(disk_paths=[config.models_dir, "/"])
    app.models = ModelManager(config.models_dir)

    # --- Lifecycle hooks ---

    @app.before_request
    async def _start_monitors(request):
        """Start system monitor on first request (lazy init)."""
        if not hasattr(app, "_monitors_started"):
            app.sysmon.start()
            app._monitors_started = True

    # --- Static / UI ---

    @app.get("/")
    async def serve_index(request):
        """Serve the admin panel SPA."""
        global _index_html
        if _index_html is None:
            _index_html = _load_index_html()
        from llama_deck._vendor.httpserver import Response

        return Response(
            body=_index_html,
            status_code=200,
            content_type="text/html; charset=utf-8",
            headers={"Cache-Control": "no-cache, no-store, must-revalidate"},
        )

    # --- Status ---

    @app.get("/api/status")
    async def get_status(request):
        """Get combined server status, system info, and health."""
        server_status = app.llama.status()
        health = await app.llama.health_check()
        system = app.sysmon.latest or {}

        return {
            "server": server_status,
            "health": health,
            "gpu": system.get("gpu"),
            "cpu": system.get("cpu"),
            "ram": system.get("ram"),
            "disks": system.get("disks"),
            "temperatures": system.get("temperatures"),
        }

    # --- Models ---

    @app.get("/api/models")
    async def list_models(request):
        """List local GGUF model files."""
        return {"models": app.models.list_models()}

    @app.delete("/api/models/<path:name>")
    async def delete_model(request, name):
        """Delete a model file."""
        try:
            app.models.delete_model(name)
            return {"deleted": name}
        except ValueError as e:
            abort(400, str(e))
        except FileNotFoundError as e:
            abort(404, str(e))

    @app.get("/api/models/search")
    async def search_models(request):
        """Search HuggingFace for GGUF models."""
        q = request.query_params.get("q", [""])[0]
        if not q:
            abort(400, "Missing query parameter 'q'")

        try:
            results = await app.models.search_hf(q)
            return {"results": results}
        except RuntimeError as e:
            abort(502, str(e))

    @app.get("/api/models/files")
    async def list_repo_files(request):
        """List GGUF files in a HuggingFace repo."""
        repo = request.query_params.get("repo", [""])[0]
        if not repo:
            abort(400, "Missing query parameter 'repo'")

        try:
            files = await app.models.list_repo_files(repo)
            return {"files": files}
        except RuntimeError as e:
            abort(502, str(e))

    # --- Downloads ---

    @app.post("/api/models/download")
    async def start_download(request):
        """Start downloading a model from HuggingFace."""
        data = request.json()
        repo = data.get("repo_id") or data.get("repo")
        filename = data.get("filename")

        if not repo or not filename:
            abort(400, "Missing 'repo_id' and/or 'filename'")

        try:
            task = await app.models.start_download(repo, filename)
            return {"download": task.to_dict()}, 202
        except RuntimeError as e:
            abort(409, str(e))

    @app.get("/api/downloads")
    async def get_downloads(request):
        """Get active and recent download status."""
        return {"downloads": app.models.get_downloads()}

    # --- Server lifecycle ---

    @app.post("/api/server/start")
    async def start_server(request):
        """Start llama-server with a model."""
        data = request.json()
        model = data.get("model")

        if not model:
            abort(400, "Missing 'model' (filename)")

        try:
            model_path = app.models.get_model_path(model)
        except (ValueError, FileNotFoundError) as e:
            abort(400, str(e))

        # Optional: override server args for this start
        args = None
        if "args" in data:
            from llama_deck.config import ServerArgs

            args = ServerArgs.from_dict(data["args"])

        try:
            await app.llama.start(model_path, args)
            return {"status": "started", "model": model}
        except RuntimeError as e:
            abort(409, str(e))
        except FileNotFoundError as e:
            abort(404, str(e))

    @app.post("/api/server/stop")
    async def stop_server(request):
        """Stop the running llama-server."""
        await app.llama.stop()
        return {"status": "stopped"}

    @app.post("/api/server/restart")
    async def restart_server(request):
        """Restart llama-server, optionally with new model/args."""
        data = request.json() if request.body else {}
        model = data.get("model")
        model_path = None

        if model:
            try:
                model_path = app.models.get_model_path(model)
            except (ValueError, FileNotFoundError) as e:
                abort(400, str(e))

        args = None
        if "args" in data:
            from llama_deck.config import ServerArgs

            args = ServerArgs.from_dict(data["args"])

        try:
            await app.llama.restart(model_path, args)
            return {"status": "restarted"}
        except (RuntimeError, ValueError) as e:
            abort(400, str(e))

    # --- Config ---

    @app.get("/api/config")
    async def get_config(request):
        """Get current configuration."""
        return {"config": app.config.to_dict()}

    @app.post("/api/config")
    async def update_config(request):
        """Update configuration and persist to disk."""
        data = request.json()

        # Update top-level fields
        if "llama_server_path" in data:
            app.config.llama_server_path = data["llama_server_path"]
        if "models_dir" in data:
            app.config.models_dir = data["models_dir"]
            app.models = ModelManager(data["models_dir"])
        if "default_args" in data:
            from llama_deck.config import ServerArgs

            app.config.default_args = ServerArgs.from_dict(data["default_args"])

        # Persist
        save_config(app.config, app.config_path)
        return {"config": app.config.to_dict()}

    # --- Logs ---

    @app.get("/api/logs")
    async def get_logs(request):
        """Get recent log lines from llama-server."""
        n = int(request.query_params.get("n", ["200"])[0])
        n = min(n, 2000)
        return {"logs": app.llama.get_logs(n)}

    # --- SSE Event Stream ---

    @app.get("/api/events")
    async def event_stream(request):
        """Server-Sent Events stream for real-time updates.

        Streams:
        - log: llama-server stdout/stderr lines
        - status: server state changes
        - gpu: GPU stats updates
        - download: download progress updates
        """
        log_q = app.llama.subscribe_logs()
        status_q = app.llama.subscribe_status()
        sys_q = app.sysmon.subscribe()
        dl_q = app.models.subscribe_downloads()

        async def generate():
            try:
                while True:
                    # Check all queues with a short timeout
                    events = []

                    # Drain available events from each queue
                    for event_type, q in [
                        ("log", log_q),
                        ("status", status_q),
                        ("system", sys_q),
                        ("download", dl_q),
                    ]:
                        while True:
                            try:
                                data = q.get_nowait()
                                events.append((event_type, data))
                            except asyncio.QueueEmpty:
                                break

                    if events:
                        for event_type, data in events:
                            yield f"event: {event_type}\ndata: {json.dumps(data)}\n\n"
                    else:
                        # Send keepalive comment every few seconds
                        yield ": keepalive\n\n"

                    await asyncio.sleep(0.5)

            except asyncio.CancelledError:
                pass
            finally:
                app.llama.unsubscribe_logs(log_q)
                app.llama.unsubscribe_status(status_q)
                app.sysmon.unsubscribe(sys_q)
                app.models.unsubscribe_downloads(dl_q)

        return StreamingResponse(
            generate(),
            content_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    return app
