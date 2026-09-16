"""HTTP server and API routes for llama-wrangler."""

import asyncio
import importlib.resources
import json
import logging
import re
import time
import urllib.error
import urllib.request
from pathlib import Path

from llama_wrangler._vendor.httpserver import App, StreamingResponse, abort
from llama_wrangler.config import DeckConfig, ServerArgs, save_config
from llama_wrangler.llama_manager import InstanceManager
from llama_wrangler.model_manager import ModelManager
from llama_wrangler.system_monitor import SystemMonitor

logger = logging.getLogger(__name__)

# Cache the index.html content
_index_html: str | None = None

# Cache for GitHub release data
_github_release_cache: dict = {"data": None, "fetched_at": 0.0}
_GITHUB_CACHE_TTL = 3600  # seconds


def _load_index_html() -> str:
    """Load index.html from package data."""
    return (
        importlib.resources.files("llama_wrangler").joinpath("static/index.html").read_text("utf-8")
    )


def create_app(config: DeckConfig, config_path: Path) -> App:
    """Create and configure the llama-wrangler HTTP application.

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
    app.instances = InstanceManager(config)
    app.sysmon = SystemMonitor(disk_paths=[config.models_dir, "/"])
    app.models = ModelManager(config.models_dir)

    # --- Lifecycle hooks ---

    @app.on_startup
    async def _start_monitors():
        app.sysmon.start()

    @app.on_shutdown
    async def _stop_instances():
        logger.info("Stopping all instances")
        await app.instances.stop_all()

    @app.on_shutdown
    async def _stop_monitors():
        app.sysmon.stop()

    # --- Static / UI ---

    @app.get("/")
    async def serve_index(request):
        """Serve the admin panel SPA."""
        global _index_html
        if _index_html is None:
            _index_html = _load_index_html()
        from llama_wrangler._vendor.httpserver import Response

        return Response(
            body=_index_html,
            status_code=200,
            content_type="text/html; charset=utf-8",
            headers={"Cache-Control": "no-cache, no-store, must-revalidate"},
        )

    # --- Status ---

    @app.get("/api/status")
    async def get_status(request):
        """Get system info and all instances summary."""
        system = app.sysmon.latest or {}
        instances = app.instances.list_all()

        # Fetch health for all running instances
        for inst_status in instances:
            if inst_status["running"]:
                inst = app.instances.get(inst_status["instance_id"])
                inst_status["health"] = await inst.health_check()
            else:
                inst_status["health"] = {"status": "not_running"}

        return {
            "instances": instances,
            "running_count": app.instances.running_count,
            "gpu": system.get("gpu"),
            "cpu": system.get("cpu"),
            "ram": system.get("ram"),
            "disks": system.get("disks"),
            "temperatures": system.get("temperatures"),
        }

    # --- Instances ---

    @app.get("/api/instances")
    async def list_instances(request):
        """List all instances with status and health."""
        instances = app.instances.list_all()
        for inst_status in instances:
            if inst_status["running"]:
                inst = app.instances.get(inst_status["instance_id"])
                inst_status["health"] = await inst.health_check()
            else:
                inst_status["health"] = {"status": "not_running"}
        return {
            "instances": instances,
            "running_count": app.instances.running_count,
        }

    @app.post("/api/instances")
    async def create_instance(request):
        """Create and start a new llama-server instance."""
        if not request.body:
            abort(400, "Request body required")
        data = request.json()
        model = data.get("model")

        if not model:
            abort(400, "Missing 'model' (filename)")

        try:
            model_path = app.models.get_model_path(model)
        except (ValueError, FileNotFoundError) as e:
            abort(400, str(e))

        args = None
        if "args" in data:
            args = ServerArgs.from_dict(data["args"])

        name = data.get("name")

        try:
            instance = await app.instances.create(model_path, args, name)
            return {"instance": instance.status()}, 201
        except ValueError as e:
            abort(400, str(e))
        except RuntimeError as e:
            abort(409, str(e))
        except FileNotFoundError as e:
            abort(404, str(e))

    @app.get("/api/instances/<str:instance_id>")
    async def get_instance(request, instance_id):
        """Get specific instance status and health."""
        try:
            inst = app.instances.get(instance_id)
        except KeyError as e:
            abort(404, str(e))

        status = inst.status()
        status["health"] = await inst.health_check()
        return {"instance": status}

    @app.delete("/api/instances/<str:instance_id>")
    async def remove_instance(request, instance_id):
        """Stop and remove an instance."""
        try:
            await app.instances.remove(instance_id)
            return {"removed": instance_id}
        except KeyError as e:
            abort(404, str(e))

    @app.post("/api/instances/<str:instance_id>/stop")
    async def stop_instance(request, instance_id):
        """Stop an instance (keep in registry)."""
        try:
            inst = app.instances.get(instance_id)
        except KeyError as e:
            abort(404, str(e))
        await inst.stop()
        return {"instance": inst.status()}

    @app.post("/api/instances/<str:instance_id>/restart")
    async def restart_instance(request, instance_id):
        """Restart an instance."""
        try:
            inst = app.instances.get(instance_id)
        except KeyError as e:
            abort(404, str(e))

        data = request.json() if request.body else {}
        model_path = None
        if "model" in data:
            try:
                model_path = app.models.get_model_path(data["model"])
            except (ValueError, FileNotFoundError) as e:
                abort(400, str(e))

        args = None
        if "args" in data:
            args = ServerArgs.from_dict(data["args"])

        try:
            await inst.restart(model_path, args)
            return {"instance": inst.status()}
        except (RuntimeError, ValueError) as e:
            abort(400, str(e))

    @app.get("/api/instances/<str:instance_id>/logs")
    async def get_instance_logs(request, instance_id):
        """Get log lines for a specific instance."""
        try:
            inst = app.instances.get(instance_id)
        except KeyError as e:
            abort(404, str(e))
        n = int(request.query_params.get("n", ["200"])[0])
        n = min(n, 2000)
        return {"logs": inst.get_logs(n)}

    # --- Models ---

    @app.get("/api/models")
    async def list_models(request):
        """List local GGUF model files with running instance info."""
        models = app.models.list_models()
        # Annotate with running instances
        for m in models:
            m["instances"] = app.instances.get_instances_for_model(m["filename"])
        return {"models": models}

    @app.delete("/api/models/<path:name>")
    async def delete_model(request, name):
        """Delete a model file."""
        # Check if model is in use
        running = app.instances.get_instances_for_model(name)
        if running:
            names = ", ".join(i["name"] for i in running)
            abort(409, f"Model in use by running instance(s): {names}")

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

    @app.delete("/api/downloads")
    async def clear_downloads(request):
        """Clear finished download entries."""
        status = request.query_params.get("status", [None])[0]
        removed = app.models.clear_downloads(status)
        return {"cleared": removed}

    @app.post("/api/downloads/cancel")
    async def cancel_download(request):
        """Cancel an active download."""
        data = request.json()
        repo = data.get("repo_id")
        filename = data.get("filename")
        if not repo or not filename:
            abort(400, "Missing 'repo_id' and/or 'filename'")
        if app.models.cancel_download(repo, filename):
            return {"cancelled": True}
        abort(404, "Download not found or already finished")

    # --- Config ---

    @app.get("/api/config")
    async def get_config(request):
        """Get current configuration."""
        return {"config": app.config.to_dict()}

    @app.post("/api/config")
    async def update_config(request):
        """Update configuration and persist to disk."""
        data = request.json()

        if "llama_server_path" in data:
            app.config.llama_server_path = data["llama_server_path"]
        if "models_dir" in data:
            app.config.models_dir = data["models_dir"]
            app.models = ModelManager(data["models_dir"])
        if "default_args" in data:
            app.config.default_args = ServerArgs.from_dict(data["default_args"])

        save_config(app.config, app.config_path)
        return {"config": app.config.to_dict()}

    # --- llama-server Version Check ---

    async def _get_installed_version(server_path: str) -> dict:
        """Run llama-server --version and parse the output."""
        try:
            proc = await asyncio.create_subprocess_exec(
                server_path,
                "--version",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=10.0)
            text = stdout.decode("utf-8", errors="replace").strip()
            # Format A: "version: 0.3.0-dev (build 10797, commit 235f4e8)"
            m = re.search(
                r"version:\s+(\S+)\s+\(build\s+(\d+),\s+commit\s+([0-9a-f]+)\)",
                text,
            )
            if m:
                return {
                    "version": m.group(1),
                    "build": int(m.group(2)),
                    "commit": m.group(3),
                }
            # Format B: "version: 9863 (94875285e)"
            m = re.search(r"version:\s+(\d+)\s+\(([0-9a-f]+)\)", text)
            if m:
                return {
                    "version": None,
                    "build": int(m.group(1)),
                    "commit": m.group(2),
                }
            return {"error": "parse_failed", "raw": text}
        except FileNotFoundError:
            return {"error": "not_found"}
        except asyncio.TimeoutError:
            return {"error": "timeout"}
        except Exception as e:
            return {"error": str(e)}

    async def _get_latest_release() -> dict:
        """Fetch latest llama.cpp release from GitHub, with caching.

        Tries /releases/latest first (stable releases). If the tag is not a
        bNNNNN build tag, falls back to listing recent releases to find the
        newest prerelease with a bNNNNN tag — llama.cpp cuts frequent
        prereleases with build numbers that match llama-server --version.
        """
        now = time.time()
        if (
            _github_release_cache["data"] is not None
            and now - _github_release_cache["fetched_at"] < _GITHUB_CACHE_TTL
        ):
            return _github_release_cache["data"]

        def _parse_release(data: dict) -> dict:
            tag = data.get("tag_name", "")
            m = re.match(r"b(\d+)", tag)
            return {
                "tag": tag,
                "build": int(m.group(1)) if m else None,
                "url": data.get("html_url", ""),
            }

        def _fetch() -> dict:
            base = "https://api.github.com/repos/ggml-org/llama.cpp/releases"
            headers = {
                "Accept": "application/vnd.github+json",
                "User-Agent": "llama-wrangler",
            }
            try:
                # Try /releases/latest (non-prerelease)
                req = urllib.request.Request(f"{base}/latest", headers=headers)
                with urllib.request.urlopen(req, timeout=15) as resp:
                    result = _parse_release(json.loads(resp.read()))
                if result["build"] is not None:
                    return result

                # Stable release has no build number — scan recent prereleases
                req = urllib.request.Request(
                    f"{base}?per_page=5",
                    headers=headers,
                )
                with urllib.request.urlopen(req, timeout=15) as resp:
                    releases = json.loads(resp.read())
                for rel in releases:
                    parsed = _parse_release(rel)
                    if parsed["build"] is not None:
                        return parsed
                # No bNNNNN tag found at all — return the stable release info
                return result
            except Exception as e:
                return {"error": str(e)}

        result = await asyncio.to_thread(_fetch)
        if "error" not in result:
            _github_release_cache["data"] = result
            _github_release_cache["fetched_at"] = now
        return result

    @app.get("/api/llama-server/version")
    async def get_llama_server_version(request):
        """Get installed llama-server version and latest available release."""
        installed = await _get_installed_version(app.config.llama_server_path)
        latest = await _get_latest_release()

        update_available = False
        if "error" not in installed and "error" not in latest and latest.get("build") is not None:
            update_available = latest["build"] > installed["build"]

        return {
            "installed": installed,
            "latest": latest,
            "update_available": update_available,
        }

    # --- SSE Event Stream ---

    @app.get("/api/events")
    async def event_stream(request):
        """Server-Sent Events stream for real-time updates.

        Streams:
        - log: llama-server stdout/stderr lines (tagged with instance_id)
        - status: instance state changes (tagged with instance_id)
        - system: GPU/CPU/RAM/disk updates
        - download: download progress updates
        """
        log_q = app.instances.subscribe_logs()
        status_q = app.instances.subscribe_status()
        sys_q = app.sysmon.subscribe()
        dl_q = app.models.subscribe_downloads()

        async def generate():
            try:
                while True:
                    events = []

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
                        yield ": keepalive\n\n"

                    await asyncio.sleep(0.5)

            except asyncio.CancelledError:
                pass

        def unsubscribe_all():
            for fn, q in [
                (app.instances.unsubscribe_logs, log_q),
                (app.instances.unsubscribe_status, status_q),
                (app.sysmon.unsubscribe, sys_q),
                (app.models.unsubscribe_downloads, dl_q),
            ]:
                try:
                    fn(q)
                except Exception:
                    logger.debug("Failed to unsubscribe %s", fn, exc_info=True)

        return StreamingResponse(
            generate(),
            content_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
            background=unsubscribe_all,
        )

    return app
