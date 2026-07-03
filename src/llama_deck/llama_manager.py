"""llama-server process lifecycle management."""

import asyncio
import json
import logging
import signal
import time
import urllib.error
import urllib.request
from collections import deque
from dataclasses import dataclass
from pathlib import Path

from llama_deck.config import DeckConfig, ServerArgs

logger = logging.getLogger(__name__)

# Ring buffer size for log lines
LOG_BUFFER_SIZE = 2000


@dataclass
class LogLine:
    """A single log line from llama-server."""

    timestamp: float
    stream: str  # "stdout" or "stderr"
    text: str


class LlamaManager:
    """Manages the llama-server subprocess lifecycle."""

    def __init__(self, config: DeckConfig):
        self.config = config
        self._process: asyncio.subprocess.Process | None = None
        self._logs: deque[LogLine] = deque(maxlen=LOG_BUFFER_SIZE)
        self._log_subscribers: list[asyncio.Queue] = []
        self._status_subscribers: list[asyncio.Queue] = []
        self._current_model: str | None = None
        self._started_at: float | None = None
        self._reader_tasks: list[asyncio.Task] = []

    @property
    def is_running(self) -> bool:
        """Check if llama-server process is alive."""
        return self._process is not None and self._process.returncode is None

    @property
    def current_model(self) -> str | None:
        """Currently loaded model filename."""
        return self._current_model

    @property
    def uptime(self) -> float | None:
        """Seconds since server started, or None if not running."""
        if self._started_at and self.is_running:
            return time.monotonic() - self._started_at
        return None

    def get_logs(self, n: int = 200) -> list[dict]:
        """Get last N log lines."""
        logs = list(self._logs)[-n:]
        return [
            {"timestamp": l.timestamp, "stream": l.stream, "text": l.text}
            for l in logs
        ]

    def subscribe_logs(self) -> asyncio.Queue:
        """Subscribe to real-time log events. Returns a queue to read from."""
        q: asyncio.Queue = asyncio.Queue(maxsize=100)
        self._log_subscribers.append(q)
        return q

    def unsubscribe_logs(self, q: asyncio.Queue) -> None:
        """Remove a log subscriber."""
        try:
            self._log_subscribers.remove(q)
        except ValueError:
            pass

    def subscribe_status(self) -> asyncio.Queue:
        """Subscribe to status change events."""
        q: asyncio.Queue = asyncio.Queue(maxsize=50)
        self._status_subscribers.append(q)
        return q

    def unsubscribe_status(self, q: asyncio.Queue) -> None:
        """Remove a status subscriber."""
        try:
            self._status_subscribers.remove(q)
        except ValueError:
            pass

    async def start(
        self,
        model_path: str,
        args: ServerArgs | None = None,
    ) -> None:
        """Start llama-server with the given model.

        Args:
            model_path: Path to the GGUF model file.
            args: Server arguments. Falls back to config defaults.
        """
        if self.is_running:
            raise RuntimeError("llama-server is already running")

        server_path = self.config.llama_server_path
        if not Path(server_path).exists():
            raise FileNotFoundError(f"llama-server not found: {server_path}")

        if not Path(model_path).exists():
            raise FileNotFoundError(f"Model not found: {model_path}")

        if args is None:
            args = self.config.default_args

        cmd = self._build_command(server_path, model_path, args)
        logger.info("Starting llama-server: %s", " ".join(cmd))

        self._process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        self._current_model = Path(model_path).name
        self._started_at = time.monotonic()

        # Start stdout/stderr reader tasks
        self._reader_tasks = [
            asyncio.create_task(self._read_stream(self._process.stdout, "stdout")),
            asyncio.create_task(self._read_stream(self._process.stderr, "stderr")),
            asyncio.create_task(self._wait_process()),
        ]

        await self._notify_status("started")

    async def stop(self) -> None:
        """Stop the running llama-server process."""
        if not self.is_running or self._process is None:
            return

        logger.info("Stopping llama-server (PID %d)", self._process.pid)

        # Try graceful shutdown first
        try:
            self._process.send_signal(signal.SIGTERM)
            try:
                await asyncio.wait_for(self._process.wait(), timeout=10.0)
            except asyncio.TimeoutError:
                logger.warning("llama-server didn't stop gracefully, killing")
                self._process.kill()
                await self._process.wait()
        except ProcessLookupError:
            pass  # Already dead

        # Cancel reader tasks
        for task in self._reader_tasks:
            task.cancel()
        self._reader_tasks.clear()

        self._current_model = None
        self._started_at = None
        await self._notify_status("stopped")

    async def restart(
        self,
        model_path: str | None = None,
        args: ServerArgs | None = None,
    ) -> None:
        """Restart llama-server, optionally with new model/args."""
        old_model = None
        if self.is_running and model_path is None:
            # Reuse current model
            old_model = self._current_model

        await self.stop()

        if model_path is None and old_model:
            model_path = str(Path(self.config.models_dir) / old_model)

        if model_path is None:
            raise ValueError("No model specified and no previous model to restart with")

        await self.start(model_path, args)

    async def health_check(self) -> dict:
        """Poll llama-server /health endpoint.

        Returns:
            Health status dict with 'status' key.
        """
        if not self.is_running:
            return {"status": "not_running"}

        args = self.config.default_args
        host = "127.0.0.1" if args.host == "0.0.0.0" else args.host
        url = f"http://{host}:{args.port}/health"

        def _check() -> dict:
            try:
                req = urllib.request.Request(url)
                with urllib.request.urlopen(req, timeout=5) as resp:
                    if resp.status == 200:
                        return json.loads(resp.read())
                    return {"status": "error", "http_status": resp.status}
            except urllib.error.URLError:
                return {"status": "loading"}
            except Exception as e:
                return {"status": "error", "detail": str(e)}

        return await asyncio.to_thread(_check)

    def status(self) -> dict:
        """Get current server status summary."""
        return {
            "running": self.is_running,
            "model": self._current_model,
            "uptime": self.uptime,
            "pid": self._process.pid if self._process and self.is_running else None,
        }

    def _build_command(
        self, server_path: str, model_path: str, args: ServerArgs
    ) -> list[str]:
        """Build the llama-server command line."""
        cmd = [
            server_path,
            "--model", model_path,
            "--host", args.host,
            "--port", str(args.port),
            "--n-gpu-layers", str(args.n_gpu_layers),
            "--ctx-size", str(args.ctx_size),
            "--batch-size", str(args.batch_size),
            "--ubatch-size", str(args.ubatch_size),
            "--parallel", str(args.parallel),
        ]

        if args.threads > 0:
            cmd.extend(["--threads", str(args.threads)])

        if args.flash_attn:
            cmd.append("--flash-attn")

        if args.cont_batching:
            cmd.append("--cont-batching")

        if args.metrics:
            cmd.append("--metrics")

        return cmd

    async def _read_stream(
        self,
        stream: asyncio.StreamReader | None,
        name: str,
    ) -> None:
        """Read lines from a subprocess stream and buffer them."""
        if stream is None:
            return

        try:
            while True:
                line_bytes = await stream.readline()
                if not line_bytes:
                    break
                text = line_bytes.decode("utf-8", errors="replace").rstrip("\n")
                log_line = LogLine(
                    timestamp=time.time(),
                    stream=name,
                    text=text,
                )
                self._logs.append(log_line)
                await self._broadcast_log(log_line)
        except asyncio.CancelledError:
            pass

    async def _wait_process(self) -> None:
        """Wait for process to exit and update status."""
        if self._process is None:
            return
        try:
            await self._process.wait()
            self._current_model = None
            self._started_at = None
            await self._notify_status("exited")
        except asyncio.CancelledError:
            pass

    async def _broadcast_log(self, log_line: LogLine) -> None:
        """Send a log line to all subscribers."""
        event = {
            "timestamp": log_line.timestamp,
            "stream": log_line.stream,
            "text": log_line.text,
        }
        dead: list[asyncio.Queue] = []
        for q in self._log_subscribers:
            try:
                q.put_nowait(event)
            except asyncio.QueueFull:
                dead.append(q)
        for q in dead:
            self._log_subscribers.remove(q)

    async def _notify_status(self, event: str) -> None:
        """Broadcast a status change event."""
        status = {**self.status(), "event": event}
        dead: list[asyncio.Queue] = []
        for q in self._status_subscribers:
            try:
                q.put_nowait(status)
            except asyncio.QueueFull:
                dead.append(q)
        for q in dead:
            self._status_subscribers.remove(q)
