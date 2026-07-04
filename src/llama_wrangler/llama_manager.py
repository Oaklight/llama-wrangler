"""llama-server process lifecycle management — multi-instance."""

import asyncio
import json
import logging
import signal
import time
import urllib.error
import urllib.request
import uuid
from collections import deque
from dataclasses import dataclass
from pathlib import Path

from llama_wrangler.config import DeckConfig, ServerArgs

logger = logging.getLogger(__name__)

# Ring buffer size for log lines per instance
LOG_BUFFER_SIZE = 2000


@dataclass
class LogLine:
    """A single log line from llama-server."""

    timestamp: float
    stream: str  # "stdout" or "stderr"
    text: str


class LlamaInstance:
    """Manages a single llama-server subprocess.

    Each instance has its own process, log buffer, and subscribers.
    """

    def __init__(
        self,
        instance_id: str,
        config: DeckConfig,
        *,
        name: str | None = None,
    ):
        self.instance_id = instance_id
        self.config = config
        self.name = name or instance_id[:8]
        self._process: asyncio.subprocess.Process | None = None
        self._logs: deque[LogLine] = deque(maxlen=LOG_BUFFER_SIZE)
        self._log_subscribers: list[asyncio.Queue] = []
        self._status_subscribers: list[asyncio.Queue] = []
        self._current_model: str | None = None
        self._model_path: str | None = None
        self._args: ServerArgs | None = None
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
    def port(self) -> int | None:
        """Port this instance is listening on."""
        return self._args.port if self._args else None

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
            {"timestamp": line.timestamp, "stream": line.stream, "text": line.text} for line in logs
        ]

    def subscribe_logs(self) -> asyncio.Queue:
        """Subscribe to real-time log events."""
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
            raise RuntimeError(f"Instance {self.name} is already running")

        server_path = self.config.llama_server_path
        if not Path(server_path).exists():
            raise FileNotFoundError(f"llama-server not found: {server_path}")

        if not Path(model_path).exists():
            raise FileNotFoundError(f"Model not found: {model_path}")

        if args is None:
            args = self.config.default_args

        self._args = args
        self._model_path = model_path

        cmd = self._build_command(server_path, model_path, args)
        logger.info(
            "Starting instance %s (%s) on port %d: %s",
            self.instance_id,
            self.name,
            args.port,
            " ".join(cmd),
        )

        self._process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        self._current_model = Path(model_path).name
        self._started_at = time.monotonic()

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

        logger.info(
            "Stopping instance %s (%s, PID %d)",
            self.instance_id,
            self.name,
            self._process.pid,
        )

        try:
            self._process.send_signal(signal.SIGTERM)
            try:
                await asyncio.wait_for(self._process.wait(), timeout=10.0)
            except asyncio.TimeoutError:
                logger.warning("Instance %s didn't stop gracefully, killing", self.name)
                self._process.kill()
                await self._process.wait()
        except ProcessLookupError:
            pass

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
        old_model_path = self._model_path
        old_args = self._args

        await self.stop()

        if model_path is None:
            model_path = old_model_path
        if args is None:
            args = old_args

        if model_path is None:
            raise ValueError("No model specified and no previous model to restart with")

        await self.start(model_path, args)

    async def health_check(self) -> dict:
        """Poll llama-server /health endpoint."""
        if not self.is_running or self._args is None:
            return {"status": "not_running"}

        host = "127.0.0.1" if self._args.host == "0.0.0.0" else self._args.host
        url = f"http://{host}:{self._args.port}/health"

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

    @property
    def mode(self) -> str:
        """Instance mode: 'chat', 'embedding', 'reranking', or 'chat'."""
        if self._args:
            if self._args.reranking:
                return "reranking"
            if self._args.embedding:
                return "embedding"
        return "chat"

    def status(self) -> dict:
        """Get current instance status summary."""
        return {
            "instance_id": self.instance_id,
            "name": self.name,
            "running": self.is_running,
            "model": self._current_model,
            "port": self.port,
            "mode": self.mode,
            "uptime": self.uptime,
            "pid": self._process.pid if self._process and self.is_running else None,
            "args": self._args.to_dict() if self._args else None,
        }

    def _build_command(self, server_path: str, model_path: str, args: ServerArgs) -> list[str]:
        """Build the llama-server command line."""
        cmd = [
            server_path,
            "--model",
            model_path,
            "--host",
            args.host,
            "--port",
            str(args.port),
            "--n-gpu-layers",
            str(args.n_gpu_layers),
            "--ctx-size",
            str(args.ctx_size),
            "--batch-size",
            str(args.batch_size),
            "--ubatch-size",
            str(args.ubatch_size),
            "--parallel",
            str(args.parallel),
        ]

        if args.threads > 0:
            cmd.extend(["--threads", str(args.threads)])

        cmd.extend(["--flash-attn", "on" if args.flash_attn else "off"])

        if args.cont_batching:
            cmd.append("--cont-batching")

        if args.metrics:
            cmd.append("--metrics")

        if args.embedding:
            cmd.append("--embedding")

        if args.reranking:
            cmd.append("--reranking")

        if args.pooling:
            cmd.extend(["--pooling", args.pooling])

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
            "instance_id": self.instance_id,
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


class InstanceManager:
    """Registry and lifecycle manager for multiple LlamaInstance objects."""

    def __init__(self, config: DeckConfig):
        self.config = config
        self._instances: dict[str, LlamaInstance] = {}
        # Global subscribers — receive events from ALL instances
        self._global_log_subs: list[asyncio.Queue] = []
        self._global_status_subs: list[asyncio.Queue] = []

    @property
    def running_count(self) -> int:
        """Number of currently running instances."""
        return sum(1 for inst in self._instances.values() if inst.is_running)

    def list_all(self) -> list[dict]:
        """Get status of all instances."""
        return [inst.status() for inst in self._instances.values()]

    def get(self, instance_id: str) -> LlamaInstance:
        """Get an instance by ID.

        Raises:
            KeyError: If instance not found.
        """
        if instance_id not in self._instances:
            raise KeyError(f"Instance not found: {instance_id}")
        return self._instances[instance_id]

    def get_instances_for_model(self, filename: str) -> list[dict]:
        """Get all instances running a specific model."""
        return [
            inst.status()
            for inst in self._instances.values()
            if inst.current_model == filename and inst.is_running
        ]

    async def create(
        self,
        model_path: str,
        args: ServerArgs | None = None,
        name: str | None = None,
    ) -> LlamaInstance:
        """Create and start a new llama-server instance.

        Args:
            model_path: Path to the GGUF model file.
            args: Server arguments. Falls back to config defaults.
            name: Optional display name.

        Returns:
            The created and started LlamaInstance.

        Raises:
            RuntimeError: If port is already in use by another instance.
        """
        if args is None:
            args = ServerArgs.from_dict(self.config.default_args.to_dict())

        # Check port conflict
        for inst in self._instances.values():
            if inst.is_running and inst.port == args.port:
                raise RuntimeError(
                    f"Port {args.port} already in use by instance "
                    f"'{inst.name}' ({inst.instance_id})"
                )

        instance_id = uuid.uuid4().hex[:12]
        instance = LlamaInstance(
            instance_id=instance_id,
            config=self.config,
            name=name,
        )

        # Wire up global subscribers — forward instance events to global queues
        instance._global_log_forwarder = self._make_log_forwarder(instance)
        instance._global_status_forwarder = self._make_status_forwarder(instance)

        self._instances[instance_id] = instance

        try:
            await instance.start(model_path, args)
        except Exception:
            del self._instances[instance_id]
            raise

        return instance

    async def remove(self, instance_id: str) -> None:
        """Stop and remove an instance.

        Raises:
            KeyError: If instance not found.
        """
        instance = self.get(instance_id)
        await instance.stop()
        del self._instances[instance_id]

    async def stop_all(self) -> None:
        """Stop all running instances."""
        tasks = [inst.stop() for inst in self._instances.values() if inst.is_running]
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    # --- Global subscribers (for SSE stream) ---

    def subscribe_logs(self) -> asyncio.Queue:
        """Subscribe to log events from ALL instances."""
        q: asyncio.Queue = asyncio.Queue(maxsize=200)
        self._global_log_subs.append(q)
        return q

    def unsubscribe_logs(self, q: asyncio.Queue) -> None:
        """Remove a global log subscriber."""
        try:
            self._global_log_subs.remove(q)
        except ValueError:
            pass

    def subscribe_status(self) -> asyncio.Queue:
        """Subscribe to status events from ALL instances."""
        q: asyncio.Queue = asyncio.Queue(maxsize=100)
        self._global_status_subs.append(q)
        return q

    def unsubscribe_status(self, q: asyncio.Queue) -> None:
        """Remove a global status subscriber."""
        try:
            self._global_status_subs.remove(q)
        except ValueError:
            pass

    def _make_log_forwarder(self, instance: LlamaInstance) -> asyncio.Queue:
        """Create a per-instance log subscriber that forwards to global subs."""
        q = instance.subscribe_logs()

        async def _forward():
            try:
                while True:
                    event = await q.get()
                    dead: list[asyncio.Queue] = []
                    for gq in self._global_log_subs:
                        try:
                            gq.put_nowait(event)
                        except asyncio.QueueFull:
                            dead.append(gq)
                    for gq in dead:
                        self._global_log_subs.remove(gq)
            except asyncio.CancelledError:
                pass

        asyncio.create_task(_forward())
        return q

    def _make_status_forwarder(self, instance: LlamaInstance) -> asyncio.Queue:
        """Create a per-instance status subscriber that forwards to global subs."""
        q = instance.subscribe_status()

        async def _forward():
            try:
                while True:
                    event = await q.get()
                    dead: list[asyncio.Queue] = []
                    for gq in self._global_status_subs:
                        try:
                            gq.put_nowait(event)
                        except asyncio.QueueFull:
                            dead.append(gq)
                    for gq in dead:
                        self._global_status_subs.remove(gq)
            except asyncio.CancelledError:
                pass

        asyncio.create_task(_forward())
        return q
