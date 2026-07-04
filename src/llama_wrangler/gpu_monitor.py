"""GPU monitoring via nvidia-smi."""

import asyncio
import logging
import shutil

logger = logging.getLogger(__name__)

# nvidia-smi query fields
_QUERY_FIELDS = [
    "name",
    "memory.used",
    "memory.total",
    "temperature.gpu",
    "utilization.gpu",
    "power.draw",
    "power.max_limit",
]


class GPUMonitor:
    """Polls nvidia-smi for GPU status at regular intervals."""

    def __init__(self, interval: float = 3.0):
        self._interval = interval
        self._latest: dict | None = None
        self._task: asyncio.Task | None = None
        self._subscribers: list[asyncio.Queue] = []
        self._available = shutil.which("nvidia-smi") is not None

    @property
    def available(self) -> bool:
        """Whether nvidia-smi is available on this system."""
        return self._available

    @property
    def latest(self) -> dict | None:
        """Most recent GPU reading, or None if not yet polled."""
        return self._latest

    def start(self) -> None:
        """Start the background polling loop."""
        if not self._available:
            logger.warning("nvidia-smi not found, GPU monitoring disabled")
            return
        if self._task is None or self._task.done():
            self._task = asyncio.create_task(self._poll_loop())
            logger.info("GPU monitor started (interval=%.1fs)", self._interval)

    def stop(self) -> None:
        """Stop the background polling loop."""
        if self._task and not self._task.done():
            self._task.cancel()
            self._task = None

    def subscribe(self) -> asyncio.Queue:
        """Subscribe to GPU status updates."""
        q: asyncio.Queue = asyncio.Queue(maxsize=20)
        self._subscribers.append(q)
        return q

    def unsubscribe(self, q: asyncio.Queue) -> None:
        """Remove a subscriber."""
        try:
            self._subscribers.remove(q)
        except ValueError:
            pass

    async def poll_once(self) -> dict | None:
        """Run a single nvidia-smi query.

        Returns:
            GPU info dict, or None on failure.
        """
        if not self._available:
            return None

        query = ",".join(_QUERY_FIELDS)
        cmd = ["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader,nounits"]

        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=5.0)

            if proc.returncode != 0:
                logger.warning("nvidia-smi failed: %s", stderr.decode().strip())
                return None

            line = stdout.decode().strip()
            if not line:
                return None

            return self._parse_output(line)

        except asyncio.TimeoutError:
            logger.warning("nvidia-smi timed out")
            return None
        except FileNotFoundError:
            self._available = False
            logger.warning("nvidia-smi disappeared from PATH")
            return None
        except Exception:
            logger.exception("GPU monitor error")
            return None

    async def _poll_loop(self) -> None:
        """Background loop that polls nvidia-smi."""
        try:
            while True:
                info = await self.poll_once()
                if info:
                    self._latest = info
                    await self._broadcast(info)
                await asyncio.sleep(self._interval)
        except asyncio.CancelledError:
            pass

    def _parse_output(self, line: str) -> dict:
        """Parse a single CSV line from nvidia-smi."""
        parts = [p.strip() for p in line.split(",")]

        if len(parts) < len(_QUERY_FIELDS):
            # Pad with None for missing fields
            parts.extend([None] * (len(_QUERY_FIELDS) - len(parts)))

        def _float(val: str | None) -> float | None:
            if val is None or val in ("[N/A]", "N/A", ""):
                return None
            try:
                return float(val)
            except ValueError:
                return None

        return {
            "name": parts[0] or "Unknown GPU",
            "vram_used_mb": _float(parts[1]),
            "vram_total_mb": _float(parts[2]),
            "temperature_c": _float(parts[3]),
            "utilization_pct": _float(parts[4]),
            "power_draw_w": _float(parts[5]),
            "power_limit_w": _float(parts[6]),
        }

    async def _broadcast(self, info: dict) -> None:
        """Send GPU info to all subscribers."""
        dead: list[asyncio.Queue] = []
        for q in self._subscribers:
            try:
                q.put_nowait(info)
            except asyncio.QueueFull:
                dead.append(q)
        for q in dead:
            self._subscribers.remove(q)
