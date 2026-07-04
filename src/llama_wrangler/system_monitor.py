"""System hardware monitoring — GPU, CPU, RAM, disk, temperatures.

Uses nvidia-smi for GPU stats and psutil for everything else.
Cross-platform where psutil supports it; GPU monitoring requires
nvidia-smi on PATH.
"""

import asyncio
import logging
import os
import shutil

import psutil

logger = logging.getLogger(__name__)

# nvidia-smi query fields
_GPU_QUERY_FIELDS = [
    "name",
    "memory.used",
    "memory.total",
    "temperature.gpu",
    "utilization.gpu",
    "power.draw",
    "power.max_limit",
]


class SystemMonitor:
    """Unified hardware monitor with background polling.

    Polls GPU (nvidia-smi), CPU, RAM, disk, and temperatures at a
    configurable interval. Subscribers receive the full snapshot on
    each tick.
    """

    def __init__(
        self,
        interval: float = 3.0,
        disk_paths: list[str] | None = None,
    ):
        self._interval = interval
        self._disk_paths = disk_paths or ["/"]
        self._latest: dict | None = None
        self._task: asyncio.Task | None = None
        self._subscribers: list[asyncio.Queue] = []
        self._gpu_available = shutil.which("nvidia-smi") is not None

    @property
    def gpu_available(self) -> bool:
        """Whether nvidia-smi is available on this system."""
        return self._gpu_available

    @property
    def latest(self) -> dict | None:
        """Most recent full system snapshot."""
        return self._latest

    def start(self) -> None:
        """Start the background polling loop."""
        if self._task is None or self._task.done():
            self._task = asyncio.create_task(self._poll_loop())
            logger.info(
                "System monitor started (interval=%.1fs, gpu=%s)",
                self._interval,
                self._gpu_available,
            )

    def stop(self) -> None:
        """Stop the background polling loop."""
        if self._task and not self._task.done():
            self._task.cancel()
            self._task = None

    def subscribe(self) -> asyncio.Queue:
        """Subscribe to system status updates."""
        q: asyncio.Queue = asyncio.Queue(maxsize=20)
        self._subscribers.append(q)
        return q

    def unsubscribe(self, q: asyncio.Queue) -> None:
        """Remove a subscriber."""
        try:
            self._subscribers.remove(q)
        except ValueError:
            pass

    # ------------------------------------------------------------------
    # Polling
    # ------------------------------------------------------------------

    async def poll_once(self) -> dict:
        """Collect one full system snapshot.

        Returns:
            Dict with keys: gpu, cpu, ram, disks, temperatures.
        """
        # CPU/RAM/disk/temps are fast enough for asyncio.to_thread
        cpu_ram_disk = await asyncio.to_thread(self._read_cpu_ram_disk)
        gpu = await self._read_gpu() if self._gpu_available else None

        snapshot = {**cpu_ram_disk, "gpu": gpu}
        return snapshot

    async def _poll_loop(self) -> None:
        """Background loop that polls all system metrics."""
        try:
            while True:
                snapshot = await self.poll_once()
                self._latest = snapshot
                await self._broadcast(snapshot)
                await asyncio.sleep(self._interval)
        except asyncio.CancelledError:
            pass

    # ------------------------------------------------------------------
    # CPU / RAM / Disk / Temperatures  (sync, run in thread)
    # ------------------------------------------------------------------

    def _read_cpu_ram_disk(self) -> dict:
        """Read CPU, RAM, disk, and temperature data via psutil.

        This runs in a thread to avoid blocking the event loop
        (psutil.cpu_percent sleeps briefly on first call).
        """
        # CPU
        cpu = {
            "percent": psutil.cpu_percent(interval=0.5),
            "count_logical": psutil.cpu_count(logical=True),
            "count_physical": psutil.cpu_count(logical=False),
            "freq_mhz": None,
            "load_avg": None,
        }
        freq = psutil.cpu_freq()
        if freq:
            cpu["freq_mhz"] = round(freq.current)

        # Load average (Unix only, safe no-op on Windows)
        try:
            load = psutil.getloadavg()
            cpu["load_avg"] = [round(v, 2) for v in load]
        except (AttributeError, OSError):
            pass

        # RAM
        vm = psutil.virtual_memory()
        ram = {
            "total_mb": round(vm.total / 1048576),
            "used_mb": round(vm.used / 1048576),
            "available_mb": round(vm.available / 1048576),
            "percent": vm.percent,
        }

        # Disks — walk up to nearest existing ancestor if path doesn't exist
        disks = []
        seen_paths: set[str] = set()
        for path in self._disk_paths:
            resolved = path
            while resolved and not os.path.exists(resolved):
                resolved = os.path.dirname(resolved)
            if not resolved:
                resolved = "/"
            if resolved in seen_paths:
                continue
            seen_paths.add(resolved)
            try:
                usage = psutil.disk_usage(resolved)
                disks.append(
                    {
                        "path": resolved,
                        "total_gb": round(usage.total / 1073741824, 1),
                        "used_gb": round(usage.used / 1073741824, 1),
                        "free_gb": round(usage.free / 1073741824, 1),
                        "percent": usage.percent,
                    }
                )
            except (FileNotFoundError, PermissionError, OSError):
                pass

        # Temperatures (Linux primarily; safe no-op elsewhere)
        temperatures = {}
        try:
            temps = psutil.sensors_temperatures()
            for chip, entries in temps.items():
                for entry in entries:
                    key = f"{chip}/{entry.label}" if entry.label else chip
                    temperatures[key] = {
                        "current": entry.current,
                        "high": entry.high,
                        "critical": entry.critical,
                    }
        except (AttributeError, OSError):
            # sensors_temperatures not available on this platform
            pass

        return {
            "cpu": cpu,
            "ram": ram,
            "disks": disks,
            "temperatures": temperatures,
        }

    # ------------------------------------------------------------------
    # GPU (async subprocess)
    # ------------------------------------------------------------------

    async def _read_gpu(self) -> dict | None:
        """Run a single nvidia-smi query."""
        query = ",".join(_GPU_QUERY_FIELDS)
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

            return self._parse_gpu_output(line)

        except asyncio.TimeoutError:
            logger.warning("nvidia-smi timed out")
            return None
        except FileNotFoundError:
            self._gpu_available = False
            logger.warning("nvidia-smi disappeared from PATH")
            return None
        except Exception:
            logger.exception("GPU monitor error")
            return None

    @staticmethod
    def _parse_gpu_output(line: str) -> dict:
        """Parse a single CSV line from nvidia-smi."""
        parts = [p.strip() for p in line.split(",")]

        if len(parts) < len(_GPU_QUERY_FIELDS):
            parts.extend([None] * (len(_GPU_QUERY_FIELDS) - len(parts)))

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
        """Send snapshot to all subscribers."""
        dead: list[asyncio.Queue] = []
        for q in self._subscribers:
            try:
                q.put_nowait(info)
            except asyncio.QueueFull:
                dead.append(q)
        for q in dead:
            self._subscribers.remove(q)
