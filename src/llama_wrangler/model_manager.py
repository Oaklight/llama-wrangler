"""GGUF model file management and HuggingFace integration."""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class ModelInfo:
    """Information about a local GGUF model file."""

    filename: str
    size_bytes: int
    modified: float  # Unix timestamp
    path: str

    def to_dict(self) -> dict:
        """Serialize for API response."""
        return {
            "filename": self.filename,
            "size_bytes": self.size_bytes,
            "size_human": _human_size(self.size_bytes),
            "modified": self.modified,
            "path": self.path,
        }


@dataclass
class DownloadTask:
    """Tracks an active model download."""

    repo_id: str
    filename: str
    started_at: float = field(default_factory=time.time)
    progress: float = 0.0  # 0.0 - 1.0
    downloaded_bytes: int = 0
    total_bytes: int = 0
    status: str = "downloading"  # downloading, completed, failed, cancelled
    error: str | None = None
    _cancel_event: asyncio.Event = field(default_factory=asyncio.Event)

    def to_dict(self) -> dict:
        """Serialize for API response."""
        return {
            "repo_id": self.repo_id,
            "filename": self.filename,
            "started_at": self.started_at,
            "progress": self.progress,
            "downloaded_bytes": self.downloaded_bytes,
            "total_bytes": self.total_bytes,
            "status": self.status,
            "error": self.error,
        }


class ModelManager:
    """Manages local GGUF models and HuggingFace downloads."""

    def __init__(self, models_dir: str):
        self.models_dir = Path(models_dir)
        self._downloads: dict[str, DownloadTask] = {}
        self._download_subscribers: list[asyncio.Queue] = []

    def list_models(self) -> list[dict]:
        """Scan models directory for .gguf files.

        Returns:
            List of model info dicts, sorted by filename.
        """
        if not self.models_dir.exists():
            return []

        models = []
        for entry in self.models_dir.iterdir():
            if entry.is_file() and entry.suffix.lower() == ".gguf":
                stat = entry.stat()
                info = ModelInfo(
                    filename=entry.name,
                    size_bytes=stat.st_size,
                    modified=stat.st_mtime,
                    path=str(entry),
                )
                models.append(info.to_dict())

        return sorted(models, key=lambda m: m["filename"])

    def delete_model(self, filename: str) -> bool:
        """Delete a model file safely.

        Args:
            filename: Model filename (must not contain path separators).

        Returns:
            True if deleted successfully.

        Raises:
            ValueError: If filename contains path traversal.
            FileNotFoundError: If model doesn't exist.
        """
        # Path traversal protection
        if "/" in filename or "\\" in filename or ".." in filename:
            raise ValueError(f"Invalid filename: {filename}")

        model_path = self.models_dir / filename
        resolved = model_path.resolve()

        # Ensure the resolved path is still within models_dir
        if not str(resolved).startswith(str(self.models_dir.resolve())):
            raise ValueError(f"Path traversal detected: {filename}")

        if not resolved.exists():
            raise FileNotFoundError(f"Model not found: {filename}")

        resolved.unlink()
        logger.info("Deleted model: %s", filename)
        return True

    def get_model_path(self, filename: str) -> str:
        """Get full path to a model file.

        Raises:
            FileNotFoundError: If model doesn't exist.
        """
        if "/" in filename or "\\" in filename or ".." in filename:
            raise ValueError(f"Invalid filename: {filename}")

        path = self.models_dir / filename
        if not path.exists():
            raise FileNotFoundError(f"Model not found: {filename}")
        return str(path)

    async def search_hf(self, query: str, limit: int = 20) -> list[dict]:
        """Search HuggingFace for GGUF models.

        Args:
            query: Search query string.
            limit: Max results to return.

        Returns:
            List of repo/file info dicts with model metadata.
        """
        from huggingface_hub import HfApi

        api = HfApi()
        results = []

        try:
            models = await asyncio.to_thread(
                api.list_models,
                search=query,
                filter="gguf",
                sort="downloads",
                limit=limit,
                expand=["gguf", "lastModified"],
            )

            for model in models:
                # Extract GGUF metadata if available
                gguf = model.gguf or {}
                param_count = gguf.get("total")
                arch = gguf.get("architecture")
                ctx_len = gguf.get("context_length")
                total_file_size = gguf.get("totalFileSize")

                results.append(
                    {
                        "repo_id": model.id,
                        "author": model.author,
                        "downloads": model.downloads,
                        "likes": model.likes,
                        "pipeline_tag": model.pipeline_tag,
                        "last_modified": model.last_modified.isoformat()
                        if model.last_modified
                        else None,
                        "tags": model.tags or [],
                        "param_count": param_count,
                        "param_count_human": _human_params(param_count) if param_count else None,
                        "architecture": arch,
                        "context_length": ctx_len,
                        "total_file_size": total_file_size,
                        "total_file_size_human": _human_size(total_file_size)
                        if total_file_size
                        else None,
                    }
                )

        except Exception as e:
            logger.exception("HuggingFace search failed")
            raise RuntimeError(f"HuggingFace search failed: {e}") from e

        return results

    async def list_repo_files(self, repo_id: str) -> list[dict]:
        """List GGUF files in a HuggingFace repo.

        Args:
            repo_id: HuggingFace repo ID (e.g., "TheBloke/Llama-2-7B-GGUF").

        Returns:
            List of file info dicts with name and size.
        """
        from huggingface_hub import HfApi

        api = HfApi()

        try:
            files = await asyncio.to_thread(api.list_repo_tree, repo_id, recursive=True)
            gguf_files = []
            for f in files:
                if hasattr(f, "rfilename") and f.rfilename.endswith(".gguf"):
                    gguf_files.append(
                        {
                            "filename": f.rfilename,
                            "size_bytes": f.size,
                            "size_human": _human_size(f.size) if f.size else "unknown",
                        }
                    )
            return sorted(gguf_files, key=lambda f: f["filename"])
        except Exception as e:
            logger.exception("Failed to list repo files: %s", repo_id)
            raise RuntimeError(f"Failed to list repo files: {e}") from e

    async def start_download(self, repo_id: str, filename: str) -> DownloadTask:
        """Start downloading a GGUF model from HuggingFace.

        Args:
            repo_id: HuggingFace repo ID.
            filename: GGUF filename within the repo.

        Returns:
            The DownloadTask for tracking progress.
        """
        key = f"{repo_id}/{filename}"

        if key in self._downloads and self._downloads[key].status == "downloading":
            raise RuntimeError(f"Already downloading: {key}")

        task = DownloadTask(repo_id=repo_id, filename=filename)
        self._downloads[key] = task

        # Start download in background
        asyncio.create_task(self._do_download(task))
        return task

    def cancel_download(self, repo_id: str, filename: str) -> bool:
        """Cancel an active download."""
        key = f"{repo_id}/{filename}"
        task = self._downloads.get(key)
        if task and task.status == "downloading":
            task._cancel_event.set()
            task.status = "cancelled"
            return True
        return False

    def get_downloads(self) -> list[dict]:
        """Get all download tasks (active and recent)."""
        return [t.to_dict() for t in self._downloads.values()]

    def clear_downloads(self, status: str | None = None) -> int:
        """Remove finished download entries from the list.

        Args:
            status: If set, only clear downloads with this status
                    (e.g. "completed", "failed", "cancelled").
                    If None, clear all non-downloading entries.

        Returns:
            Number of entries removed.
        """
        to_remove = []
        for key, task in self._downloads.items():
            if task.status == "downloading":
                continue
            if status is None or task.status == status:
                to_remove.append(key)
        for key in to_remove:
            del self._downloads[key]
        return len(to_remove)

    def subscribe_downloads(self) -> asyncio.Queue:
        """Subscribe to download progress events."""
        q: asyncio.Queue = asyncio.Queue(maxsize=50)
        self._download_subscribers.append(q)
        return q

    def unsubscribe_downloads(self, q: asyncio.Queue) -> None:
        """Remove a download subscriber."""
        try:
            self._download_subscribers.remove(q)
        except ValueError:
            pass

    async def _do_download(self, task: DownloadTask) -> None:
        """Perform the actual download using huggingface_hub."""
        from huggingface_hub import hf_hub_download

        try:
            self.models_dir.mkdir(parents=True, exist_ok=True)

            # huggingface_hub handles progress via callbacks internally
            # We use a thread since hf_hub_download is synchronous
            dest = await asyncio.to_thread(
                hf_hub_download,
                repo_id=task.repo_id,
                filename=task.filename,
                local_dir=str(self.models_dir),
                local_dir_use_symlinks=False,
            )

            task.progress = 1.0
            task.status = "completed"
            logger.info("Download completed: %s → %s", task.filename, dest)

        except Exception as e:
            task.status = "failed"
            task.error = str(e)
            logger.exception("Download failed: %s/%s", task.repo_id, task.filename)

        await self._broadcast_download(task)

    async def _broadcast_download(self, task: DownloadTask) -> None:
        """Send download progress to subscribers."""
        event = task.to_dict()
        dead: list[asyncio.Queue] = []
        for q in self._download_subscribers:
            try:
                q.put_nowait(event)
            except asyncio.QueueFull:
                dead.append(q)
        for q in dead:
            self._download_subscribers.remove(q)


def _human_size(size_bytes: int) -> str:
    """Convert bytes to human-readable string."""
    if size_bytes < 1024:
        return f"{size_bytes} B"
    for unit in ("KB", "MB", "GB", "TB"):
        size_bytes /= 1024
        if size_bytes < 1024:
            return f"{size_bytes:.1f} {unit}"
    return f"{size_bytes:.1f} PB"


def _human_params(count: int) -> str:
    """Convert parameter count to human-readable string (e.g. 0.6B, 7B)."""
    if count >= 1_000_000_000:
        return f"{count / 1_000_000_000:.1f}B"
    if count >= 1_000_000:
        return f"{count / 1_000_000:.0f}M"
    if count >= 1_000:
        return f"{count / 1_000:.0f}K"
    return str(count)
