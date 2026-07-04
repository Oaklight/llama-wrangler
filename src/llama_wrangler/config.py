"""Configuration management for llama-wrangler."""

import json
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_CONFIG_DIR = Path.home() / ".config" / "llama-wrangler"
DEFAULT_CONFIG_PATH = DEFAULT_CONFIG_DIR / "config.json"


@dataclass
class ServerArgs:
    """Default arguments for llama-server."""

    host: str = "0.0.0.0"
    port: int = 8080
    n_gpu_layers: int = 99
    ctx_size: int = 8192
    flash_attn: bool = True
    batch_size: int = 2048
    ubatch_size: int = 512
    threads: int = 0  # 0 = auto-detect
    parallel: int = 1
    cont_batching: bool = True
    metrics: bool = True
    # Embedding / reranking mode
    embedding: bool = False
    reranking: bool = False
    pooling: str = ""  # none, mean, cls, last, rank (empty = model default)

    def to_dict(self) -> dict:
        """Serialize to dict."""
        return {
            "host": self.host,
            "port": self.port,
            "n_gpu_layers": self.n_gpu_layers,
            "ctx_size": self.ctx_size,
            "flash_attn": self.flash_attn,
            "batch_size": self.batch_size,
            "ubatch_size": self.ubatch_size,
            "threads": self.threads,
            "parallel": self.parallel,
            "cont_batching": self.cont_batching,
            "metrics": self.metrics,
            "embedding": self.embedding,
            "reranking": self.reranking,
            "pooling": self.pooling,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ServerArgs":
        """Create from dict, ignoring unknown keys."""
        known = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in data.items() if k in known})


@dataclass
class DeckConfig:
    """Top-level configuration for llama-wrangler."""

    llama_server_path: str = "/opt/llama.cpp/build/bin/llama-server"
    models_dir: str = "/mnt/data/models"
    default_args: ServerArgs = field(default_factory=ServerArgs)

    def to_dict(self) -> dict:
        """Serialize to dict."""
        return {
            "llama_server_path": self.llama_server_path,
            "models_dir": self.models_dir,
            "default_args": self.default_args.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "DeckConfig":
        """Create from dict."""
        args_data = data.get("default_args", {})
        return cls(
            llama_server_path=data.get("llama_server_path", cls.llama_server_path),
            models_dir=data.get("models_dir", cls.models_dir),
            default_args=ServerArgs.from_dict(args_data) if args_data else ServerArgs(),
        )


def load_config(path: str | None = None) -> tuple["DeckConfig", Path]:
    """Load config from file, falling back to defaults.

    Args:
        path: Explicit config path, or None to auto-discover.

    Returns:
        Tuple of (config, resolved_path).
    """
    if path:
        config_path = Path(path)
    else:
        config_path = DEFAULT_CONFIG_PATH

    if config_path.exists():
        with open(config_path) as f:
            data = json.load(f)
        return DeckConfig.from_dict(data), config_path

    # Return defaults, config will be saved on first write
    return DeckConfig(), config_path


def save_config(config: "DeckConfig", path: Path) -> None:
    """Atomically save config to JSON file.

    Uses write-to-temp + rename for crash safety.
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    data = config.to_dict()
    content = json.dumps(data, indent=2) + "\n"

    # Atomic write: temp file in same dir, then rename
    fd, tmp_path = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            f.write(content)
        os.replace(tmp_path, path)
    except BaseException:
        # Clean up temp file on failure
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise
