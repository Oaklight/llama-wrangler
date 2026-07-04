# llama-wrangler

Lightweight web admin panel for [llama.cpp](https://github.com/ggerganov/llama.cpp) server management.

## Features

- **Model Browser** — Scan local directory for `.gguf` files, view name/size/modified
- **Model Download** — Search HuggingFace for GGUF models, download with progress tracking
- **Server Lifecycle** — Start/stop/restart `llama-server` subprocess from the browser
- **Parameter Config** — Visual editor for llama-server flags (context size, GPU layers, batch size, flash attention, etc.)
- **System Monitoring** — Real-time GPU (VRAM, temp, utilization, power), CPU, RAM, and disk usage
- **Log Viewer** — Stream llama-server stdout/stderr via Server-Sent Events
- **Health Monitoring** — Poll `/health` endpoint, show status badge
- **i18n** — English and Chinese interface, switchable at runtime

## Install

```bash
pip install llama-wrangler
```

### Prerequisites

llama-wrangler manages a `llama-server` process on the host machine. Make sure you have:

- **llama.cpp** compiled with `llama-server` binary ([build instructions](https://github.com/ggerganov/llama.cpp#build))
- **NVIDIA GPU driver** installed (for GPU inference and monitoring)

## Quick Start

```bash
# Start the admin panel
llama-wrangler --host 0.0.0.0 --port 7860

# With custom config
llama-wrangler --config /path/to/config.json
```

Then open `http://localhost:7860` in your browser.

## Configuration

Config is stored at `~/.config/llama-wrangler/config.json`:

```json
{
  "llama_server_path": "/path/to/llama-server",
  "models_dir": "/path/to/models",
  "default_args": {
    "host": "0.0.0.0",
    "port": 8080,
    "n_gpu_layers": 99,
    "ctx_size": 8192,
    "flash_attn": true,
    "batch_size": 2048,
    "ubatch_size": 512,
    "threads": 0,
    "parallel": 1,
    "cont_batching": true,
    "metrics": true
  }
}
```

## Docker

### Host prerequisites

The following must be set up on the **host machine** before running the container:

1. **NVIDIA GPU driver** — install from [NVIDIA](https://www.nvidia.com/drivers) or your distro's package manager
2. **NVIDIA Container Toolkit** — required for `--gpus` flag to work:
    ```bash
    # Ubuntu/Debian
    curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | \
      sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
    curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
      sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
      sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
    sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
    sudo nvidia-ctk runtime configure --runtime=docker
    sudo systemctl restart docker
    ```
    See the [official install guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) for other distros.
3. **llama.cpp** compiled on the host with `llama-server` binary
4. **Verify** everything works: `docker run --rm --gpus all ubuntu nvidia-smi`

### Build and run

```bash
# Build
docker build -t llama-wrangler .

# Run
docker run --gpus all -p 7860:7860 \
  -v /path/to/models:/mnt/data/models \
  -v /path/to/llama-server:/opt/llama-server:ro \
  -v /sys:/sys:ro \
  -v ~/.config/llama-wrangler:/root/.config/llama-wrangler \
  llama-wrangler
```

Volume mounts explained:

| Mount | Purpose |
|-------|---------|
| `-v /path/to/models:/mnt/data/models` | GGUF model files (read/write for downloads) |
| `-v /path/to/llama-server:/opt/llama-server:ro` | llama-server binary from host |
| `-v /sys:/sys:ro` | Sensor data (disk/NVMe temperatures via psutil) |
| `-v ~/.config/llama-wrangler:...` | Persist configuration across restarts |
| `--gpus all` | GPU access (nvidia-smi, CUDA for llama-server) |

> **Note**: CPU and RAM metrics work out of the box in Docker — psutil reads `/proc` which is shared from the host. GPU monitoring requires `--gpus all` via nvidia-container-toolkit.

### Without GPU

llama-wrangler works without a GPU (CPU-only inference). Simply omit `--gpus all`:

```bash
docker run -p 7860:7860 \
  -v /path/to/models:/mnt/data/models \
  -v /path/to/llama-server:/opt/llama-server:ro \
  llama-wrangler
```

The GPU section on the dashboard will be hidden automatically.

## Tech Stack

- **Backend**: Python asyncio (zero-framework, vendored HTTP server)
- **Frontend**: Single-file vanilla HTML/CSS/JS
- **Dependencies**: `huggingface-hub`, `psutil` only
- **No**: Flask, FastAPI, React, npm, database

## License

MIT
