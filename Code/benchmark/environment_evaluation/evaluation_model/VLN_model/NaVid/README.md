# NaVid Server for SAGE-3D Benchmark

This directory contains the server script for running NaVid (Navigation Video) model inference as part of the SAGE-3D benchmark evaluation.

## Overview

NaVid is a video-based Vision-Language Navigation model that uses historical video observations to make navigation decisions.

- **NaVid Repository**: https://github.com/jzhzhang/NaVid-VLN-CE
- **Model Weights**: https://huggingface.co/jzhzhang/navid-7b-full-224-video-fps-1-grid-2-r2r-rxr-training-split

## Prerequisites

- **Hardware**: NVIDIA GPU with at least 24GB VRAM (e.g., RTX 3090, RTX 4090, A100)
- **Software**: Ubuntu 20.04+, CUDA 11.8+, Python 3.8+

## Installation

### Step 1: Clone NaVid-VLN-CE Repository

```bash
git clone https://github.com/jzhzhang/NaVid-VLN-CE.git
cd NaVid-VLN-CE
```

### Step 2: Setup Environment

Follow the [NaVid installation guide](https://github.com/jzhzhang/NaVid-VLN-CE#installation):

```bash
# Create conda environment
conda create -n NaVid python=3.8
conda activate NaVid

# Install PyTorch (adjust CUDA version as needed)
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118

# Install NaVid dependencies
pip install -r requirements.txt
```

### Step 3: Download Model Weights

#### NaVid Model Checkpoint

```bash
# Option 1: Using huggingface-cli
huggingface-cli download jzhzhang/navid-7b-full-224-video-fps-1-grid-2-r2r-rxr-training-split \
    --local-dir /path/to/navid-model

# Option 2: Using git lfs
git lfs install
git clone https://huggingface.co/jzhzhang/navid-7b-full-224-video-fps-1-grid-2-r2r-rxr-training-split /path/to/navid-model
```

#### EVA-ViT-G Weights

```bash
wget https://storage.googleapis.com/sfr-vision-language-research/LAVIS/models/BLIP2/eva_vit_g.pth -O /path/to/eva_vit_g.pth
```

### Step 4: Configure and Copy Server Script

1. Copy the server files to NaVid project:

```bash
cp navid_server.py /path/to/NaVid-VLN-CE/
cp start_server.sh /path/to/NaVid-VLN-CE/
```

2. Edit `start_server.sh` to configure paths:

```bash
# Edit these paths in start_server.sh
MODEL_PATH="/path/to/navid-7b-full-224-video-fps-1-grid-2-r2r-rxr-training-split"
EVA_VIT_PATH="/path/to/eva_vit_g.pth"
NAVID_PROJECT_PATH="/path/to/NaVid-VLN-CE"
```

## Usage

### Starting the Server

```bash
# Navigate to NaVid project directory
cd /path/to/NaVid-VLN-CE

# Make script executable
chmod +x start_server.sh

# Start the server
./start_server.sh
```

Or run directly with Python:

```bash
conda activate NaVid
export PYTHONPATH=$PYTHONPATH:/path/to/NaVid-VLN-CE

python navid_server.py \
    --model_path /path/to/navid-model \
    --eva_vit_path /path/to/eva_vit_g.pth \
    --port 4321
```

### Server Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--model_path` | (required) | Path to NaVid model checkpoint directory |
| `--eva_vit_path` | (required) | Path to EVA-ViT-G weights file |
| `--host` | `localhost` | Host address to bind the server |
| `--port` | `4321` | Port number to listen on |
| `--device` | `cuda` | Device to run on (`cuda` or `cpu`) |

## Running SAGE-3D Benchmark with NaVid

Once the server is running, start the SAGE-3D benchmark in a **separate terminal**:

```bash
# Terminal 2: Run SAGE-3D Benchmark
cd /path/to/SAGE-3D_Official

python Code/benchmark/environment_evaluation/run_benchmark.py \
    --scene_usd_path /path/to/scenes \
    --batch_test_dir /path/to/test/data \
    --output_root /path/to/output \
    --vlm-port 4321 \
    --input-type rgb \
    --protocol socket
```

## Communication Protocol

The server uses TCP socket communication:

### Request Format
1. **Size Header**: 8 bytes (big-endian) indicating JSON payload size
2. **JSON Payload**:
```json
{
    "images": ["base64_encoded_image_1", "base64_encoded_image_2", ...],
    "instruction": "Navigate to the red chair in the living room.",
    "current_yaw": 0.0
}
```

### Response Format
1. **Size Header**: 8 bytes (big-endian) indicating JSON payload size
2. **JSON Payload**:
```json
{
    "vx": 0.25,
    "vy": 0.0,
    "yaw_rate": 0.0,
    "duration_s": 1.0,
    "stop": false,
    "raw_response": "Move forward 50 units",
    "text_response": "Move forward 50 units",
    "status": "success"
}
```

### Reset Command
To reset the model state for a new episode:
```json
{
    "action": "reset"
}
```

## Action Mapping

NaVid outputs discrete actions that are converted to velocity commands:

| Action | vx | yaw_rate | duration_s | Description |
|--------|-----|----------|------------|-------------|
| STOP | 0.0 | 0.0 | 0.0 | Stop navigation |
| FORWARD | 0.25 | 0.0 | 1.0 | Move forward 0.25m |
| LEFT | 0.0 | +0.52 | 1.0 | Turn left 30° |
| RIGHT | 0.0 | -0.52 | 1.0 | Turn right 30° |

## Troubleshooting

### Out of Memory Error
- Use a GPU with more VRAM (24GB+ recommended)
- Reduce image resolution
- Use `--device cpu` (very slow, not recommended)

### Import Errors (navid module not found)
Ensure PYTHONPATH includes NaVid project:
```bash
export PYTHONPATH=$PYTHONPATH:/path/to/NaVid-VLN-CE
```

### Connection Refused
- Verify server is running: `netstat -tlnp | grep 4321`
- Check if port is already in use
- Check firewall settings for remote access

## Citation

If you use NaVid in your research, please cite:

```bibtex
@inproceedings{zhang2024navid,
    title={NaVid: Video-based VLM Plans the Next Step for Vision-and-Language Navigation},
    author={Zhang, Jiazhao and Wang, Kunyu and Xu, Rongtao and others},
    booktitle={RSS},
    year={2024}
}
```

## License

The NaVid model and code are subject to their respective licenses. Please refer to the [NaVid-VLN-CE repository](https://github.com/jzhzhang/NaVid-VLN-CE) for license details.

