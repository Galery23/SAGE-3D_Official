# NaVILA Server for SAGE-3D Benchmark

This directory contains the server script for running NaVILA (Vision-Language-Action) model inference as part of the SAGE-3D benchmark evaluation.

## Overview

NaVILA is a Vision-Language-Action model for robot navigation. There are two related repositories:

- **[NaVILA](https://github.com/AnjieCheng/NaVILA)** - The main NaVILA model implementation (training, model architecture)
- **[NaVILA-Bench](https://github.com/yang-zj1026/NaVILA-Bench)** - The benchmark evaluation framework in Isaac Lab

**This server script is designed to work with [NaVILA-Bench](https://github.com/yang-zj1026/NaVILA-Bench).**

## Prerequisites

- **Hardware**: NVIDIA GPU with at least 24GB VRAM (e.g., RTX 3090, RTX 4090, A100)
- **Software**: Ubuntu 20.04+, CUDA 11.8+, Python 3.10

## Installation

### Step 1: Clone and Setup NaVILA-Bench

```bash
# Clone the NaVILA-Bench repository (NOT the main NaVILA repo)
git clone https://github.com/yang-zj1026/NaVILA-Bench.git
cd NaVILA-Bench
```

### Step 2: Setup Environment

Follow the [NaVILA-Bench installation guide](https://github.com/yang-zj1026/NaVILA-Bench#installation):

```bash
# Create conda environment
conda create -n navila-eval python=3.10
conda activate navila-eval

# Install Isaac Sim (for Ubuntu 22.04+)
pip install isaacsim-rl==4.1.0 isaacsim-replicator==4.1.0 isaacsim-extscache-physics==4.1.0 \
    isaacsim-extscache-kit-sdk==4.1.0 isaacsim-extscache-kit==4.1.0 isaacsim-app==4.1.0 \
    --extra-index-url https://pypi.nvidia.com
```

### Step 3: Install NaVILA Model Dependencies

The NaVILA model requires additional dependencies from the [NaVILA repo](https://github.com/AnjieCheng/NaVILA):

```bash
# Clone NaVILA model repo for llava dependencies
git clone https://github.com/AnjieCheng/NaVILA.git
cd NaVILA

# Install dependencies
pip install flash-attn==2.5.8
pip install -e .
pip install -e ".[train]"
pip install -e ".[eval]"

# Install HuggingFace Transformers
pip install git+https://github.com/huggingface/transformers@v4.37.2

# Apply necessary patches
site_pkg_path=$(python -c 'import site; print(site.getsitepackages()[0])')
cp -rv ./llava/train/transformers_replace/* $site_pkg_path/transformers/
cp -rv ./llava/train/deepspeed_replace/* $site_pkg_path/deepspeed/

cd ..  # Return to NaVILA-Bench directory
```

### Step 4: Download Model Weights

Download the pre-trained NaVILA model checkpoint from HuggingFace:

```bash
# Option 1: Using huggingface-cli
huggingface-cli download a8cheng/navila-llama3-8b-8f --local-dir /path/to/navila-llama3-8b-8f

# Option 2: Using git lfs
git lfs install
git clone https://huggingface.co/a8cheng/navila-llama3-8b-8f /path/to/navila-llama3-8b-8f
```

### Step 5: Copy Server Script

Copy the `navila_server.py` script to the NaVILA-Bench project:

```bash
# Copy to NaVILA-Bench scripts folder
cp navila_server.py /path/to/NaVILA-Bench/scripts/
```

## Usage

### Starting the Server

```bash
# Navigate to NaVILA-Bench project directory
cd /path/to/NaVILA-Bench

# Activate the environment
conda activate navila-eval

# Start the server
python scripts/navila_server.py \
    --model_path /path/to/navila-llama3-8b-8f \
    --port 54321 \
    --device cuda
```

### Server Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--model_path` | (required) | Path to NaVILA model checkpoint directory |
| `--host` | `localhost` | Host address to bind the server |
| `--port` | `54321` | Port number to listen on |
| `--device` | `cuda` | Device to run on (`cuda` or `cpu`) |
| `--precision` | `W16A16` | Compute precision |
| `--conv_mode` | `llama_3` | Conversation template mode |
| `--num_video_frames` | `8` | Number of video frames to process |

### Example Commands

```bash
# Basic usage
python scripts/navila_server.py --model_path /home/user/Models/navila-llama3-8b-8f

# Custom port and host (for remote access)
python scripts/navila_server.py \
    --model_path /home/user/Models/navila-llama3-8b-8f \
    --host 0.0.0.0 \
    --port 8888

# Multi-GPU (automatic distribution)
python scripts/navila_server.py \
    --model_path /home/user/Models/navila-llama3-8b-8f \
    --port 54321
```

## Running SAGE-3D Benchmark with NaVILA

Once the server is running, you can start the SAGE-3D benchmark evaluation in a **separate terminal**:

```bash
# Terminal 2: Run SAGE-3D Benchmark
cd /path/to/SAGE-3D_Official

python Code/benchmark/environment_evaluation/run_benchmark.py \
    --scene_usd_path /path/to/scenes \
    --batch_test_dir /path/to/test/data \
    --output_root /path/to/output \
    --vlm-host localhost \
    --vlm-port 54321 \
    --model-type navila \
    --task-type vln
```

## Communication Protocol

The server uses TCP socket communication with the following protocol:

### Request Format
1. **Size Header**: 8 bytes (big-endian) indicating JSON payload size
2. **JSON Payload**:
```json
{
    "images": ["base64_encoded_image_1", "base64_encoded_image_2", ...],
    "query": "Navigate to the red chair in the living room."
}
```

### Response Format
1. **Size Header**: 8 bytes (big-endian) indicating JSON payload size
2. **JSON Payload**: Model output string (action description)

## Troubleshooting

### Out of Memory Error
If you encounter OOM errors, try:
- Using a GPU with more VRAM
- Reducing batch size
- Using `--device cpu` (very slow, not recommended)

### CUDA Version Mismatch
Ensure your CUDA version matches the requirements:
```bash
nvidia-smi  # Check CUDA version
```

### Connection Refused
- Verify the server is running: `netstat -tlnp | grep 54321`
- Check firewall settings if connecting remotely
- Ensure the port is not already in use

### Import Errors (llava module not found)
Make sure you have installed the NaVILA model dependencies:
```bash
cd /path/to/NaVILA
pip install -e .
```

## Related Repositories

- **NaVILA Model**: https://github.com/AnjieCheng/NaVILA
- **NaVILA-Bench**: https://github.com/yang-zj1026/NaVILA-Bench
- **Model Weights**: https://huggingface.co/a8cheng/navila-llama3-8b-8f

## Citation

If you use NaVILA in your research, please cite:

```bibtex
@inproceedings{cheng2025navila,
    title={NaVILA: Legged Robot Vision-Language-Action Model for Navigation},
    author={Cheng, An-Chieh and Ji, Yandong and Yang, Zhaojing and Gongye, Zaitian and Zou, Xueyan and Kautz, Jan and Bıyık, Erdem and Yin, Hongxu and Liu, Sifei and Wang, Xiaolong},
    booktitle={RSS},
    year={2025}
}
```

## License

This server script is part of the SAGE-3D project. The NaVILA model and NaVILA-Bench are subject to their respective licenses (Apache-2.0 for NaVILA, MIT for NaVILA-Bench).
