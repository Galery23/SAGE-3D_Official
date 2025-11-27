# MLLM Server for SAGE-3D Benchmark

This directory contains the server script for running Multimodal Large Language Models (MLLMs) for VLN evaluation in the SAGE-3D benchmark.

## Overview

This server supports **any multimodal model** that can process images and text. The modular adapter design makes it easy to add new models. We provide built-in adapters for some popular models as examples:

| Model | Example | Description |
|-------|---------|-------------|
| Qwen2.5-VL | [Qwen/Qwen2.5-VL-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct) | Alibaba's vision-language model |
| LLaVA | [llava-hf/llava-1.5-7b-hf](https://huggingface.co/llava-hf/llava-1.5-7b-hf) | Open-source VLM |
| InternVL | [OpenGVLab/InternVL2-8B](https://huggingface.co/OpenGVLab/InternVL2-8B) | OpenGVLab's vision-language model |

**You can easily add support for any other multimodal model** by creating a new adapter class (see [Adding New Models](#adding-new-models) section).

## Prerequisites

- **Hardware**: NVIDIA GPU with at least 16GB VRAM (24GB+ recommended)
- **Software**: Python 3.8+, PyTorch 2.0+, Transformers 4.37+

## Installation

### Step 1: Create Environment

```bash
# Create conda environment
conda create -n mllm python=3.10
conda activate mllm

# Install PyTorch (adjust CUDA version as needed)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Install dependencies
pip install transformers accelerate pillow numpy
```

### Step 2: Download Model Weights

#### Option 1: Download to Local Path

```bash
# Qwen2.5-VL
huggingface-cli download Qwen/Qwen2.5-VL-7B-Instruct --local-dir /path/to/Qwen2.5-VL-7B-Instruct

# LLaVA
huggingface-cli download llava-hf/llava-1.5-7b-hf --local-dir /path/to/llava-1.5-7b-hf

# InternVL
huggingface-cli download OpenGVLab/InternVL2-8B --local-dir /path/to/InternVL2-8B
```

#### Option 2: Use HuggingFace Model ID Directly

You can also use the HuggingFace model ID directly (model will be downloaded automatically):

```bash
python mllm_server.py --model_type qwen-vl --model_path Qwen/Qwen2.5-VL-7B-Instruct --port 7777
```

## Usage

### Direct Python Command

```bash
# Activate environment
conda activate mllm

# Start Qwen2.5-VL server
python mllm_server.py \
    --model_type qwen-vl \
    --model_path /path/to/Qwen2.5-VL-7B-Instruct \
    --port 7777

# Start LLaVA server
python mllm_server.py \
    --model_type llava \
    --model_path /path/to/llava-1.5-7b-hf \
    --port 7778

# Start InternVL server
python mllm_server.py \
    --model_type internvl \
    --model_path /path/to/InternVL2-8B \
    --port 7779
```

### Server Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--model_type` | (required) | Model type: `qwen-vl`, `llava`, `internvl` |
| `--model_path` | (required) | Path to model or HuggingFace model ID |
| `--host` | `localhost` | Host address to bind |
| `--port` | `7777` | Port number to listen on |
| `--device` | `cuda` | Device: `cuda` or `cpu` |

## Running SAGE-3D Benchmark with MLLM

Once the server is running, start the SAGE-3D benchmark in a **separate terminal**:

```bash
# Terminal 2: Run SAGE-3D Benchmark
cd /path/to/SAGE-3D_Official

python Code/benchmark/environment_evaluation/run_benchmark.py \
    --scene_usd_path /path/to/scenes \
    --batch_test_dir /path/to/test/data \
    --output_root /path/to/output \
    --vlm-port 7777 \
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
    "images": ["base64_encoded_image_1", ...],
    "query": "Navigate to the red chair."
}
```

### Response Format
1. **Size Header**: 8 bytes (big-endian) indicating JSON payload size
2. **JSON Payload**:
```json
{
    "status": "success",
    "result": "MOVE_FORWARD",
    "inference_time": 0.5
}
```

## Action Space

The MLLM outputs one of four discrete actions:

| Action | Description |
|--------|-------------|
| `MOVE_FORWARD` | Move forward |
| `TURN_LEFT` | Turn left |
| `TURN_RIGHT` | Turn right |
| `STOP` | Stop navigation |

## Troubleshooting

### Out of Memory Error

- Use a GPU with more VRAM
- Try smaller model variants (e.g., 2B instead of 7B)
- Use `--device cpu` (very slow)

### Model Loading Errors

Ensure you have the correct transformers version:
```bash
pip install transformers>=4.37.0
```

### Connection Refused

- Verify server is running: `netstat -tlnp | grep 7777`
- Check firewall settings
- Ensure port is not in use

## Adding New Models

To add support for a new MLLM:

1. Create a new adapter class inheriting from `MLLMAdapter`
2. Implement `load_model()` and `generate_response()` methods
3. Add the model type to the argument parser
4. Update the model creation logic in `main()`

Example:
```python
class NewModelAdapter(MLLMAdapter):
    def load_model(self):
        # Load your model here
        pass
    
    def generate_response(self, image, instruction, **kwargs):
        # Run inference here
        return self.extract_action(output)
```

## License

This server script is part of the SAGE-3D project. The MLLM models are subject to their respective licenses.

