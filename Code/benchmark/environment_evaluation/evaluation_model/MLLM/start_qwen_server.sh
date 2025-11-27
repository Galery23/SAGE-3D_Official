#!/bin/bash
#
# Qwen2.5-VL Server Startup Script for SAGE-3D Benchmark
#
# Usage:
#   ./start_qwen_server.sh
#   ./start_qwen_server.sh --port 8888
#

# ==================== Configuration ====================
# !!! IMPORTANT: Modify these paths according to your setup !!!

# Path to Qwen2.5-VL model
# Download from: https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct
# Or use HuggingFace model ID directly: "Qwen/Qwen2.5-VL-7B-Instruct"
MODEL_PATH="/path/to/your/Qwen2.5-VL-7B-Instruct"

# Conda environment name
CONDA_ENV="your_env_name"

# Server settings
HOST="localhost"
PORT=7777
DEVICE="cuda"

# ==================== End Configuration ====================

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --port)
            PORT="$2"
            shift 2
            ;;
        --host)
            HOST="$2"
            shift 2
            ;;
        --device)
            DEVICE="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo "=============================================="
echo "Qwen2.5-VL Server for SAGE-3D Benchmark"
echo "=============================================="

# Check model path
echo "[CHECK] Verifying model path..."
if [ ! -d "$MODEL_PATH" ] && [[ "$MODEL_PATH" != *"/"* ]]; then
    echo "[INFO] Model path appears to be a HuggingFace model ID: $MODEL_PATH"
    echo "[INFO] Model will be downloaded automatically"
elif [ ! -d "$MODEL_PATH" ]; then
    echo "[ERROR] Model path does not exist: $MODEL_PATH"
    echo "Please download from: https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct"
    exit 1
else
    echo "[OK] Model path verified"
fi

# Activate conda environment
echo "[CHECK] Activating conda environment: $CONDA_ENV"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"
if [ $? -ne 0 ]; then
    echo "[ERROR] Failed to activate conda environment: $CONDA_ENV"
    exit 1
fi
echo "[OK] Environment activated"

# Check dependencies
echo "[CHECK] Checking dependencies..."
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import transformers; print(f'Transformers: {transformers.__version__}')"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

# Display configuration
echo ""
echo "Configuration:"
echo "  Model path: $MODEL_PATH"
echo "  Server: $HOST:$PORT"
echo "  Device: $DEVICE"
echo ""

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Start server
echo "[START] Starting Qwen2.5-VL server..."
python "$SCRIPT_DIR/mllm_server.py" \
    --model_type qwen-vl \
    --model_path "$MODEL_PATH" \
    --host "$HOST" \
    --port "$PORT" \
    --device "$DEVICE"

