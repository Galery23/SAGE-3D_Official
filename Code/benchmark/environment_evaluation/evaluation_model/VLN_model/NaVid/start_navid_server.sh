#!/bin/bash
#
# NaVid Server Startup Script for SAGE-3D Benchmark
#
# Usage:
#   ./start_navid_server.sh
#   ./start_navid_server.sh --port 8888
#

# ==================== Configuration ====================
# !!! IMPORTANT: Modify these paths according to your setup !!!

# Path to NaVid model checkpoint
# Download from: https://huggingface.co/jzhzhang/navid-7b-full-224-video-fps-1-grid-2-r2r-rxr-training-split
MODEL_PATH="/path/to/your/navid-7b-full-224-video-fps-1-grid-2-r2r-rxr-training-split"

# Path to EVA-ViT-G weights
# Download from: https://storage.googleapis.com/sfr-vision-language-research/LAVIS/models/BLIP2/eva_vit_g.pth
EVA_VIT_PATH="/path/to/your/eva_vit_g.pth"

# Path to NaVid-VLN-CE project
# Clone from: https://github.com/jzhzhang/NaVid-VLN-CE
NAVID_PROJECT_PATH="/path/to/your/NaVid-VLN-CE"

# Conda environment name
CONDA_ENV="NaVid"

# Server settings (can be overridden by command line args)
HOST="localhost"
PORT=4321
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
echo "NaVid Server for SAGE-3D Benchmark"
echo "=============================================="

# Check model weights
echo "[CHECK] Verifying model weights..."
if [ ! -d "$MODEL_PATH" ]; then
    echo "[ERROR] NaVid model path does not exist: $MODEL_PATH"
    echo "Please download from: https://huggingface.co/jzhzhang/navid-7b-full-224-video-fps-1-grid-2-r2r-rxr-training-split"
    exit 1
fi
if [ ! -f "$EVA_VIT_PATH" ]; then
    echo "[ERROR] EVA-ViT-G weights file does not exist: $EVA_VIT_PATH"
    echo "Please download from: https://storage.googleapis.com/sfr-vision-language-research/LAVIS/models/BLIP2/eva_vit_g.pth"
    exit 1
fi
echo "[OK] Model weights verified"

# Activate conda environment
echo "[CHECK] Activating conda environment: $CONDA_ENV"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"
if [ $? -ne 0 ]; then
    echo "[ERROR] Failed to activate conda environment: $CONDA_ENV"
    exit 1
fi
echo "[OK] Environment activated"

# Check GPU availability
echo "[CHECK] Checking GPU status..."
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU count: {torch.cuda.device_count()}')"

# Set PYTHONPATH
export PYTHONPATH="$PYTHONPATH:$NAVID_PROJECT_PATH"

# Change to NaVid project directory
cd "$NAVID_PROJECT_PATH"

# Display configuration
echo ""
echo "Configuration:"
echo "  Model path: $MODEL_PATH"
echo "  EVA-ViT path: $EVA_VIT_PATH"
echo "  Server: $HOST:$PORT"
echo "  Device: $DEVICE"
echo ""

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Start server
echo "[START] Starting NaVid server..."
python "$SCRIPT_DIR/navid_server.py" \
    --model_path "$MODEL_PATH" \
    --eva_vit_path "$EVA_VIT_PATH" \
    --host "$HOST" \
    --port "$PORT" \
    --device "$DEVICE"
