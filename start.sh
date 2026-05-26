#!/usr/bin/env bash
# Start Mars inference server on Ubuntu 4090
# Usage: ./start.sh [port]
set -e

PORT=${1:-8765}
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

export INFER_PORT=$PORT
export INFER_DEVICE=${INFER_DEVICE:-cuda:0}

# Optional: point to local model weights
# export YOLO_MODEL_PATH=/path/to/yolo26n-seg.pt
# export GDINO_MODEL_ID=IDEA-Research/grounding-dino-base
# export SAM2_MODEL_ID=facebook/sam2-hiera-small

cd "$SCRIPT_DIR"
exec python server.py
