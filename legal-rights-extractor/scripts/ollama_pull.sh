#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/ollama_env.sh"

MODEL="${1:-qwen2.5:3b-instruct}"

echo "OLLAMA_MODELS=$OLLAMA_MODELS"
echo "Pulling model: $MODEL"
ollama pull "$MODEL"

echo "Done. Models stored inside: $OLLAMA_MODELS"
