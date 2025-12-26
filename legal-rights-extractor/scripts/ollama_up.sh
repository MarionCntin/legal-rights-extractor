#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/ollama_env.sh"

echo "OLLAMA_MODELS=$OLLAMA_MODELS"
echo "Starting Ollama server on $OLLAMA_HOST ..."
# Foreground (recommended while developing)
ollama serve
