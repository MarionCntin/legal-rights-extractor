# shellcheck shell=sh

# Resolve project root reliably (works in bash + zsh)
PROJECT_ROOT="$(cd "$(dirname "${(%):-%x}" 2>/dev/null || dirname "$0")/.." && pwd)"

# Project-local Ollama model store
export OLLAMA_MODELS="$PROJECT_ROOT/.ollama/models"

# Explicit host (safe default)
export OLLAMA_HOST="127.0.0.1:11434"
