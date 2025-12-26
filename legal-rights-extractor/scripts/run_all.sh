#!/usr/bin/env bash
set -euo pipefail

source ./scripts/ollama_env.sh

# 1) ingest + chunk
lre ingest run --raw-dir data/raw --out-dir data/out

# 2) retrieval
lre retrieve --out-dir data/out --top-k 12

# 3) extraction for every doc
export LRE_VERBOSE=1

find data/out -name retrieval.jsonl -print0 | while IFS= read -r -d '' r; do
  doc_dir="$(dirname "$r")"
  company_id="$(basename "$(dirname "$doc_dir")")"
  out="$doc_dir/extraction.jsonl"

  echo "=== EXTRACT company_id=$company_id retrieval=$r out=$out"
  lre extract \
    --company-id "$company_id" \
    --retrieval "$r" \
    --out "$out" \
    --max-chars 3000 \
    --num-ctx 2048 \
    --timeout-s 900
done
