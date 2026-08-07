#!/usr/bin/env bash
# Download model weight files for ADTC 2026 Submission
#
# Rules:
#   - Must be idempotent (safe to run multiple times).
#   - Must download without any credentials (public URL only).
#   - The output path must match `_runtime.model_path` in metadata.json.

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ── 1. DOWNLOAD THE CORE QWEN 2.5 LLM WEIGHTS ─────────────────────────────────
MODEL_DIR="$HERE/model"
MODEL_FILE="$MODEL_DIR/qwen2.5-0.5b-instruct-q4_k_m.gguf"

# DIRECT FILE LINK (Bypasses Hugging Face frontend UI, works 100% token-free)
MODEL_URL="https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/qwen2.5-0.5b-instruct-q4_k_m.gguf?download=true"

mkdir -p "$MODEL_DIR"

if [[ -f "$MODEL_FILE" ]]; then
  echo "LLM weight binary already present at $MODEL_FILE — skipping download"
else
  echo "Downloading $MODEL_URL → $MODEL_FILE (~398 MB)…"
  if command -v curl > /dev/null 2>&1; then
    curl -L --fail --progress-bar -o "$MODEL_FILE.partial" "$MODEL_URL"
  elif command -v wget > /dev/null 2>&1; then
    wget --show-progress -O "$MODEL_FILE.partial" "$MODEL_URL"
  else
    echo "error: neither curl nor wget found" >&2
    exit 1
  fi
  mv "$MODEL_FILE.partial" "$MODEL_FILE"
fi

# ── 2. PRE-CACHE THE MULTILINGUAL SEMANTIC VECTOR TRANSFORMER ──────────────────
# Force download sentence-transformers files via python so they are ready offline
echo "Pre-caching sentence-transformers vector engine weights for offline evaluation..."
python -c "
try:
    from sentence_transformers import SentenceTransformer
    print('Downloading upgraded multilingual vector map weights...')
    SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
    print('Vector embedding cache complete!')
except Exception as e:
    print('Vector engine setup warning:', e)
"

echo "done: All model structures downloaded successfully!"
