#!/usr/bin/env bash
# Install F5-TTS into its own venv, isolated from the project's system Python.
# Rationale: f5-tts pins transformers/torch ranges; the project runs
# transformers 5.22.2 + torch 2.9.1+cu126 and must not be disturbed
# (same pattern as external/phonepure_venv).
set -u
cd "$(dirname "$0")" || exit 1
VENV="$PWD/f5tts_venv"
PY="$VENV/Scripts/python.exe"

echo "=== [1/4] create venv ==="
if [ ! -x "$PY" ]; then
  python -m venv f5tts_venv || { echo "FAIL: venv creation"; exit 1; }
fi
"$PY" -V || exit 1

echo "=== [2/4] upgrade pip ==="
"$PY" -m pip install --upgrade pip setuptools wheel 2>&1 | tail -3

echo "=== [3/4] torch cu126 ==="
"$PY" -m pip install torch==2.9.1 torchaudio==2.9.1 \
  --index-url https://download.pytorch.org/whl/cu126 2>&1 | tail -5

echo "=== [4/4] f5-tts ==="
"$PY" -m pip install f5-tts 2>&1 | tail -25

echo "=== versions ==="
"$PY" - <<'EOF'
import torch
print("torch", torch.__version__, "cuda", torch.cuda.is_available())
try:
    import f5_tts
    print("f5_tts", getattr(f5_tts, "__version__", "(no __version__)"), f5_tts.__file__)
except Exception as e:
    print("f5_tts IMPORT FAILED:", type(e).__name__, e)
EOF
echo "=== DONE ==="
