#!/usr/bin/env bash
set -euo pipefail

# Complex end-to-end test:
# 1) Ask K2DO to create code + generate Arabic-style WAV
# 2) Parse absolute WAV path from reply
# 3) Validate audio format (mono, 16-bit, 44.1kHz, ~8s)
#
# Usage:
#   tests/e2e_arabic_music.sh

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
K2DO_BIN="$ROOT_DIR/.venv/bin/k2do"
SESSION_ID="cli:e2e_arabic_music_$(date +%s)"
TMP_OUT="$(mktemp)"
trap 'rm -f "$TMP_OUT"' EXIT

PROMPT="Сделай e2e: создай python-скрипт в workspace, сгенерируй арабский WAV на 8 секунд, формат mono 16-bit 44100Hz, запусти и проверь файл. Верни только абсолютный путь к WAV."

if ! getent ahosts build-api.k2think.ai >/dev/null 2>&1; then
  echo "[e2e] FAIL: DNS can't resolve build-api.k2think.ai"
  exit 1
fi

echo "[e2e] Running K2DO complex prompt..."
if ! timeout 240 "$K2DO_BIN" agent \
  --no-markdown \
  --no-deepthink \
  --session "$SESSION_ID" \
  -m "$PROMPT" >"$TMP_OUT" 2>&1; then
  echo "[e2e] FAIL: k2do command timed out or exited with error"
  cat "$TMP_OUT"
  exit 1
fi

WAV_PATH="$(grep -Eo '/[^[:space:]]+\.wav' "$TMP_OUT" | tail -n1 || true)"
if [[ -z "$WAV_PATH" ]]; then
  echo "[e2e] FAIL: could not extract wav path from K2DO output"
  cat "$TMP_OUT"
  exit 1
fi

if [[ ! -f "$WAV_PATH" ]]; then
  echo "[e2e] FAIL: wav file not found at $WAV_PATH"
  exit 1
fi

echo "[e2e] Validating WAV format: $WAV_PATH"
python3 - "$WAV_PATH" <<'PY'
import sys
import wave

path = sys.argv[1]
with wave.open(path, "rb") as w:
    channels = w.getnchannels()
    sampwidth = w.getsampwidth()
    sr = w.getframerate()
    frames = w.getnframes()
    duration = frames / float(sr)

if channels != 1:
    raise SystemExit(f"[e2e] FAIL: channels={channels}, expected 1")
if sampwidth != 2:
    raise SystemExit(f"[e2e] FAIL: sampwidth={sampwidth}, expected 2")
if sr != 44100:
    raise SystemExit(f"[e2e] FAIL: sample_rate={sr}, expected 44100")
if not (7.5 <= duration <= 8.5):
    raise SystemExit(f"[e2e] FAIL: duration={duration:.3f}s, expected ~8s")

print(f"[e2e] PASS: channels={channels}, sampwidth={sampwidth}, sr={sr}, duration={duration:.3f}s")
PY

echo "[e2e] PASS: complex Arabic-music flow completed"
