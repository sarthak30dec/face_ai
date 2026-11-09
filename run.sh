#!/usr/bin/env bash
# Start the face_ai test UI: ./run.sh            (opens http://localhost:8000)
#                            PORT=8080 ./run.sh   (other port)
#                            ./run.sh --no-open   (don't open a browser)
# First run creates .venv, installs pinned dependencies and downloads the default models.
set -euo pipefail
cd "$(dirname "$0")"

PORT="${PORT:-8000}"
PY=.venv/bin/python

if [ ! -x "$PY" ]; then
  echo "==> Creating Python environment (.venv)"
  if command -v uv >/dev/null 2>&1; then
    uv venv --python 3.11 .venv
    uv pip install --python "$PY" -e .
  elif command -v python3.11 >/dev/null 2>&1; then
    python3.11 -m venv .venv
    "$PY" -m pip install --upgrade pip
    "$PY" -m pip install -e .
  else
    echo "Python 3.11 is required (or install uv: https://docs.astral.sh/uv/)." >&2
    exit 1
  fi
fi

if ! "$PY" -c "import face_ai, cv2, mediapipe, onnxruntime" >/dev/null 2>&1; then
  echo "==> Installing dependencies"
  if command -v uv >/dev/null 2>&1; then uv pip install --python "$PY" -e .; else "$PY" -m pip install -e .; fi
fi

need=()
[ -f models/pretrained/yunet/face_detection_yunet_2023mar.onnx ] || need+=(yunet)
[ -f models/pretrained/sface/face_recognition_sface_2021dec.onnx ] || need+=(sface)
[ -f models/pretrained/buffalo_l/w600k_r50.onnx ] || need+=(buffalo_l)
[ -f models/pretrained/face_landmarker/face_landmarker.task ] || need+=(face_landmarker)
[ -f models/pretrained/tasks_vision/package/vision_bundle.mjs ] || need+=(tasks_vision)
if [ ${#need[@]} -gt 0 ]; then
  echo "==> Downloading models: ${need[*]}"
  "$PY" scripts/download_models.py --only "${need[@]}"
fi

start_port=$PORT
while lsof -nP -iTCP:"$PORT" -sTCP:LISTEN >/dev/null 2>&1; do
  PORT=$((PORT + 1))
  if [ "$PORT" -gt $((start_port + 20)) ]; then
    echo "No free port in $start_port-$((start_port + 20)). Try: PORT=9000 ./run.sh" >&2
    exit 1
  fi
done
[ "$PORT" != "$start_port" ] && echo "==> Port $start_port is busy, using $PORT"

echo "==> Starting face_ai UI on http://localhost:$PORT  (Ctrl+C to stop)"
exec "$PY" -m face_ai.api.web_ui --port "$PORT" --open "$@"
