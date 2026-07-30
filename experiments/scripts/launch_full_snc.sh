#!/usr/bin/env bash
set -euo pipefail

source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"
WORKSPACE="$ROOT/workspace"
EXPERIMENT_NAME=${EXPERIMENT_NAME:-harness_g_snc_full_f1_3b_2wiki_b128_120_8g}
RUN_DIR="$ROOT/runs/$EXPERIMENT_NAME"

if [[ ! -f "$ROOT/reports/graph_validation.json" ]]; then
  echo "[ERROR] graph validation report is missing" >&2
  exit 2
fi
python3 - "$ROOT/reports/graph_validation.json" <<'PY'
import json, sys
report = json.load(open(sys.argv[1], encoding="utf-8"))
assert report.get("ok") is True, report
assert report.get("num_paragraphs") == 2811, report
PY
if pgrep -af "verl.trainer.main_ppo" >/dev/null; then
  echo "[ERROR] a main_ppo training process is still running; refusing launch" >&2
  pgrep -af "verl.trainer.main_ppo" >&2 || true
  exit 3
fi

mkdir -p "$RUN_DIR" "$ROOT/data/processed"
cd "$WORKSPACE"

setsid nohup env \
  EXPERIMENT_NAME="$EXPERIMENT_NAME" \
  RUN_DIR="$RUN_DIR" \
  PROCESSED_DIR="$ROOT/data/processed" \
  GRAPH_DIR="$ROOT/graph/harness_g_graph" \
  VAL_SPLIT=dev \
  API_PORT=8012 \
  CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  N_GPUS=8 \
  TP_SIZE=1 \
  ROLLOUT_N=8 \
  TRAIN_BATCH_SIZE=128 \
  PPO_MINI_BATCH_SIZE=32 \
  TOTAL_STEPS=120 \
  SAVE_FREQ=20 \
  TEST_FREQ=10 \
  DATA_SOURCE=2WikiMultiHopQA \
    BASE_MODEL=Qwen/Qwen2.5-3B-Instruct \
  MODEL_NAME=Qwen2.5-3B-Instruct \
  MAX_PROMPT_LENGTH=8192 \
  MAX_START_LENGTH=8192 \
  MAX_RESPONSE_LENGTH=2048 \
  MAX_TOOL_RESPONSE_LENGTH=4096 \
  PPO_MICRO_BATCH_PER_GPU=2 \
  LOGPROB_MICRO_BATCH_PER_GPU=4 \
  GPU_MEMORY_UTILIZATION=0.6 \
  RAY_health_check_timeout_ms=30000 \
  RAY_health_check_failure_threshold=10 \
  bash scripts/train_harness_g_8gpu.sh "$@" \
  > "$RUN_DIR/driver.log" 2>&1 < /dev/null &

pid=$!
printf '%s\n' "$pid" > "$RUN_DIR/driver.pid"
printf '%s\n' \
  "experiment=$EXPERIMENT_NAME" \
  "pid=$pid" \
  "run_dir=$RUN_DIR" \
  "graph_dir=$ROOT/graph/harness_g_graph"
