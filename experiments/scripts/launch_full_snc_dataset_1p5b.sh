#!/usr/bin/env bash
set -euo pipefail

source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"
DATA_SOURCE=${1:?usage: launch_full_snc_dataset.sh DATA_SOURCE [trainer overrides...]}
shift
case "$DATA_SOURCE" in
  2WikiMultiHopQA) SLUG=2wiki ;;
  HotpotQA) SLUG=hotpotqa ;;
  Musique) SLUG=musique ;;
  NQ) SLUG=nq ;;
  PopQA) SLUG=popqa ;;
  TriviaQA) SLUG=triviaqa ;;
  *) echo "[ERROR] unsupported DATA_SOURCE=$DATA_SOURCE" >&2; exit 2 ;;
esac

WORKSPACE=$ROOT/workspace
EXPERIMENT_NAME=${EXPERIMENT_NAME:-harness_g_snc_full_f1_1p5b_${SLUG}_b128_120_8g}
RUN_DIR=$ROOT/runs/$EXPERIMENT_NAME
CHECKPOINT_DIR=$ROOT/checkpoints/Harness-G/$EXPERIMENT_NAME
RESULTS_DIR=$ROOT/expr_results/$EXPERIMENT_NAME
BASE_MODEL=${BASE_MODEL:-Qwen/Qwen2.5-1.5B-Instruct}
MODEL_NAME=${MODEL_NAME:-Qwen2.5-1.5B-Instruct}
if [[ "$DATA_SOURCE" == "2WikiMultiHopQA" ]]; then
  PROCESSED_DIR=$ROOT/data/processed
  GRAPH_DIR=$ROOT/graph/harness_g_graph
  GRAPH_REPORT=$ROOT/reports/graph_validation.json
  CORPUS_REPORT=
else
  PROCESSED_DIR=$ROOT/data/$DATA_SOURCE/processed
  GRAPH_DIR=$ROOT/graphs/$DATA_SOURCE/harness_g_graph
  GRAPH_REPORT=$ROOT/reports/$DATA_SOURCE/graph_validation.json
  CORPUS_REPORT=$ROOT/corpora/$DATA_SOURCE/corpus_manifest.json
fi
API_PORT=${API_PORT:-8012}
EXPECTED_GPU_COUNT=${EXPECTED_GPU_COUNT:-8}
MIN_GPU_FREE_MIB=${MIN_GPU_FREE_MIB:-70000}
PY=${PYTHON_BIN:-python}

"$PY" - "$GRAPH_REPORT" "$CORPUS_REPORT" "$DATA_SOURCE" "$GRAPH_DIR" "$PROCESSED_DIR" "$BASE_MODEL" <<'PY'
import json,sys
from pathlib import Path
graph_report,corpus_report,data_source,graph_dir,processed,base_model=sys.argv[1:]
g=json.load(open(graph_report))
assert g.get('ok') is True,g
assert Path(g.get('graph_dir','')).resolve()==Path(graph_dir).resolve(),g
if data_source=='2WikiMultiHopQA':
 assert int(g.get('num_paragraphs'))==2811,g
else:
 c=json.load(open(corpus_report))
 assert c.get('ok') is True,c
 assert g.get('data_source')==data_source and c.get('data_source')==data_source,(g,c)
 assert int(g.get('num_paragraphs'))==int(c.get('num_chunks')),(g,c)
 assert int(c.get('max_token_size'))==1200 and int(c.get('overlap_token_size'))==100,c
for split in ('train','dev','test'):
 p=Path(processed)/f'{split}.parquet'
 assert p.is_file() and p.stat().st_size>0,p
model=Path(base_model)
assert model.is_dir() and (model/'config.json').is_file(),model
cfg=json.load(open(model/'config.json'))
assert int(cfg.get('hidden_size',-1))==1536 and int(cfg.get('num_hidden_layers',-1))==28,cfg
PY

if pgrep -af 'python -m verl[.]trainer[.]main_ppo' >/dev/null; then
  echo "[ERROR] a main_ppo training process is active; refusing launch" >&2
  pgrep -af 'python -m verl[.]trainer[.]main_ppo' >&2 || true
  exit 3
fi
RECOVERY_MODE=${RECOVERY_MODE:-0}
if [[ "$RECOVERY_MODE" != "1" ]]; then
  for path in "$RUN_DIR" "$CHECKPOINT_DIR" "$RESULTS_DIR"; do
    if [[ -e "$path" || -L "$path" ]]; then
      echo "[ERROR] target exists; refusing overwrite: $path" >&2
      exit 4
    fi
  done
else
  # The audited recovery handler preserves checkpoints/evaluations and empties
  # per-attempt run files before calling this launcher.
  mkdir -p "$RUN_DIR"
  if find "$RUN_DIR" -mindepth 1 -maxdepth 1 -print -quit | grep -q .; then
    echo "[ERROR] recovery run directory is not empty: $RUN_DIR" >&2
    exit 4
  fi
fi
if ss -H -ltn "sport = :$API_PORT" 2>/dev/null | grep -q .; then
  echo "[ERROR] API port $API_PORT is occupied" >&2
  ss -ltnp "sport = :$API_PORT" >&2 || true
  exit 5
fi
read -r gpu_count min_free <<< "$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | awk 'NF{n++;v=$1+0;if(n==1||v<min)min=v}END{print n+0,(n?min:0)}')"
if (( gpu_count != EXPECTED_GPU_COUNT )); then
  echo "[ERROR] expected $EXPECTED_GPU_COUNT GPUs, got $gpu_count" >&2
  exit 6
fi
if (( min_free < MIN_GPU_FREE_MIB )); then
  echo "[ERROR] minimum GPU free memory ${min_free}MiB < ${MIN_GPU_FREE_MIB}MiB" >&2
  exit 7
fi

mkdir -p "$RUN_DIR"
cd "$WORKSPACE"
setsid nohup env \
  EXPERIMENT_NAME="$EXPERIMENT_NAME" \
  RUN_DIR="$RUN_DIR" \
  PROCESSED_DIR="$PROCESSED_DIR" \
  GRAPH_DIR="$GRAPH_DIR" \
  VAL_SPLIT=dev \
  API_PORT="$API_PORT" \
  CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  N_GPUS=8 \
  TP_SIZE=1 \
  ROLLOUT_N=8 \
  TRAIN_BATCH_SIZE=128 \
  PPO_MINI_BATCH_SIZE=32 \
  TOTAL_STEPS=120 \
  SAVE_FREQ=20 \
  TEST_FREQ=10 \
  DATA_SOURCE="$DATA_SOURCE" \
    BASE_MODEL="$BASE_MODEL" \
  MODEL_NAME="$MODEL_NAME" \
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
  "data_source=$DATA_SOURCE" \
  "pid=$pid" \
  "run_dir=$RUN_DIR" \
  "processed_dir=$PROCESSED_DIR" \
  "graph_dir=$GRAPH_DIR" \
  "base_model=$BASE_MODEL" \
  "model_name=$MODEL_NAME" \
  "adv_estimator=grpo_snc" \
  "chunk_size=1200" \
  "chunk_overlap=100"
