#!/usr/bin/env bash
set -euo pipefail

ENV_NAME=${ENV_NAME:-s3}
CONDA_BIN=${CONDA_EXE:-}
if [[ -z "$CONDA_BIN" || ! -x "$CONDA_BIN" ]]; then
  CONDA_BIN=$(command -v conda || true)
fi
if [[ -z "$CONDA_BIN" ]]; then
  echo "[ERROR] conda executable not found. Activate conda or set CONDA_EXE." >&2
  exit 127
fi
PYTHON=("$CONDA_BIN" run -n "$ENV_NAME" python)
DATA_SOURCE=${DATA_SOURCE:-2WikiMultiHopQA}
BASE_MODEL=${BASE_MODEL:-Qwen/Qwen2.5-3B-Instruct}
MODEL_NAME=${MODEL_NAME:-Qwen2.5-3B-Instruct}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-harness_g_${DATA_SOURCE}}
GRAPH_DIR=${GRAPH_DIR:-expr/${DATA_SOURCE}/harness_g_graph}
TOTAL_STEPS=${TOTAL_STEPS:-}
TRAIN_LIMIT=${TRAIN_LIMIT:--1}
VAL_LIMIT=${VAL_LIMIT:--1}
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
N_GPUS=${N_GPUS:-8}
TP_SIZE=${TP_SIZE:-1}
ROLLOUT_N=${ROLLOUT_N:-8}
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-128}
PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-32}
MAX_TURNS=${MAX_TURNS:-6}
DRY_RUN=${DRY_RUN:-false}
RUN_DIR=${RUN_DIR:-runs/${EXPERIMENT_NAME}}
API_PORT=${API_PORT:-8001}
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-2048}
MAX_TOOL_RESPONSE_LENGTH=${MAX_TOOL_RESPONSE_LENGTH:-4096}
MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-8192}
MAX_START_LENGTH=${MAX_START_LENGTH:-8192}
SAVE_FREQ=${SAVE_FREQ:-20}
TEST_FREQ=${TEST_FREQ:-10}
REMOVE_PREVIOUS_CKPT=${REMOVE_PREVIOUS_CKPT:-True}
VAL_SPLIT=${VAL_SPLIT:-dev}
if [[ "$VAL_SPLIT" != "dev" && "$VAL_SPLIT" != "test" ]]; then
  echo "[ERROR] VAL_SPLIT must be 'dev' or 'test' (got '$VAL_SPLIT')." >&2
  exit 2
fi
ACTOR_LR=${ACTOR_LR:-5e-7}
ACTOR_KL_LOSS_COEF=${ACTOR_KL_LOSS_COEF:-0.001}
ALGORITHM_KL_COEF=${ALGORITHM_KL_COEF:-0.001}
ACTOR_CLIP_RATIO=${ACTOR_CLIP_RATIO:-0.2}
ACTOR_GRAD_CLIP=${ACTOR_GRAD_CLIP:-1.0}
PYTEST_TARGETS=${PYTEST_TARGETS:-tests/}

effective_split_rows() {
  local filename="$1"
  local limit="$2"
  "${PYTHON[@]}" -c '
import json
import sys
from pathlib import Path

data_source, filename, limit_raw = sys.argv[1], sys.argv[2], sys.argv[3]
limit = int(limit_raw)
path = Path("datasets") / data_source / "raw" / filename
if path.exists():
    total = len(json.load(path.open("r", encoding="utf-8")))
else:
    total = max(limit, 0)
if limit < 0 or limit >= total:
    print(total)
else:
    print(limit)
' "$DATA_SOURCE" "$filename" "$limit"
}

EFFECTIVE_TRAIN_ROWS=$(effective_split_rows qa_train.json "$TRAIN_LIMIT")
if [[ "$EFFECTIVE_TRAIN_ROWS" -lt 1 ]]; then
  EFFECTIVE_TRAIN_ROWS=$TRAIN_BATCH_SIZE
fi
STEPS_PER_EPOCH=$(( (EFFECTIVE_TRAIN_ROWS + TRAIN_BATCH_SIZE - 1) / TRAIN_BATCH_SIZE ))
if [[ "$STEPS_PER_EPOCH" -lt 1 ]]; then
  STEPS_PER_EPOCH=1
fi
if [[ -z "${TOTAL_EPOCHS:-}" ]]; then
  if [[ -n "$TOTAL_STEPS" ]]; then
    TOTAL_EPOCHS=$(( (TOTAL_STEPS + STEPS_PER_EPOCH - 1) / STEPS_PER_EPOCH ))
  else
    TOTAL_EPOCHS=1
  fi
fi
if [[ "$TOTAL_EPOCHS" -lt 1 ]]; then
  TOTAL_EPOCHS=1
fi
if [[ -z "$TOTAL_STEPS" ]]; then
  TOTAL_STEPS=$(( STEPS_PER_EPOCH * TOTAL_EPOCHS ))
fi

USE_SPACY=${USE_SPACY:-true}
SPACY_MODEL=${SPACY_MODEL:-en_core_web_sm}
SPACY_BATCH_SIZE=${SPACY_BATCH_SIZE:-1024}
SPACY_N_PROCESS=${SPACY_N_PROCESS:-4}
SPACY_GPU=${SPACY_GPU:-false}
BUILD_EMBEDDINGS=${BUILD_EMBEDDINGS:-true}
EMBEDDING_BACKEND=${EMBEDDING_BACKEND:-bge}
EMBEDDING_MODEL_PATH=${EMBEDDING_MODEL_PATH:-BAAI/bge-large-en-v1.5}
EMBEDDING_BATCH_SIZE=${EMBEDDING_BATCH_SIZE:-64}
EMBEDDING_DEVICE=${EMBEDDING_DEVICE:-cuda}

mkdir -p "$RUN_DIR"

export CUDA_VISIBLE_DEVICES
export HARNESS_G_RUN_DIR="$RUN_DIR"
export HARNESS_G_REWARD_METRICS_PATH="$RUN_DIR/reward_metrics.jsonl"
export HARNESS_G_NAV_EVENTS_PATH="$RUN_DIR/nav_events.jsonl"
export HARNESS_G_GROUPS_PATH="$RUN_DIR/groups.jsonl"
export HARNESS_G_RUN_ID="${EXPERIMENT_NAME}"
export HARNESS_G_API_URL="http://localhost:${API_PORT}/harness_g_step"
export HARNESS_G_GRAPH_DIR="$GRAPH_DIR"
export HARNESS_G_DATA_SOURCE="$DATA_SOURCE"
export HARNESS_G_EMBEDDING_MODEL_PATH="$EMBEDDING_MODEL_PATH"
export HARNESS_G_EMBEDDING_DEVICE="$EMBEDDING_DEVICE"
export HARNESS_G_API_TIMEOUT=${HARNESS_G_API_TIMEOUT:-300}
export NO_PROXY=${NO_PROXY:-localhost,127.0.0.1}
export no_proxy=${no_proxy:-localhost,127.0.0.1}
export VLLM_ATTENTION_BACKEND=${VLLM_ATTENTION_BACKEND:-XFORMERS}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
export HYDRA_FULL_ERROR=1

echo "[Harness-G] using python: ${PYTHON[*]}"
"${PYTHON[@]}" --version
echo "[Harness-G] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}; N_GPUS=${N_GPUS}; TP_SIZE=${TP_SIZE}; rollout_n=${ROLLOUT_N}"
echo "[Harness-G] target steps: ${TOTAL_STEPS}; train_limit: ${TRAIN_LIMIT}; batch_size: ${TRAIN_BATCH_SIZE}; epochs: ${TOTAL_EPOCHS}"

echo "[Harness-G] running unit tests"
"${PYTHON[@]}" -m pytest $PYTEST_TARGETS -q
: > "$HARNESS_G_REWARD_METRICS_PATH"
: > "$HARNESS_G_NAV_EVENTS_PATH" 2>/dev/null || true
: > "$HARNESS_G_GROUPS_PATH" 2>/dev/null || true
: > "$RUN_DIR/snc_struct_diag.jsonl" 2>/dev/null || true
: > "$RUN_DIR/snc_step_diag.jsonl" 2>/dev/null || true

CMD=("${PYTHON[@]}" -m verl.trainer.main_ppo
  algorithm.adv_estimator=grpo_snc
  data.train_files="${PROCESSED_DIR:-datasets/${DATA_SOURCE}/processed}/train.parquet"
  data.val_files="${PROCESSED_DIR:-datasets/${DATA_SOURCE}/processed}/${VAL_SPLIT}.parquet"
  data.train_batch_size="$TRAIN_BATCH_SIZE"
  data.max_prompt_length="$MAX_PROMPT_LENGTH"
  data.max_response_length="$MAX_RESPONSE_LENGTH"
  data.max_start_length="$MAX_START_LENGTH"
  data.max_tool_response_length="$MAX_TOOL_RESPONSE_LENGTH"
  actor_rollout_ref.model.path="$BASE_MODEL"
  actor_rollout_ref.actor.optim.lr="$ACTOR_LR"
  actor_rollout_ref.actor.grad_clip="$ACTOR_GRAD_CLIP"
  actor_rollout_ref.actor.clip_ratio="$ACTOR_CLIP_RATIO"
  actor_rollout_ref.model.use_remove_padding=True
  actor_rollout_ref.actor.ppo_mini_batch_size="$PPO_MINI_BATCH_SIZE"
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu="${PPO_MICRO_BATCH_PER_GPU:-2}"
  actor_rollout_ref.actor.use_kl_loss=True
  actor_rollout_ref.actor.kl_loss_coef="$ACTOR_KL_LOSS_COEF"
  actor_rollout_ref.actor.kl_loss_type=low_var_kl
  actor_rollout_ref.model.enable_gradient_checkpointing=True
  actor_rollout_ref.actor.fsdp_config.param_offload=False
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=False
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu="${LOGPROB_MICRO_BATCH_PER_GPU:-4}"
  actor_rollout_ref.rollout.tensor_model_parallel_size="$TP_SIZE"
  actor_rollout_ref.rollout.name=vllm
  actor_rollout_ref.rollout.gpu_memory_utilization="${GPU_MEMORY_UTILIZATION:-0.6}"
  actor_rollout_ref.rollout.n=1
  actor_rollout_ref.rollout.n_repeat="$ROLLOUT_N"
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu="${LOGPROB_MICRO_BATCH_PER_GPU:-4}"
  actor_rollout_ref.ref.fsdp_config.param_offload=True
  algorithm.kl_ctrl.kl_coef="$ALGORITHM_KL_COEF"
  trainer.critic_warmup=0
  "trainer.logger=['console']"
  trainer.project_name=Harness-G
  trainer.experiment_name="$EXPERIMENT_NAME"
  trainer.n_gpus_per_node="$N_GPUS"
  trainer.nnodes=1
  trainer.save_freq="${SAVE_FREQ}"
  trainer.test_freq="${TEST_FREQ}"
  trainer.total_epochs="$TOTAL_EPOCHS"
  trainer.total_training_steps="$TOTAL_STEPS"
  trainer.val_before_train=False
  trainer.remove_previous_ckpt_in_save="$REMOVE_PREVIOUS_CKPT"
  tool.env=harness_g
  tool.max_turns="$MAX_TURNS"
  "$@")

{
  printf 'ENV_NAME=%q ' "$ENV_NAME"
  while IFS='=' read -r flag_name flag_value; do
    printf '%s=%q ' "$flag_name" "$flag_value"
  done < <(env | grep '^HARNESS_G_' | sort)
  printf '%q ' "${CMD[@]}"
  echo
} > "$RUN_DIR/train_command.txt"
env | grep '^HARNESS_G_' | sort > "$RUN_DIR/env_flags.txt"
cat > "$RUN_DIR/run_config.json" <<JSON
{
  "data_source": "${DATA_SOURCE}",
  "base_model": "${BASE_MODEL}",
  "model_name": "${MODEL_NAME}",
  "env_name": "${ENV_NAME}",
  "experiment_name": "${EXPERIMENT_NAME}",
  "graph_dir": "${GRAPH_DIR}",
  "api_port": ${API_PORT},
  "total_steps": ${TOTAL_STEPS},
  "train_limit": ${TRAIN_LIMIT},
  "val_limit": ${VAL_LIMIT},
  "validation_split": "${VAL_SPLIT}",
  "save_freq": ${SAVE_FREQ},
  "test_freq": ${TEST_FREQ},
  "cuda_visible_devices": "${CUDA_VISIBLE_DEVICES}",
  "n_gpus": ${N_GPUS},
  "tp_size": ${TP_SIZE},
  "rollout_n": ${ROLLOUT_N},
  "train_batch_size": ${TRAIN_BATCH_SIZE},
  "ppo_mini_batch_size": ${PPO_MINI_BATCH_SIZE},
  "max_turns": ${MAX_TURNS},
  "max_prompt_length": ${MAX_PROMPT_LENGTH},
  "max_start_length": ${MAX_START_LENGTH},
  "max_response_length": ${MAX_RESPONSE_LENGTH},
  "max_tool_response_length": ${MAX_TOOL_RESPONSE_LENGTH},
  "actor_lr": "${ACTOR_LR}",
  "actor_kl_loss_coef": "${ACTOR_KL_LOSS_COEF}",
  "algorithm_kl_coef": "${ALGORITHM_KL_COEF}",
  "actor_clip_ratio": "${ACTOR_CLIP_RATIO}",
  "actor_grad_clip": "${ACTOR_GRAD_CLIP}",
  "remove_previous_ckpt": "${REMOVE_PREVIOUS_CKPT}",
  "adv_estimator": "grpo_snc",
  "reward": "f1_outcome",
  "dry_run": "${DRY_RUN}"
}
JSON
git status --short > "$RUN_DIR/git_status.txt" || true
git diff --stat > "$RUN_DIR/git_diff_stat.txt" || true

if [[ "$DRY_RUN" == "true" ]]; then
  echo "[Harness-G] DRY_RUN=true; GRPO command:"
  cat "$RUN_DIR/train_command.txt"
  exit 0
fi

echo "[Harness-G] processing splits into datasets/${DATA_SOURCE}/processed"
PROCESSED_DIR="${PROCESSED_DIR:-datasets/${DATA_SOURCE}/processed}"
"${PYTHON[@]}" script_process_harness_g.py \
  --data_source "$DATA_SOURCE" \
  --train_limit "$TRAIN_LIMIT" \
  --val_limit "$VAL_LIMIT" \
  --output_dir "$PROCESSED_DIR"
for REPORT in "${PROCESSED_DIR}/harness_g_process_report.json" \
              "${PROCESSED_DIR}/harness_g_process_report.json"; do
  if [[ -f "$REPORT" ]]; then
    cp "$REPORT" "$RUN_DIR/$(basename "$REPORT")"
  fi
done

NEED_BUILD=false
if [[ ! -f "$GRAPH_DIR/graph_manifest.json" ]]; then
  NEED_BUILD=true
elif [[ "$USE_SPACY" == "true" ]]; then
  CURRENT_EXTRACTOR=$("${PYTHON[@]}" -c "import json; print(json.load(open('$GRAPH_DIR/graph_manifest.json')).get('entity_extractor',''))" 2>/dev/null || true)
  if [[ "$CURRENT_EXTRACTOR" != spacy:* ]]; then
    echo "[Harness-G] existing graph extractor is '$CURRENT_EXTRACTOR'; rebuilding with spaCy."
    NEED_BUILD=true
  fi
fi
if [[ "$NEED_BUILD" == "false" && "$BUILD_EMBEDDINGS" == "true" ]]; then
  CURRENT_EMBEDDING=$("${PYTHON[@]}" -c "import json; m=json.load(open('$GRAPH_DIR/graph_manifest.json')); print(m.get('embedding_backend','') + '|' + str(m.get('build_embeddings', False)) + '|' + str(m.get('embedding_model_path','')))" 2>/dev/null || true)
  EXPECTED_EMBEDDING="bge_transformers|True|${EMBEDDING_MODEL_PATH}"
  if [[ "$CURRENT_EMBEDDING" != "$EXPECTED_EMBEDDING" ]]; then
    echo "[Harness-G] existing graph embedding config is '$CURRENT_EMBEDDING'; rebuilding with BGE."
    NEED_BUILD=true
  fi
fi
if [[ "$NEED_BUILD" == "false" ]]; then
  CURRENT_EDGE_SCHEMA=$("${PYTHON[@]}" -c "import json; m=json.load(open('$GRAPH_DIR/graph_manifest.json')); print(str(m.get('graph_directed', True)) + '|' + str(m.get('sentence_adjacency_edges', False)) + '|' + str(m.get('entity_synonym_edges', False)) + '|' + str(m.get('entity_synonym_threshold', '')))" 2>/dev/null || true)
  if [[ "$CURRENT_EDGE_SCHEMA" != "False|True|True|0.8" && "$CURRENT_EDGE_SCHEMA" != "False|True|True|0.80" ]]; then
    echo "[Harness-G] existing graph edge schema is '$CURRENT_EDGE_SCHEMA'; rebuilding sentence/entity synonym edges."
    NEED_BUILD=true
  fi
fi

if [[ "$NEED_BUILD" == "true" ]]; then
  echo "[Harness-G] building graph at $GRAPH_DIR"
  BUILD_GRAPH_CMD=("${PYTHON[@]}" scripts/build_harness_g_graph.py \
    --data_source "$DATA_SOURCE" \
    --output_dir "$GRAPH_DIR" \
    --use_spacy "$USE_SPACY" \
    --spacy_model "$SPACY_MODEL" \
    --spacy_batch_size "$SPACY_BATCH_SIZE" \
    --spacy_n_process "$SPACY_N_PROCESS" \
    --spacy_gpu "$SPACY_GPU" \
    --build_embeddings "$BUILD_EMBEDDINGS" \
    --embedding_backend "$EMBEDDING_BACKEND" \
    --embedding_model_path "$EMBEDDING_MODEL_PATH" \
    --embedding_batch_size "$EMBEDDING_BATCH_SIZE" \
    --embedding_device "$EMBEDDING_DEVICE" \
    --build_sentence_edges true \
    --build_entity_synonyms true \
    --entity_synonym_topk "${ENTITY_SYNONYM_TOPK:-5}" \
    --entity_synonym_threshold "${ENTITY_SYNONYM_THRESHOLD:-0.80}" \
    --entity_synonym_candidate_limit "${ENTITY_SYNONYM_CANDIDATE_LIMIT:-256}" \
    --reuse_embeddings true)
  if [[ -n "${GRAPH_MAX_DOCS:-}" ]]; then
    BUILD_GRAPH_CMD+=(--max_docs "$GRAPH_MAX_DOCS")
  fi
  "${BUILD_GRAPH_CMD[@]}"
fi

"${PYTHON[@]}" scripts/validate_harness_g_graph.py --graph_dir "$GRAPH_DIR"
"${PYTHON[@]}" scripts/diagnose_harness_g_reachability.py \
  --data_source "$DATA_SOURCE" \
  --graph_dir "$GRAPH_DIR" \
  --sample_size "${REACHABILITY_SAMPLE_SIZE:-20}"
if [[ -f "$GRAPH_DIR/reachability_report.json" ]]; then
  cp "$GRAPH_DIR/reachability_report.json" "$RUN_DIR/reachability_report.json"
fi

if [[ -f "$GRAPH_DIR/reachability_report.json" ]]; then
REACH_ZERO=$("${PYTHON[@]}" -c '
import json
import sys

r = json.load(open(sys.argv[1], "r", encoding="utf-8"))
print(int(
    r.get("init_visible_contains_answer_rate", 0) == 0
    and r.get("context_visible_contains_answer_rate", 0) == 0
    and r.get("bridge_visible_contains_answer_rate", 0) == 0
    and r.get("expanded_visible_contains_answer_rate", 0) == 0
))
' "$GRAPH_DIR/reachability_report.json")
  if [[ "$REACH_ZERO" == "1" && "${HARNESS_G_FORCE_TRAIN_WITH_LOW_REACHABILITY:-false}" != "true" ]]; then
    echo "[ERROR] reachability is zero. Set HARNESS_G_FORCE_TRAIN_WITH_LOW_REACHABILITY=true to override."
    exit 1
  fi
fi

echo "[Harness-G] starting API on port $API_PORT"
setsid env CUDA_VISIBLE_DEVICES="${API_CUDA_DEVICES:-$CUDA_VISIBLE_DEVICES}" "${PYTHON[@]}" -u scripts/run_harness_g_api.py \
  --data_source "$DATA_SOURCE" \
  --graph_dir "$GRAPH_DIR" \
  --port "$API_PORT" \
  --max_turns "$MAX_TURNS" \
  > "$RUN_DIR/api.log" 2>&1 &
API_PID=$!
cleanup() {
  if [[ -n "${API_PID:-}" ]] && kill -0 "$API_PID" >/dev/null 2>&1; then
    kill -- "-$API_PID" >/dev/null 2>&1 || kill "$API_PID" >/dev/null 2>&1 || true
    wait "$API_PID" >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT

API_READY=false
for _ in $(seq 1 30); do
  if ! kill -0 "$API_PID" >/dev/null 2>&1; then
    break
  fi
  if "${PYTHON[@]}" -c '
import requests
import sys

port, data_source, graph_dir = sys.argv[1:4]
health = requests.get(f"http://localhost:{port}/health", timeout=2).json()
assert health["status"] == "ok", health
assert health.get("data_source") == data_source, health
assert health.get("graph_dir") == graph_dir, health
' "$API_PORT" "$DATA_SOURCE" "$GRAPH_DIR" >/dev/null 2>&1
  then
    API_READY=true
    break
  fi
  sleep 1
done

if [[ "$API_READY" != "true" ]]; then
  echo "[ERROR] Harness-G API failed to start for ${DATA_SOURCE} on port ${API_PORT}."
  echo "[ERROR] API log:"
  tail -n 80 "$RUN_DIR/api.log" || true
  exit 1
fi

"${PYTHON[@]}" -c '
import json
import requests
import sys

port, data_source, graph_dir = sys.argv[1:4]
health = requests.get(f"http://localhost:{port}/health", timeout=5).json()
assert health["status"] == "ok", health
assert health.get("data_source") == data_source, health
assert health.get("graph_dir") == graph_dir, health
print("[Harness-G] health:", json.dumps(health, ensure_ascii=False))
' "$API_PORT" "$DATA_SOURCE" "$GRAPH_DIR"

echo "[Harness-G] launching GRPO training"
"${CMD[@]}" 2>&1 | tee "$RUN_DIR/train.log"
