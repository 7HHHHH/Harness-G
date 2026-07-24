#!/usr/bin/env bash
set -euo pipefail

source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"
DATA_SOURCE=${1:?usage: build_dataset_chunk1200_graph.sh DATA_SOURCE}
case "$DATA_SOURCE" in
  HotpotQA|Musique|NQ|PopQA|TriviaQA) ;;
  *) echo "[ERROR] unsupported DATA_SOURCE=$DATA_SOURCE" >&2; exit 2 ;;
esac

WORKSPACE=$ROOT/workspace
CODE_ROOT=${CODE_ROOT:-$REPO_ROOT}
CONDA_BIN=${CONDA_BIN:-conda}
PYTHON=("$CONDA_BIN" run -n s3 python)
CORPUS_DIR=$ROOT/corpora_chunk1200/$DATA_SOURCE
CORPUS_INPUT=$CORPUS_DIR/graph_input.jsonl
GRAPH_PARENT=$ROOT/graphs_chunk1200/$DATA_SOURCE
FINAL_GRAPH=$GRAPH_PARENT/harness_g_graph
BUILDING_GRAPH=$GRAPH_PARENT/harness_g_graph.building
REPORT_DIR=$ROOT/reports/chunk1200/$DATA_SOURCE
REPORT=$REPORT_DIR/graph_validation.json
BUILDING_REPORT=$REPORT_DIR/graph_validation.building.json
LOG=$ROOT/logs/build_chunk1200_graph_${DATA_SOURCE}.log
RESOURCE_WAIT_SECONDS=${RESOURCE_WAIT_SECONDS:-1200}
MIN_GPU_FREE_MIB=${MIN_GPU_FREE_MIB:-70000}

if [[ ! -s "$CORPUS_DIR/corpus_manifest.json" || ! -s "$CORPUS_INPUT" ]]; then
  echo "[ERROR] prepared chunk1200 corpus missing for $DATA_SOURCE" >&2
  exit 3
fi
"${PYTHON[@]}" - "$CORPUS_DIR/corpus_manifest.json" "$DATA_SOURCE" <<'PY'
import json,sys
m=json.load(open(sys.argv[1])); d=sys.argv[2]
assert m.get('ok') is True,m
assert m.get('data_source')==d,m
assert int(m.get('max_token_size'))==1200,m
assert int(m.get('overlap_token_size'))==100,m
assert int(m.get('max_slice_tokens'))<=1200,m
assert int(m.get('loader_records'))==int(m.get('num_chunks')),m
assert int(m.get('loader_structured_titles'))==0,m
PY

if pgrep -af 'python -m verl[.]trainer[.]main_ppo' >/dev/null; then
  echo "[ERROR] main_ppo is active; refusing graph build" >&2
  pgrep -af 'python -m verl[.]trainer[.]main_ppo' >&2 || true
  exit 4
fi
if [[ -d "$FINAL_GRAPH" ]]; then
  echo "[chunk1200] validating existing graph: $FINAL_GRAPH"
  "${PYTHON[@]}" "$SCRIPT_DIR/validate_dataset_chunk1200_graph.py" \
    --root "$ROOT" --data_source "$DATA_SOURCE" --graph_dir "$FINAL_GRAPH" \
    --report_path "$REPORT"
  exit 0
fi
if [[ -e "$BUILDING_GRAPH" ]]; then
  echo "[ERROR] partial graph exists; refusing destructive cleanup: $BUILDING_GRAPH" >&2
  exit 5
fi

mkdir -p "$GRAPH_PARENT" "$REPORT_DIR" "$ROOT/logs"
deadline=$(( $(date +%s) + RESOURCE_WAIT_SECONDS ))
while true; do
  read -r count min_free <<< "$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | awk 'NF{n++;v=$1+0;if(n==1||v<min)min=v}END{print n+0,(n?min:0)}')"
  if (( count != 8 )); then echo "[ERROR] expected 8 GPUs, found $count" >&2; exit 6; fi
  if (( min_free >= MIN_GPU_FREE_MIB )); then break; fi
  if (( $(date +%s) >= deadline )); then
    echo "[ERROR] GPUs not released: minimum_free=${min_free}MiB" >&2
    exit 7
  fi
  echo "[chunk1200] waiting for GPU release: minimum_free=${min_free}MiB"
  sleep 30
done

cd "$WORKSPACE"
chunks=$("${PYTHON[@]}" -c 'import json,sys; print(json.load(open(sys.argv[1], encoding="utf-8"))["num_chunks"])' "$CORPUS_DIR/corpus_manifest.json")
echo "[$(date '+%F %T %Z')] building $DATA_SOURCE chunk1200 graph chunks=$chunks" | tee "$LOG"
CUDA_VISIBLE_DEVICES=0 PYTHONPATH="$CODE_ROOT" "${PYTHON[@]}" "$CODE_ROOT/scripts/build_harness_g_graph.py" \
  --data_source "$DATA_SOURCE" \
  --corpus_path "$CORPUS_INPUT" \
  --output_dir "$BUILDING_GRAPH" \
  --use_spacy true \
  --spacy_model en_core_web_sm \
  --spacy_batch_size 1024 \
  --spacy_n_process 4 \
  --spacy_gpu false \
  --build_embeddings true \
  --embedding_backend bge \
  --embedding_model_path BAAI/bge-large-en-v1.5 \
  --embedding_batch_size 64 \
  --embedding_device cuda \
  --build_sentence_edges true \
  --build_entity_synonyms true \
  --entity_synonym_topk 5 \
  --entity_synonym_threshold 0.80 \
  --entity_synonym_candidate_limit 256 \
  --reuse_embeddings true \
  2>&1 | tee -a "$LOG"

"${PYTHON[@]}" "$CODE_ROOT/scripts/validate_harness_g_graph.py" \
  --graph_dir "$BUILDING_GRAPH" 2>&1 | tee -a "$LOG"
"${PYTHON[@]}" "$SCRIPT_DIR/validate_dataset_chunk1200_graph.py" \
  --root "$ROOT" --data_source "$DATA_SOURCE" --graph_dir "$BUILDING_GRAPH" \
  --report_path "$BUILDING_REPORT" 2>&1 | tee -a "$LOG"

mv "$BUILDING_GRAPH" "$FINAL_GRAPH"
rm -f "$BUILDING_REPORT"
"${PYTHON[@]}" "$SCRIPT_DIR/validate_dataset_chunk1200_graph.py" \
  --root "$ROOT" --data_source "$DATA_SOURCE" --graph_dir "$FINAL_GRAPH" \
  --report_path "$REPORT" 2>&1 | tee -a "$LOG"
echo "[$(date '+%F %T %Z')] graph ready: $FINAL_GRAPH" | tee -a "$LOG"
