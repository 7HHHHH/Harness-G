#!/usr/bin/env bash
set -euo pipefail

source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"
CODE_ROOT=${CODE_ROOT:-$REPO_ROOT}
CONDA_BIN=${CONDA_BIN:-conda}
PYTHON=("$CONDA_BIN" run -n s3 python)
FINAL_GRAPH="$ROOT/graph/harness_g_graph"
BUILDING_GRAPH="$ROOT/graph/harness_g_graph.building"

if pgrep -af "verl.trainer.main_ppo" >/dev/null; then
  echo "[ERROR] a main_ppo training process is still running; refusing graph build" >&2
  pgrep -af "verl.trainer.main_ppo" >&2 || true
  exit 3
fi
if [[ -f "$FINAL_GRAPH/graph_manifest.json" ]]; then
  echo "[chunk1200] graph already complete: $FINAL_GRAPH"
  exit 0
fi
if [[ -e "$BUILDING_GRAPH" ]]; then
  echo "[ERROR] partial graph directory exists: $BUILDING_GRAPH" >&2
  exit 4
fi

mkdir -p "$ROOT/graph" "$ROOT/logs"
cd "$ROOT/workspace"
PYTHONPATH="$CODE_ROOT" "${PYTHON[@]}" "$CODE_ROOT/scripts/build_harness_g_graph.py" \
  --data_source 2WikiMultiHopQA \
  --corpus_path "$ROOT/corpus/graph_input.jsonl" \
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
  2>&1 | tee "$ROOT/logs/build_graph.log"

"${PYTHON[@]}" "$CODE_ROOT/scripts/validate_harness_g_graph.py" \
  --graph_dir "$BUILDING_GRAPH" \
  2>&1 | tee "$ROOT/logs/validate_graph_storage.log"

mv "$BUILDING_GRAPH" "$FINAL_GRAPH"
"${PYTHON[@]}" "$SCRIPT_DIR/validate_graph.py" --root "$ROOT" \
  2>&1 | tee "$ROOT/logs/validate_chunk_boundaries.log"
