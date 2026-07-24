#!/usr/bin/env bash
set -Eeuo pipefail

source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"
EVAL_REPO=${EVAL_REPO:-$REPO_ROOT}
OUT_ROOT=${OUT_ROOT:-$ROOT/runs/graph_r1_official_eval_chunk1200_step120/full}
LOG=${LOG:-$ROOT/logs/graph_r1_official_step120_full_eval.log}
PID_FILE=${PID_FILE:-$ROOT/logs/graph_r1_official_step120_full_eval.pid}
LOCK=${LOCK:-$ROOT/logs/graph_r1_official_step120_full_eval.lock}
KEY_FILE=${KEY_FILE:-/tmp/graph_r1_ge_key_$(id -u)}
BASE_URL=${OPENAI_BASE_URL:-https://api.openai.com/v1/}
MODEL=${OPENAI_MODEL:-gpt-4o-mini}
WORKERS=${WORKERS:-16}
MAX_FALLBACK_DIMENSIONS=${MAX_FALLBACK_DIMENSIONS:-32}
PY=${PYTHON_BIN:-python}

mkdir -p "$OUT_ROOT" "$ROOT/logs"
exec 9>"$LOCK"
flock -n 9 || { echo "another step-120 full evaluator is active" >&2; exit 9; }
printf '%s\n' "$$" > "$PID_FILE"

log() {
  printf '[%s] %s\n' "$(date '+%F %T %Z')" "$*" | tee -a "$LOG"
}

on_exit() {
  rc=$?
  if (( rc != 0 )); then
    log "ABORT rc=$rc; API key file retained with mode 600 for audited resume"
  fi
}
trap on_exit EXIT

[[ -s "$KEY_FILE" ]] || { log "ERROR missing API key file: $KEY_FILE"; exit 10; }
[[ "$(stat -c '%a' "$KEY_FILE")" == "600" ]] || {
  log "ERROR API key file must have mode 600"
  exit 11
}
export OPENAI_API_KEY
OPENAI_API_KEY=$(cat "$KEY_FILE")
export OPENAI_BASE_URL="$BASE_URL"
export OPENAI_MODEL="$MODEL"
[[ ${#OPENAI_API_KEY} -ge 20 ]] || { log "ERROR invalid empty/short API key"; exit 12; }

rows=(
  "2WikiMultiHopQA 2wiki"
  "HotpotQA hotpotqa"
  "Musique musique"
  "NQ nq"
  "PopQA popqa"
  "TriviaQA triviaqa"
)

log "START Graph-R1 official full evaluation; selection=fixed-step-120; models=3b,1p5b; metrics=EM,F1,R-Sim,G-E; judge=$MODEL; workers=$WORKERS"

for size in 3b 1p5b; do
  for row in "${rows[@]}"; do
    read -r dataset slug <<< "$row"
    exp="harness_g_snc_full_chunk1200_f1_${size}_${slug}_b128_120_8g"
    results="$ROOT/expr_results/$exp/results_step120.json"
    out="$OUT_ROOT/$size/${dataset}_step120"
    mkdir -p "$out"

    "$PY" - "$results" <<'PY'
import json, sys
p = sys.argv[1]
x = json.load(open(p))
assert len(x) == 128, (p, len(x))
required = {"question", "golden_answers", "context", "prediction"}
for i, row in enumerate(x):
    missing = required - row.keys()
    assert not missing, (p, i, sorted(missing))
PY

    if [[ -s "$out/FULL_COMPLETE" && -s "$out/test_score.json" && -s "$out/test_result.json" ]]; then
      log "SKIP complete model=$size dataset=$dataset step=120"
      continue
    fi

    rm -f "$out/test_score.json" "$out/test_result.json" "$out/FULL_COMPLETE"
    log "EVAL model=$size dataset=$dataset step=120"
    (
      cd "$EVAL_REPO"
      TOKENIZERS_PARALLELISM=false \
      GRAPH_R1_RSIM_LOCAL_ONLY=1 \
      PYTHONPATH=evaluation \
        conda run --no-capture-output -n s3 \
          python evaluation/get_remote_score.py \
          --results_file "$results" \
          --out_dir "$out" \
          --workers "$WORKERS"
    ) >> "$LOG" 2>&1

    fallback=$("$PY" - "$out/test_score.json" "$out/test_result.json" "$results" <<'PY'
import json, math, sys
score_path, detail_path, expected_results = sys.argv[1:]
score = json.load(open(score_path))
detail = json.load(open(detail_path))
assert score["results_file"] == expected_results, (score["results_file"], expected_results)
assert len(detail) == 128, len(detail)
for key in ("overall_em", "overall_f1", "overall_rsim", "overall_gen"):
    value = float(score[key])
    assert math.isfinite(value), (key, value)
    assert 0.0 <= value <= 1.0, (key, value)
metrics = {
    "comprehensiveness", "knowledgeability", "correctness", "relevance",
    "diversity", "logical_coherence", "factuality",
}
fallback = 0
for i, row in enumerate(detail):
    exp = row.get("gen_exp", {})
    assert set(exp) == metrics, (i, sorted(exp))
    for metric in metrics:
        value = float(exp[metric]["score"])
        assert 0.0 <= value <= 1.0, (i, metric, value)
        explanation = str(exp[metric].get("explanation", ""))
        if explanation.startswith("Failed to parse GPT output"):
            fallback += 1
print(fallback)
PY
)

    if (( fallback > MAX_FALLBACK_DIMENSIONS )); then
      log "ERROR excessive G-E fallback dimensions model=$size dataset=$dataset fallback=$fallback/896"
      exit 30
    fi

    metrics=$($PY - "$out/test_score.json" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
print("EM={:.4f} F1={:.4f} R-Sim={:.4f} G-E={:.4f}".format(
    d["overall_em"], d["overall_f1"], d["overall_rsim"], d["overall_gen"]
))
PY
)
    printf 'completed=%s\nmodel=%s\ndataset=%s\nstep=120\nfallback_dimensions=%s\n' \
      "$(date '+%F %T %Z')" "$size" "$dataset" "$fallback" > "$out/FULL_COMPLETE"
    log "DONE model=$size dataset=$dataset step=120 $metrics fallback=$fallback/896"
  done
done

"$PY" - "$OUT_ROOT" <<'PY'
from pathlib import Path
import json, sys

root = Path(sys.argv[1])
datasets = ["2WikiMultiHopQA", "HotpotQA", "Musique", "NQ", "PopQA", "TriviaQA"]
keys = ("overall_em", "overall_f1", "overall_rsim", "overall_gen")
rows = []
for model in ("3b", "1p5b"):
    for dataset in datasets:
        d = root / model / f"{dataset}_step120"
        score = json.load(open(d / "test_score.json"))
        detail = json.load(open(d / "test_result.json"))
        fallback = sum(
            str(v.get("explanation", "")).startswith("Failed to parse GPT output")
            for row in detail for v in row["gen_exp"].values()
        )
        rows.append({
            "model": model,
            "dataset": dataset,
            "step": 120,
            **{key: float(score[key]) for key in keys},
            "fallback_dimensions": fallback,
        })
    selected = [row for row in rows if row["model"] == model]
    rows.append({
        "model": model,
        "dataset": "MacroAverage",
        "step": 120,
        **{key: sum(row[key] for row in selected) / len(selected) for key in keys},
        "fallback_dimensions": sum(row["fallback_dimensions"] for row in selected),
    })

summary = {
    "protocol": "official Graph-R1",
    "selection": "fixed step 120",
    "judge_model": "gpt-4o-mini",
    "metrics": ["EM", "F1", "R-Sim", "G-E"],
    "rows": rows,
}
(root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
PY

printf 'completed=%s\n' "$(date '+%F %T %Z')" > "$OUT_ROOT/ALL_FULL_COMPLETE"
log "ALL_FULL_COMPLETE summary=$OUT_ROOT/summary.json"

unset OPENAI_API_KEY
if command -v shred >/dev/null 2>&1; then
  shred -u "$KEY_FILE"
else
  rm -f "$KEY_FILE"
fi
log "API key staging file securely removed"
