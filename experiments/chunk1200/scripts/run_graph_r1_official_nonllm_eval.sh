#!/usr/bin/env bash
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"
EVAL_REPO=${EVAL_REPO:-$REPO_ROOT}
OUT_ROOT=${OUT_ROOT:-$ROOT/runs/graph_r1_official_eval_chunk1200/non_llm}
LOG=${LOG:-$ROOT/logs/graph_r1_official_nonllm_eval.log}
PID_FILE=${PID_FILE:-$ROOT/logs/graph_r1_official_nonllm_eval.pid}
LOCK=${LOCK:-$ROOT/logs/graph_r1_official_nonllm_eval.lock}
PY=${PYTHON_BIN:-python}
mkdir -p "$OUT_ROOT" "$ROOT/logs"
exec 9>"$LOCK"
flock -n 9 || { echo "another non-LLM evaluator is active" >&2; exit 9; }
printf '%s\n' "$$" > "$PID_FILE"
log(){ printf '[%s] %s\n' "$(date '+%F %T %Z')" "$*" | tee -a "$LOG"; }
rows=(
  "2WikiMultiHopQA 2wiki"
  "HotpotQA hotpotqa"
  "Musique musique"
  "NQ nq"
  "PopQA popqa"
  "TriviaQA triviaqa"
)
log "START Graph-R1 official EM/F1/R-Sim; selection=best-dev-F1; models=3b,1p5b"
for size in 3b 1p5b; do
  for row in "${rows[@]}"; do
    read -r dataset slug <<< "$row"
    exp=$(printf 'harness_g_snc_full_chunk1200_f1_%s_%s_b128_120_8g' "$size" "$slug")
    expr="$ROOT/expr_results/$exp"
    read -r step f1 <<< "$("$PY" - "$expr" <<'PY'
import glob,json,math,os,re,sys
expr=sys.argv[1]; vals=[]
for p in glob.glob(os.path.join(expr,'evals_step*.json')):
 m=re.search(r'step(\d+)',os.path.basename(p))
 if not m:continue
 try:
  d=json.load(open(p));ks=[k for k in d if 'answer_f1_score' in k]
  v=float(d[ks[0]]) if ks else float(next(d[k] for k in d if 'f1' in k.lower()))
  if math.isfinite(v):vals.append((v,int(m.group(1))))
 except Exception:pass
assert vals,expr
v,s=max(vals)
print(s,f'{v:.12f}')
PY
)"
    results="$expr/results_step$step.json"
    out="$OUT_ROOT/$size/${dataset}_step$step"
    mkdir -p "$out"
    if [[ -s "$out/NON_LLM_COMPLETE" && -s "$out/test_score.json" && -s "$out/test_result.json" ]]; then
      log "SKIP complete model=$size dataset=$dataset step=$step"
      continue
    fi
    [[ -s "$results" ]] || { log "ERROR missing $results"; exit 20; }
    rm -f "$out/test_score.json" "$out/test_result.json" "$out/NON_LLM_COMPLETE"
    log "EVAL model=$size dataset=$dataset step=$step dev_f1=$f1"
    (
      cd "$EVAL_REPO"
      TOKENIZERS_PARALLELISM=false GRAPH_R1_RSIM_LOCAL_ONLY=1 PYTHONPATH=evaluation \
        conda run -n s3 python evaluation/get_remote_score.py \
        --results_file "$results" --out_dir "$out" --workers 4 --skip_gen
    ) >> "$LOG" 2>&1
    "$PY" - "$out/test_score.json" "$results" <<'PY'
import json,math,sys
p,r=sys.argv[1:];d=json.load(open(p))
assert d['results_file']==r,(d['results_file'],r)
for k in ('overall_em','overall_f1','overall_rsim'):
 v=float(d[k]);assert math.isfinite(v), (k,v)
assert float(d['overall_gen'])==0.0,d
PY
    printf 'completed=%s\nmodel=%s\ndataset=%s\nstep=%s\n' "$(date '+%F %T %Z')" "$size" "$dataset" "$step" > "$out/NON_LLM_COMPLETE"
    log "DONE model=$size dataset=$dataset step=$step"
  done
done
"$PY" - "$OUT_ROOT" <<'PY'
from pathlib import Path
import json
root=Path(__import__('sys').argv[1]); rows=[]
for p in sorted(root.glob('*/*/test_score.json')):
 d=json.load(open(p)); model=p.parents[1].name; tag=p.parent.name
 dataset,step=tag.rsplit('_step',1)
 rows.append(dict(model=model,dataset=dataset,step=int(step),**{k:float(d[k]) for k in ('overall_em','overall_f1','overall_rsim')}))
for model in ('3b','1p5b'):
 xs=[x for x in rows if x['model']==model]
 if xs:
  rows.append(dict(model=model,dataset='MacroAverage',step=-1,**{k:sum(x[k] for x in xs)/len(xs) for k in ('overall_em','overall_f1','overall_rsim')}))
(root/'summary.json').write_text(json.dumps({'protocol':'official Graph-R1','selection':'best dev F1','rows':rows},indent=2)+'\n')
PY
printf 'completed=%s\n' "$(date '+%F %T %Z')" > "$OUT_ROOT/ALL_NON_LLM_COMPLETE"
log "ALL_NON_LLM_COMPLETE summary=$OUT_ROOT/summary.json"
