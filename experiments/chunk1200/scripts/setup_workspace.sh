#!/usr/bin/env bash
set -euo pipefail

source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"
CODE_ROOT=${CODE_ROOT:-$REPO_ROOT}
SOURCE_DATA_ROOT=${SOURCE_DATA_ROOT:-$REPO_ROOT/datasets}
WORKSPACE="$ROOT/workspace"

if [[ ! -d "$CODE_ROOT/agent" || ! -d "$CODE_ROOT/harness_g" || ! -d "$CODE_ROOT/verl" ]]; then
  echo "[ERROR] Graph-R1 runtime code is missing: $CODE_ROOT" >&2
  exit 2
fi

mkdir -p \
  "$ROOT/corpus" \
  "$ROOT/data" \
  "$ROOT/graph" \
  "$ROOT/logs" \
  "$ROOT/reports" \
  "$ROOT/runs" \
  "$ROOT/checkpoints" \
  "$ROOT/expr_results" \
  "$WORKSPACE"

link_managed() {
  local source=$1
  local target=$2
  if [[ ! -e "$source" ]]; then
    echo "[ERROR] link source is missing: $source" >&2
    exit 2
  fi
  if [[ -L "$target" ]]; then
    if [[ "$(readlink -f "$target")" == "$(readlink -f "$source")" ]]; then
      return
    fi
    # Only a managed symlink is replaced; real files/directories are preserved.
    rm -- "$target"
  elif [[ -e "$target" ]]; then
    echo "[ERROR] refusing to replace non-symlink path: $target" >&2
    exit 2
  fi
  ln -s "$source" "$target"
}

link_managed "$CODE_ROOT/agent" "$WORKSPACE/agent"
link_managed "$CODE_ROOT/harness_g" "$WORKSPACE/harness_g"
link_managed "$CODE_ROOT/verl" "$WORKSPACE/verl"
link_managed "$CODE_ROOT/scripts" "$WORKSPACE/scripts"
link_managed "$CODE_ROOT/tests" "$WORKSPACE/tests"
link_managed "$SOURCE_DATA_ROOT" "$WORKSPACE/datasets"
link_managed "$CODE_ROOT/script_process_harness_g.py" "$WORKSPACE/script_process_harness_g.py"

link_managed "$ROOT/runs" "$WORKSPACE/runs"
link_managed "$ROOT/checkpoints" "$WORKSPACE/checkpoints"
link_managed "$ROOT/expr_results" "$WORKSPACE/expr_results"

printf '%s\n' \
  "workspace=$WORKSPACE" \
  "code_root=$CODE_ROOT" \
  "artifact_root=$ROOT"
