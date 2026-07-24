#!/usr/bin/env python
"""Keep the best Harness-G checkpoints for one experiment."""

from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path


STEP_RE = re.compile(r"(?:evals_step|global_step_)(\d+)")


def step_from_name(name: str) -> int | None:
    match = STEP_RE.search(name)
    return int(match.group(1)) if match else None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--data_source", required=True)
    parser.add_argument("--keep", type=int, default=3)
    parser.add_argument("--metric", default="")
    parser.add_argument("--expr_results_dir", default="expr_results")
    parser.add_argument("--checkpoint_root", default="checkpoints/Harness-G")
    parser.add_argument("--selection_split", choices=("dev", "test"), default="dev")
    parser.add_argument("--allow_test_selection", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    if args.selection_split == "test" and not args.allow_test_selection:
        raise SystemExit(
            "refusing to select/prune checkpoints on test metrics; use dev or "
            "pass --allow_test_selection only for an explicitly documented legacy run"
        )

    metric = args.metric or f"val/test_score/{args.data_source}"
    expr_dir = Path(args.expr_results_dir) / args.experiment
    ckpt_dir = Path(args.checkpoint_root) / args.experiment
    if not expr_dir.exists():
        raise SystemExit(f"missing eval directory: {expr_dir}")
    if not ckpt_dir.exists():
        raise SystemExit(f"missing checkpoint directory: {ckpt_dir}")

    scored: list[dict[str, object]] = []
    for eval_file in sorted(expr_dir.glob("evals_step*.json")):
        step = step_from_name(eval_file.name)
        if step is None:
            continue
        data = json.loads(eval_file.read_text(encoding="utf-8"))
        if metric not in data:
            continue
        checkpoint = ckpt_dir / f"global_step_{step}"
        scored.append(
            {
                "step": step,
                "score": float(data[metric]),
                "eval_file": str(eval_file),
                "checkpoint": str(checkpoint),
                "checkpoint_exists": checkpoint.exists(),
            }
        )

    scored.sort(key=lambda row: (float(row["score"]), int(row["step"])), reverse=True)
    keep_rows = scored[: max(args.keep, 0)]
    keep_steps = {int(row["step"]) for row in keep_rows}
    removed: list[str] = []
    for checkpoint in sorted(ckpt_dir.glob("global_step_*")):
        step = step_from_name(checkpoint.name)
        if step is None or step in keep_steps:
            continue
        removed.append(str(checkpoint))
        if not args.dry_run:
            shutil.rmtree(checkpoint)

    summary = {
        "experiment": args.experiment,
        "data_source": args.data_source,
        "metric": metric,
        "selection_split": args.selection_split,
        "kept": keep_rows,
        "removed": removed,
        "dry_run": args.dry_run,
    }
    out_file = expr_dir / "checkpoint_selection.json"
    if not args.dry_run:
        out_file.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
