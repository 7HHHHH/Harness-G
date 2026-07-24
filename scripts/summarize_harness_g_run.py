#!/usr/bin/env python
"""Summarize one Harness-G training run."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--data_source", required=True)
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--summary_dir", required=True)
    parser.add_argument("--validation_split", choices=("dev", "test"), default="dev")
    parser.add_argument("--allow_test_selection", action="store_true")
    args = parser.parse_args()

    metric = f"val/test_score/{args.data_source}"
    expr_dir = Path("expr_results") / args.experiment
    evals = []
    for fp in sorted(expr_dir.glob("evals_step*.json")):
        match = re.search(r"step(\d+)", fp.name)
        if not match:
            continue
        data = json.loads(fp.read_text(encoding="utf-8"))
        evals.append({"step": int(match.group(1)), **data})
    selection_allowed = args.validation_split != "test" or args.allow_test_selection
    best = (
        max(evals, key=lambda row: row.get(metric, float("-inf")))
        if evals and selection_allowed
        else {}
    )
    final_eval = max(evals, key=lambda row: row["step"]) if evals else {}

    keys = [
        "valid_action_id_count",
        "invalid_action_count",
        "select_count",
        "stop_count",
        "answer_after_stop",
        "open_context_count",
        "bridge_entity_count",
        "expand_entity_count",
        "rewrite_count",
    ]
    reward_path = Path(args.run_dir) / "reward_metrics.jsonl"
    rows = []
    if reward_path.exists():
        with reward_path.open("r", encoding="utf-8") as f:
            for line in f:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    tail = rows[-512:] if rows else []
    reward_summary = {
        "rows": len(rows),
        "total": {key: sum(row.get(key, 0) for row in rows) for key in keys},
        "last512": {key: sum(row.get(key, 0) for row in tail) for key in keys},
        "last512_mean_reward": sum(row.get("total_reward", 0) for row in tail) / len(tail) if tail else None,
        "last512_mean_f1": sum(row.get("answer_f1", 0) for row in tail) / len(tail) if tail else None,
        "last512_mean_em": sum(row.get("answer_em", 0) for row in tail) / len(tail) if tail else None,
    }
    selection_file = expr_dir / "checkpoint_selection.json"
    selection = json.loads(selection_file.read_text(encoding="utf-8")) if selection_file.exists() else None
    summary = {
        "experiment": args.experiment,
        "data_source": args.data_source,
        "metric": metric,
        "validation_split": args.validation_split,
        "selection_allowed": selection_allowed,
        "evals": evals,
        "best_eval": best,
        "final_eval": final_eval,
        "checkpoint_selection": selection,
        "reward_metrics": reward_summary,
    }
    summary_dir = Path(args.summary_dir)
    summary_dir.mkdir(parents=True, exist_ok=True)
    out = summary_dir / f"{args.data_source}_summary.json"
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "data_source": args.data_source,
        "best_eval": best,
        "final_eval": final_eval,
        "selection_allowed": selection_allowed,
        "summary": str(out),
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
