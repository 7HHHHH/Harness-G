#!/usr/bin/env python3
import argparse
import json
import os
import random
from pathlib import Path

import datasets


# Navigation prompt.  Describes the actions that exist in the
# environment: SELECT / LOOKUP / ANSWER, plus ANSWER_WITH.
_NAVIGATION_V3_BASE = """Answer the question using Harness-G evidence navigation.

You must use this exact tool-call format:
<query>{"query": "ACTION"}</query>

Rules:
1. First call <query>{"query": "INIT"}</query> to get the initial graph observation.
2. After each [HARNESS_G_OBS], choose exactly one available action id, for example <query>{"query": "A0"}</query>.
3. Available actions are {action_list}.
4. SELECT means selecting a useful visible sentence as evidence.
5. LOOKUP means looking up an entity from the current observation to find missing information. Just choose the LOOKUP action id; the retrieval query is built for you from the question and the evidence you have already selected.
6. ANSWER means stop searching and provide the final answer.
{answer_with_rule}7. Multi-hop questions usually require SELECT evidence, then LOOKUP the missing entity, then SELECT supporting evidence about that entity, then ANSWER.
8. Do not answer before the selected evidence covers every hop required by the question.
9. Do not invent action ids. Only choose from the current available_actions.
10. Action ids can change at every step. Read the current available_actions before choosing.
11. Do not write a free-form search query by itself. Always start with an available action id.
12. After choosing ANSWER, provide the final answer in <answer>...</answer>.
13. The final answer must be brief.

Question: {question}
"""

_ANSWER_WITH_RULE = (
    "ANSWER_WITH means selecting a visible sentence as final evidence and "
    "stopping immediately. Use it only when that sentence alone is sufficient "
    "to answer the question.\n"
)


def _navigation_v3_instruction() -> str:
    # Use .replace (not .format) so the literal {"query": ...} braces in the
    # tool-call examples stay intact; {question} is left for the caller to fill.
    return (
        _NAVIGATION_V3_BASE
        .replace("{action_list}", "SELECT, LOOKUP, ANSWER_WITH, and ANSWER")
        .replace("{answer_with_rule}", _ANSWER_WITH_RULE)
    )


NAVIGATION_V3_INSTRUCTION = _navigation_v3_instruction()
INSTRUCTION = NAVIGATION_V3_INSTRUCTION


def _instruction_template() -> str:
    return _navigation_v3_instruction()


def _load_split(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _subset(rows, limit, seed):
    if limit is None or limit < 0 or limit >= len(rows):
        return rows
    if seed is None:
        return rows[:limit]
    rng = random.Random(seed)
    indices = sorted(rng.sample(range(len(rows)), limit))
    return [rows[i] for i in indices]


def _answer_from_example(example):
    if "golden_answers" in example:
        return example["golden_answers"]
    if "answers" in example:
        return example["answers"]
    if "answer" in example:
        return example["answer"]
    return []


def _supporting_evidence_from_example(example):
    for key in ("supporting_evidence", "supporting_evidences", "supporting_facts", "evidences", "evidence"):
        value = example.get(key)
        if value:
            return value
    return []


def _sidecar_key(question, answers) -> str:
    first_answer = answers[0] if answers else ""
    return " ".join(str(question).lower().split()) + " ||| " + " ".join(str(first_answer).lower().split())


def _load_supporting_facts_sidecar(raw_dir: Path) -> dict:
    path = raw_dir / "supporting_facts_sidecar.json"
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _make_dataset(rows, data_source: str, split: str, sidecar=None):
    sidecar = sidecar or {}

    def process_fn(example, idx):
        question_raw = example.get("question", "")
        answer_raw = _answer_from_example(example)
        supporting_evidence = _supporting_evidence_from_example(example)
        gold_evidences = []
        if not supporting_evidence and sidecar:
            answers = answer_raw if isinstance(answer_raw, (list, tuple)) else [answer_raw]
            entry = sidecar.get(_sidecar_key(question_raw, list(answers)))
            if entry:
                supporting_evidence = [fact["text"] for fact in entry.get("supporting_facts", []) if fact.get("text")]
                gold_evidences = entry.get("evidences", [])
        prompt = _instruction_template().replace("{question}", str(question_raw))
        return {
            "data_source": data_source,
            "prompt": [
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            "ability": "multihop_qa",
            "reward_model": {
                "style": "rule",
                "ground_truth": answer_raw,
            },
            "extra_info": {
                "split": split,
                "index": str(idx),
                "answer": answer_raw,
                "question": question_raw,
                "harness_g_version": "v2",
                "supporting_evidence": supporting_evidence,
                "gold_evidences": gold_evidences,
            },
            "context": example.get("context", []),
        }

    dataset = datasets.Dataset.from_list(rows)
    return dataset.map(function=process_fn, with_indices=True)


def build_arg_parser():
    parser = argparse.ArgumentParser(description="Process QA splits with Harness-G v2 prompts.")
    parser.add_argument("--data_source", default="2WikiMultiHopQA")
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--train_limit", type=int, default=None)
    parser.add_argument("--dev_limit", type=int, default=None)
    parser.add_argument("--test_limit", type=int, default=None)
    parser.add_argument("--val_limit", type=int, default=None, help="Alias for --test_limit for GRPO validation debug runs.")
    parser.add_argument("--sample_seed", type=int, default=None)
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    raw_dir = Path("datasets") / args.data_source / "raw"
    output_dir = Path(args.output_dir) if args.output_dir else Path("datasets") / args.data_source / "processed"
    output_dir.mkdir(parents=True, exist_ok=True)

    split_specs = {
        "train": ("qa_train.json", args.train_limit),
        "dev": ("qa_dev.json", args.dev_limit),
        "test": ("qa_test.json", args.test_limit if args.test_limit is not None else args.val_limit),
    }

    subset_report = {}
    sidecar = _load_supporting_facts_sidecar(raw_dir)
    for split, (filename, limit) in split_specs.items():
        rows = _load_split(raw_dir / filename)
        original_len = len(rows)
        rows = _subset(rows, limit, args.sample_seed)
        subset_report[split] = {
            "harness_g_version": "v2",
            "original": original_len,
            "written": len(rows),
            "limit": limit,
            "seed": args.sample_seed,
        }
        dataset = _make_dataset(rows, args.data_source, split, sidecar=sidecar)
        dataset.to_parquet(output_dir / f"{split}.parquet")
        print(f"[Harness-G] wrote {split}.parquet rows={len(rows)} original={original_len}")

    report_path = output_dir / "harness_g_process_report.json"
    report_path.write_text(json.dumps(subset_report, indent=2), encoding="utf-8")
    version_report_path = output_dir / "harness_g_process_report.json"
    version_report_path.write_text(json.dumps(subset_report, indent=2), encoding="utf-8")
    print(f"[Harness-G] process report: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
