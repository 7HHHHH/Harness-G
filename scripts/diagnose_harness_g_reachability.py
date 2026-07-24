#!/usr/bin/env python3
import argparse
import json
import random
import sys
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from harness_g.graph_index import HarnessGGraphIndex
from harness_g.text_utils import contains_any_answer


def _load_dataset(data_source: str, dataset_path):
    path = Path(dataset_path) if dataset_path else Path("datasets") / data_source / "raw" / "qa_test.json"
    with path.open("r", encoding="utf-8") as f:
        return json.load(f), path


def _answers(example):
    value = example.get("golden_answers", example.get("answers", example.get("answer", [])))
    if isinstance(value, list):
        return [str(item) for item in value]
    return [str(value)]


def diagnose(args) -> dict:
    index = HarnessGGraphIndex.load(args.graph_dir)
    rows, dataset_path = _load_dataset(args.data_source, args.dataset_path)
    if args.sample_size and args.sample_size < len(rows):
        rows = rows[: args.sample_size]

    init_hits = 0
    context_hits = 0
    bridge_hits = 0
    expanded_hits = 0
    oracle_success = 0
    no_entity = 0
    entities_per_selected = []
    expansion_candidate_counts = []
    oracle_lengths = []

    for example in rows:
        question = example.get("question", "")
        answers = _answers(example)
        visible = index.hybrid_initial_retrieve(
            question,
            paragraph_topk=args.paragraph_topk,
            high_conf_chunk_k=args.high_conf_chunk_k,
            topk=args.visible_sentence_k,
        )
        if contains_any_answer(" ".join(s.get("text", "") for s in visible), answers):
            init_hits += 1
            oracle_success += 1
            oracle_lengths.append(1)
            continue

        chosen = visible[0] if visible else None
        if not chosen:
            continue
        entities = index.get_entities_for_sentence(chosen["sid"])
        entities_per_selected.append(len(entities))

        context = index.get_local_context(chosen["sid"])
        if contains_any_answer(" ".join(s.get("text", "") for s in context), answers):
            context_hits += 1
            oracle_success += 1
            oracle_lengths.append(2)
            continue

        if not entities:
            no_entity += 1
            continue

        hit = False
        bridge_candidates = index.propose_bridge_entities(
            [entity["eid"] for entity in entities],
            question,
            [chosen["sid"]],
            topm=args.bridge_entity_topm,
        )
        for candidate in bridge_candidates:
            bridged = index.bridge_entity(
                candidate.get("source_eid", ""),
                candidate.get("target_eid") or candidate.get("eid", ""),
                question,
                [chosen["sid"]],
                topk=args.expanded_visible_sentence_k,
            )
            expansion_candidate_counts.append(len(bridged))
            if contains_any_answer(" ".join(s.get("text", "") for s in bridged), answers):
                bridge_hits += 1
                oracle_success += 1
                oracle_lengths.append(3)
                hit = True
                break
        if hit:
            continue

        for entity in entities:
            expanded = index.expand_entity(entity["eid"], question, f"{entity.get('canonical', '')} {question}", topk=args.expanded_visible_sentence_k)
            expansion_candidate_counts.append(len(expanded))
            if contains_any_answer(" ".join(s.get("text", "") for s in expanded), answers):
                expanded_hits += 1
                oracle_success += 1
                oracle_lengths.append(4)
                hit = True
                break
        if not hit:
            oracle_lengths.append(5)

    total = max(len(rows), 1)
    report = {
        "data_source": args.data_source,
        "dataset_path": str(dataset_path),
        "sample_size": len(rows),
        "init_visible_contains_answer_rate": init_hits / total,
        "init_visible_contains_gold_support_rate": None,
        "context_visible_contains_answer_rate": context_hits / total,
        "bridge_visible_contains_answer_rate": bridge_hits / total,
        "expanded_visible_contains_answer_rate": expanded_hits / total,
        "oracle_success_within_5_turns": oracle_success / total,
        "oracle_path_length_mean": mean(oracle_lengths) if oracle_lengths else 0.0,
        "no_entity_sentence_rate": no_entity / total,
        "avg_entities_per_selected_sentence": mean(entities_per_selected) if entities_per_selected else 0.0,
        "avg_candidate_sentences_per_expansion": mean(expansion_candidate_counts) if expansion_candidate_counts else 0.0,
    }
    return report


def write_reports(graph_dir: Path, report: dict) -> None:
    json_path = graph_dir / "reachability_report.json"
    md_path = graph_dir / "reachability_report.md"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    lines = ["# Harness-G Reachability Report", ""]
    lines.extend(f"- {key}: {value}" for key, value in report.items())
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[Harness-G] reachability report: {json_path}")


def build_arg_parser():
    parser = argparse.ArgumentParser(description="Diagnose Harness-G graph answer reachability.")
    parser.add_argument("--graph_dir", required=True)
    parser.add_argument("--data_source", default="2WikiMultiHopQA")
    parser.add_argument("--dataset_path", default=None)
    parser.add_argument("--sample_size", type=int, default=20)
    parser.add_argument("--paragraph_topk", type=int, default=20)
    parser.add_argument("--high_conf_chunk_k", type=int, default=5)
    parser.add_argument("--visible_sentence_k", type=int, default=6)
    parser.add_argument("--expanded_visible_sentence_k", type=int, default=6)
    parser.add_argument("--bridge_entity_topm", type=int, default=5)
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    report = diagnose(args)
    write_reports(Path(args.graph_dir), report)
    print(json.dumps(report, indent=2))
    if report["init_visible_contains_answer_rate"] == 0 and report["expanded_visible_contains_answer_rate"] == 0:
        print("[Harness-G][WARN] reachability is zero; training script will gate unless forced.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
