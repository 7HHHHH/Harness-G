#!/usr/bin/env python3
"""Validate that the built graph preserved chunk1200 paragraph boundaries."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


DEFAULT_ARTIFACT_ROOT = os.path.expanduser(
    os.environ.get("CHUNK1200_ROOT", "~/harness_g_chunk1200_experiment")
)

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        default=DEFAULT_ARTIFACT_ROOT,
    )
    parser.add_argument("--expected_chunks", type=int, default=2811)
    args = parser.parse_args()

    root = Path(args.root).resolve()
    graph_dir = root / "graph" / "harness_g_graph"
    manifest = json.loads((graph_dir / "graph_manifest.json").read_text(encoding="utf-8"))
    corpus_rows = [
        json.loads(line)
        for line in (root / "corpus" / "graph_input.jsonl").read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    ]
    passages = pd.read_parquet(graph_dir / "passages.parquet")

    expected_text = {str(row["id"]): row["contents"] for row in corpus_rows}
    actual_text = {
        str(row.doc_id): row.text for row in passages.itertuples(index=False)
    }
    checks = {
        "manifest_num_paragraphs": manifest.get("num_paragraphs") == args.expected_chunks,
        "passage_rows": len(passages) == args.expected_chunks,
        "unique_doc_ids": passages["doc_id"].astype(str).nunique() == args.expected_chunks,
        "structured_titles_empty": not passages["title"].fillna("").astype(str).str.strip().any(),
        "chunk_text_exact": actual_text == expected_text,
        "spacy_unchanged": str(manifest.get("entity_extractor", "")).startswith("spacy:"),
        "embedding_unchanged": manifest.get("embedding_backend") == "bge_transformers",
        "embedding_model_unchanged": manifest.get("embedding_model_path")
        == "BAAI/bge-large-en-v1.5",
        "undirected_unchanged": manifest.get("graph_directed") is False,
        "sentence_edges_unchanged": manifest.get("sentence_adjacency_edges") is True,
        "entity_synonyms_unchanged": manifest.get("entity_synonym_edges") is True,
        "entity_synonym_topk_unchanged": int(manifest.get("entity_synonym_topk", -1)) == 5,
        "entity_synonym_threshold_unchanged": float(
            manifest.get("entity_synonym_threshold", -1)
        )
        == 0.8,
    }
    report = {
        "ok": all(checks.values()),
        "checks": checks,
        "graph_dir": str(graph_dir),
        "num_paragraphs": manifest.get("num_paragraphs"),
        "num_sentences": manifest.get("num_sentences"),
        "num_entities": manifest.get("num_entities"),
        "num_mentions": manifest.get("num_mentions"),
    }
    report_path = root / "reports" / "graph_validation.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["ok"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
