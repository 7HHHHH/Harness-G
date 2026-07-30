#!/usr/bin/env python3
"""Validate one dataset-specific Harness-G graph."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path


DEFAULT_ARTIFACT_ROOT = os.path.expanduser(
    os.environ.get("EXPERIMENT_ROOT", "~/harness_g_experiments")
)

import pandas as pd


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_source", required=True)
    parser.add_argument(
        "--root",
        default=DEFAULT_ARTIFACT_ROOT,
    )
    parser.add_argument("--graph_dir", default=None)
    parser.add_argument("--report_path", default=None)
    args = parser.parse_args()

    root = Path(args.root).resolve()
    corpus_dir = root / "corpora" / args.data_source
    graph_dir = (
        Path(args.graph_dir).resolve()
        if args.graph_dir
        else root / "graphs" / args.data_source / "harness_g_graph"
    )
    report_path = (
        Path(args.report_path).resolve()
        if args.report_path
        else root / "reports" / args.data_source / "graph_validation.json"
    )

    corpus_manifest = json.loads(
        (corpus_dir / "corpus_manifest.json").read_text(encoding="utf-8")
    )
    graph_manifest = json.loads(
        (graph_dir / "graph_manifest.json").read_text(encoding="utf-8")
    )
    corpus_rows = [
        json.loads(line)
        for line in (corpus_dir / "graph_input.jsonl").read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    ]
    passages = pd.read_parquet(graph_dir / "passages.parquet")
    expected_count = int(corpus_manifest["num_chunks"])
    expected_text = {str(row["id"]): row["contents"] for row in corpus_rows}
    actual_text = {
        str(row.doc_id): row.text for row in passages.itertuples(index=False)
    }

    required_graph_files = [
        "graph_manifest.json",
        "passages.parquet",
        "sentences.parquet",
        "entities.parquet",
        "mentions.parquet",
        "id_maps.json",
        "sentence_entity.npz",
        "passage_entity.npz",
        "sentence_sentence.npz",
        "sentence_sentence_edges.parquet",
        "entity_embeddings.npy",
        "passage_embeddings.npy",
        "sentence_embeddings.npy",
        "entity_entity_synonym.npz",
        "entity_synonym_edges.parquet",
    ]
    checks = {
        "corpus_manifest_ok": corpus_manifest.get("ok") is True,
        "data_source_matches": corpus_manifest.get("data_source") == args.data_source
        and graph_manifest.get("data_source") == args.data_source,
        "chunk_size_1200": int(corpus_manifest.get("max_token_size", -1)) == 1200,
        "chunk_overlap_100": int(corpus_manifest.get("overlap_token_size", -1)) == 100,
        "slice_limit_preserved": int(corpus_manifest.get("max_slice_tokens", -1)) <= 1200,
        "graph_input_hash": sha256(corpus_dir / "graph_input.jsonl")
        == corpus_manifest.get("graph_input_sha256"),
        "chunk_snapshot_hash": sha256(corpus_dir / "source_chunks.jsonl")
        == corpus_manifest.get("chunk_snapshot_sha256"),
        "loader_one_record_per_chunk": int(corpus_manifest.get("loader_records", -1))
        == expected_count,
        "loader_no_structured_titles": int(
            corpus_manifest.get("loader_structured_titles", -1)
        )
        == 0,
        "corpus_rows": len(corpus_rows) == expected_count,
        "corpus_unique_ids": len(expected_text) == expected_count,
        "manifest_num_paragraphs": int(graph_manifest.get("num_paragraphs", -1))
        == expected_count,
        "manifest_num_passages": int(graph_manifest.get("num_passages", -1))
        == expected_count,
        "passage_rows": len(passages) == expected_count,
        "passage_unique_ids": passages["doc_id"].astype(str).nunique()
        == expected_count,
        "structured_titles_empty": not passages["title"]
        .fillna("")
        .astype(str)
        .str.strip()
        .any(),
        "chunk_text_exact": actual_text == expected_text,
        "spacy_unchanged": str(graph_manifest.get("entity_extractor", "")).startswith(
            "spacy:"
        ),
        "embedding_unchanged": graph_manifest.get("embedding_backend")
        == "bge_transformers",
        "embedding_model_unchanged": graph_manifest.get("embedding_model_path")
        == "BAAI/bge-large-en-v1.5",
        "undirected_unchanged": graph_manifest.get("graph_directed") is False,
        "sentence_edges_unchanged": graph_manifest.get("sentence_adjacency_edges")
        is True,
        "entity_synonyms_unchanged": graph_manifest.get("entity_synonym_edges")
        is True,
        "entity_synonym_topk_unchanged": int(
            graph_manifest.get("entity_synonym_topk", -1)
        )
        == 5,
        "entity_synonym_threshold_unchanged": float(
            graph_manifest.get("entity_synonym_threshold", -1)
        )
        == 0.8,
        "required_graph_files_nonempty": all(
            (graph_dir / name).is_file() and (graph_dir / name).stat().st_size > 0
            for name in required_graph_files
        ),
    }
    report = {
        "ok": all(checks.values()),
        "data_source": args.data_source,
        "checks": checks,
        "corpus_dir": str(corpus_dir),
        "graph_dir": str(graph_dir),
        "num_source_blocks": corpus_manifest.get("num_source_blocks"),
        "num_chunks": expected_count,
        "num_paragraphs": graph_manifest.get("num_paragraphs"),
        "num_sentences": graph_manifest.get("num_sentences"),
        "num_entities": graph_manifest.get("num_entities"),
        "num_mentions": graph_manifest.get("num_mentions"),
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["ok"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
