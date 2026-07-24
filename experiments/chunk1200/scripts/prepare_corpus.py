#!/usr/bin/env python3
"""Prepare one-record-per-chunk input without changing graph-builder code."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_ARTIFACT_ROOT = os.path.expanduser(
    os.environ.get("CHUNK1200_ROOT", "~/harness_g_chunk1200_experiment")
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        default="data/search_r1_2wiki_chunk1200/corpus.jsonl",
    )
    parser.add_argument(
        "--root",
        default=DEFAULT_ARTIFACT_ROOT,
    )
    parser.add_argument("--code_root", default=str(REPO_ROOT))
    parser.add_argument("--expected_chunks", type=int, default=2811)
    args = parser.parse_args()

    root = Path(args.root).resolve()
    source = Path(args.source).resolve()
    code_root = Path(args.code_root).resolve()
    corpus_dir = root / "corpus"
    corpus_dir.mkdir(parents=True, exist_ok=True)
    source_copy = corpus_dir / "source_chunk1200.jsonl"
    graph_input = corpus_dir / "graph_input.jsonl"
    manifest_path = corpus_dir / "corpus_manifest.json"

    sys.path.insert(0, str(code_root))
    from harness_g.corpus_loader import iter_corpus_records
    from harness_g.text_utils import normalize_text

    shutil.copy2(source, source_copy)

    try:
        import tiktoken

        encoder = tiktoken.encoding_for_model("gpt-4o-mini")
    except Exception:
        encoder = None

    ids: set[str] = set()
    rows: list[dict[str, str]] = []
    max_source_tokens = 0
    reencoded_over_limit = 0
    with source_copy.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            chunk_id = str(row["id"])
            if chunk_id in ids:
                raise ValueError(f"duplicate chunk id {chunk_id!r} at line {line_number}")
            ids.add(chunk_id)
            raw_text = str(row.get("contents") or "")
            normalized = normalize_text(raw_text)
            if not normalized:
                raise ValueError(f"empty chunk {chunk_id!r}")
            if encoder is not None:
                reencoded_tokens = len(encoder.encode(raw_text))
                max_source_tokens = max(max_source_tokens, reencoded_tokens)
                reencoded_over_limit += int(reencoded_tokens > 1200)
            rows.append({"id": chunk_id, "contents": normalized})

    if len(rows) != args.expected_chunks:
        raise ValueError(f"expected {args.expected_chunks} chunks, found {len(rows)}")
    with graph_input.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    records, warnings, source_type = iter_corpus_records(graph_input)
    loaded = list(records)
    if len(loaded) != args.expected_chunks:
        raise ValueError(
            f"Graph-R1 loader changed chunk boundaries: expected {args.expected_chunks}, "
            f"loaded {len(loaded)}"
        )
    if any(record.get("title") for record in loaded):
        raise ValueError("chunk input unexpectedly produced structured titles")
    if [record["doc_id"] for record in loaded] != [row["id"] for row in rows]:
        raise ValueError("loader changed chunk ids or ordering")
    if [record["text"] for record in loaded] != [row["contents"] for row in rows]:
        raise ValueError("loader changed normalized chunk text")

    manifest = {
        "source_path": str(source),
        "source_sha256": sha256(source),
        "source_snapshot": str(source_copy),
        "source_snapshot_sha256": sha256(source_copy),
        "graph_input": str(graph_input),
        "graph_input_sha256": sha256(graph_input),
        "num_chunks": len(rows),
        # The source builder slices at 1,200 encoded tokens, then decodes each
        # slice. Re-encoding decoded text is not perfectly token-idempotent:
        # the frozen source has a measured maximum of 1,202. Preserve the exact
        # baseline corpus instead of truncating it a second time.
        "source_slice_token_limit": 1200,
        "max_reencoded_source_tokens": max_source_tokens if encoder is not None else None,
        "reencoded_chunks_over_slice_limit": reencoded_over_limit if encoder is not None else None,
        "overlap_tokens": 100,
        "tiktoken_model": "gpt-4o-mini",
        "loader_records": len(loaded),
        "loader_structured_titles": 0,
        "loader_warnings": warnings,
        "loader_source_type": source_type,
        "invariant": "one JSONL record equals one Harness-G paragraph",
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
