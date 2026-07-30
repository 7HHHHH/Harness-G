#!/usr/bin/env python3
"""Create one-record-per-1200-token chunk corpora for Harness-G datasets."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ARTIFACT_ROOT = os.path.expanduser(
    os.environ.get("EXPERIMENT_ROOT", "~/harness_g_experiments")
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_source", required=True)
    parser.add_argument("--source", default=None)
    parser.add_argument(
        "--root",
        default=DEFAULT_ARTIFACT_ROOT,
    )
    parser.add_argument(
        "--dataset_root",
        default=os.environ.get(
            "DATASET_ROOT", str(Path(DEFAULT_ARTIFACT_ROOT) / "datasets")
        ),
    )
    parser.add_argument("--max_token_size", type=int, default=1200)
    parser.add_argument("--overlap_token_size", type=int, default=100)
    parser.add_argument("--tiktoken_model", default="gpt-4o-mini")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if args.max_token_size <= 0:
        raise ValueError("max_token_size must be positive")
    if not 0 <= args.overlap_token_size < args.max_token_size:
        raise ValueError("overlap_token_size must be in [0, max_token_size)")

    root = Path(args.root).resolve()
    source = (
        Path(args.source).resolve()
        if args.source
        else Path(args.dataset_root).resolve() / args.data_source / "corpus.jsonl"
    )
    if not source.is_file():
        raise FileNotFoundError(source)

    final_dir = root / "corpora" / args.data_source
    building_dir = final_dir.with_name(final_dir.name + ".building")
    if building_dir.exists():
        shutil.rmtree(building_dir)
    if final_dir.exists():
        if not args.force:
            raise FileExistsError(f"refusing to overwrite {final_dir}; pass --force")
        shutil.rmtree(final_dir)
    building_dir.mkdir(parents=True)

    source_snapshot = building_dir / "source_merged_blocks.jsonl"
    chunk_snapshot = building_dir / "source_chunks.jsonl"
    graph_input = building_dir / "graph_input.jsonl"
    chunk_index = building_dir / "chunk_index.jsonl"
    manifest_path = building_dir / "corpus_manifest.json"
    shutil.copy2(source, source_snapshot)

    import tiktoken

    encoder = tiktoken.encoding_for_model(args.tiktoken_model)
    stride = args.max_token_size - args.overlap_token_size
    chunks: list[dict[str, str]] = []
    index_rows: list[dict[str, int | str]] = []
    source_blocks = 0
    source_tokens = 0
    max_slice_tokens = 0

    with source_snapshot.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            block = json.loads(line)
            contents = str(block.get("contents") or "")
            if not contents.strip():
                raise ValueError(f"empty contents at source line {line_number}")
            block_id = str(block.get("id", source_blocks))
            tokens = encoder.encode(contents)
            source_tokens += len(tokens)
            source_blocks += 1
            for local_index, start in enumerate(range(0, len(tokens), stride)):
                token_slice = tokens[start : start + args.max_token_size]
                text = encoder.decode(token_slice).strip()
                if not text:
                    raise ValueError(
                        f"empty chunk for block={block_id!r} local_index={local_index}"
                    )
                chunk_id = str(len(chunks))
                chunks.append({"id": chunk_id, "contents": text})
                index_rows.append(
                    {
                        "chunk_id": chunk_id,
                        "source_block_id": block_id,
                        "source_line": line_number,
                        "chunk_order_index": local_index,
                        "start_token": start,
                        "slice_tokens": len(token_slice),
                    }
                )
                max_slice_tokens = max(max_slice_tokens, len(token_slice))

    with chunk_snapshot.open("w", encoding="utf-8") as handle:
        for row in chunks:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    with chunk_index.open("w", encoding="utf-8") as handle:
        for row in index_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    sys.path.insert(0, str(REPO_ROOT))
    from harness_g.corpus_loader import iter_corpus_records
    from harness_g.text_utils import normalize_text

    normalized_rows = [
        {"id": row["id"], "contents": normalize_text(row["contents"])}
        for row in chunks
    ]
    if any(not row["contents"] for row in normalized_rows):
        raise ValueError("normalization produced an empty chunk")
    with graph_input.open("w", encoding="utf-8") as handle:
        for row in normalized_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    records, warnings, source_type = iter_corpus_records(graph_input)
    loaded = list(records)
    if len(loaded) != len(normalized_rows):
        raise ValueError(
            f"loader changed chunk boundaries: expected {len(normalized_rows)}, got {len(loaded)}"
        )
    if any(record.get("title") for record in loaded):
        raise ValueError("chunk input unexpectedly produced structured titles")
    if [record["doc_id"] for record in loaded] != [row["id"] for row in normalized_rows]:
        raise ValueError("loader changed chunk ids or ordering")
    if [record["text"] for record in loaded] != [row["contents"] for row in normalized_rows]:
        raise ValueError("loader changed normalized chunk text")

    reencoded_lengths = [len(encoder.encode(row["contents"])) for row in chunks]
    manifest = {
        "data_source": args.data_source,
        "source_path": str(source),
        "source_sha256": sha256(source),
        "source_snapshot": str(final_dir / source_snapshot.name),
        "source_snapshot_sha256": sha256(source_snapshot),
        "chunk_snapshot": str(final_dir / chunk_snapshot.name),
        "chunk_snapshot_sha256": sha256(chunk_snapshot),
        "graph_input": str(final_dir / graph_input.name),
        "graph_input_sha256": sha256(graph_input),
        "chunk_index": str(final_dir / chunk_index.name),
        "chunk_index_sha256": sha256(chunk_index),
        "num_source_blocks": source_blocks,
        "num_chunks": len(chunks),
        "source_tokens": source_tokens,
        "max_token_size": args.max_token_size,
        "overlap_token_size": args.overlap_token_size,
        "stride_tokens": stride,
        "max_slice_tokens": max_slice_tokens,
        "max_reencoded_chunk_tokens": max(reencoded_lengths, default=0),
        "reencoded_chunks_over_slice_limit": sum(
            length > args.max_token_size for length in reencoded_lengths
        ),
        "tiktoken_model": args.tiktoken_model,
        "loader_records": len(loaded),
        "loader_structured_titles": 0,
        "loader_warnings": warnings,
        "loader_source_type": source_type,
        "invariant": "one JSONL record equals one Harness-G paragraph",
        "ok": True,
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    building_dir.rename(final_dir)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
