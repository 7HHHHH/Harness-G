#!/usr/bin/env python3
import argparse
import re
import sys
from pathlib import Path
from typing import List, Union

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from harness_g.graph_index import HarnessGGraphIndex
from harness_g.utils import read_json


REQUIRED_FILES = [
    "passages.parquet",
    "sentences.parquet",
    "entities.parquet",
    "mentions.parquet",
    "passage_entity.npz",
    "sentence_entity.npz",
    "sentence_sentence_edges.parquet",
    "sentence_sentence.npz",
    "entity_synonym_edges.parquet",
    "entity_entity_synonym.npz",
    "id_maps.json",
    "lexical_index.json",
    "graph_manifest.json",
    "build_report.md",
]


def _read_parquet(path: Path) -> List[dict]:
    try:
        import pandas as pd
    except Exception as exc:
        raise RuntimeError("pandas/pyarrow are required to validate clean Harness-G graph storage") from exc
    return pd.read_parquet(path).to_dict("records")


def _load_npz(path: Path) -> dict:
    import numpy as np

    with np.load(path) as data:
        return {key: data[key] for key in data.files}


def _check_unique(rows: List[dict], key: str, errors: List[str]) -> set:
    seen = set()
    duplicates = set()
    for row in rows:
        value = str(row.get(key))
        if value in seen:
            duplicates.add(value)
        seen.add(value)
    if duplicates:
        errors.append(f"Duplicate {key}: {sorted(duplicates)[:10]}")
    return seen


def _check_matrix(matrix: dict, rows: int, cols: int, name: str, errors: List[str]) -> int:
    required = {"row", "col", "data", "shape"}
    missing = required - set(matrix)
    if missing:
        errors.append(f"{name} missing arrays: {sorted(missing)}")
        return 0
    shape = tuple(int(value) for value in matrix["shape"].tolist())
    if shape != (rows, cols):
        errors.append(f"{name} shape mismatch: matrix={shape} expected={(rows, cols)}")
    if len(matrix["row"]) != len(matrix["col"]) or len(matrix["row"]) != len(matrix["data"]):
        errors.append(f"{name} row/col/data length mismatch")
    if len(matrix["row"]) and (int(matrix["row"].min()) < 0 or int(matrix["row"].max()) >= rows):
        errors.append(f"{name} row index out of bounds")
    if len(matrix["col"]) and (int(matrix["col"].min()) < 0 or int(matrix["col"].max()) >= cols):
        errors.append(f"{name} col index out of bounds")
    return len(matrix["data"])


def validate_graph(graph_dir: Union[str, Path]) -> dict:
    graph_dir = Path(graph_dir)
    errors: List[str] = []

    for name in REQUIRED_FILES:
        if not (graph_dir / name).exists():
            errors.append(f"Missing required file: {name}")

    if errors:
        return {"ok": False, "errors": errors}

    passages = _read_parquet(graph_dir / "passages.parquet")
    sentences = _read_parquet(graph_dir / "sentences.parquet")
    entities = _read_parquet(graph_dir / "entities.parquet")
    mentions = _read_parquet(graph_dir / "mentions.parquet")
    sentence_sentence_edges = _read_parquet(graph_dir / "sentence_sentence_edges.parquet")
    entity_synonym_edges = _read_parquet(graph_dir / "entity_synonym_edges.parquet")
    id_maps = read_json(graph_dir / "id_maps.json")
    manifest = read_json(graph_dir / "graph_manifest.json")

    pids = _check_unique(passages, "pid", errors)
    sids = _check_unique(sentences, "sid", errors)
    eids = _check_unique(entities, "eid", errors)
    mids = _check_unique(mentions, "mid", errors)
    canonical_to_eid = {}
    identity_to_eid = {}
    for entity in entities:
        canonical = str(entity.get("canonical", "") or "")
        eid = str(entity.get("eid", ""))
        if not canonical:
            errors.append(f"entity missing canonical: {entity}")
            continue
        if canonical.lower() != canonical:
            errors.append(f"entity canonical must be lowercase: {entity}")
        if canonical in canonical_to_eid:
            errors.append(f"duplicate entity canonical: {canonical} -> {canonical_to_eid[canonical]}, {eid}")
        canonical_to_eid[canonical] = eid
        identity = " ".join(re.findall(r"[a-z0-9]+", canonical.lower()))
        if identity:
            if identity in identity_to_eid:
                errors.append(f"duplicate entity identity: {identity} -> {identity_to_eid[identity]}, {eid}")
            identity_to_eid[identity] = eid
    sid_to_pid = {str(sentence.get("sid")): str(sentence.get("pid")) for sentence in sentences}

    for sentence in sentences:
        if str(sentence.get("pid")) not in pids:
            errors.append(f"sentence.pid missing: {sentence.get('sid')} -> {sentence.get('pid')}")

    seen_mentions = set()
    for mention in mentions:
        mid = str(mention.get("mid"))
        pid = str(mention.get("pid"))
        sid = str(mention.get("sid"))
        eid = str(mention.get("eid"))
        if mid not in mids:
            errors.append(f"mention missing mid: {mention}")
        if pid not in pids:
            errors.append(f"mention.pid missing: {mention}")
        if sid not in sids:
            errors.append(f"mention.sid missing: {mention}")
        if eid not in eids:
            errors.append(f"mention.eid missing: {mention}")
        sentence_pid = sid_to_pid.get(sid, "")
        if sentence_pid and pid != sentence_pid:
            errors.append(f"mention pid/sid mismatch: {mention}")
        seen_mentions.add((pid, sid, eid))

    expected_maps = {
        "idx_to_pid": len(passages),
        "idx_to_sid": len(sentences),
        "idx_to_eid": len(entities),
        "pid_to_idx": len(passages),
        "sid_to_idx": len(sentences),
        "eid_to_idx": len(entities),
    }
    for key, expected_len in expected_maps.items():
        if len(id_maps.get(key, [] if key.startswith("idx") else {})) != expected_len:
            errors.append(f"id_maps {key} length mismatch")

    pe_edges = len({(str(m["pid"]), str(m["eid"])) for m in mentions})
    se_edges = len({(str(m["sid"]), str(m["eid"])) for m in mentions})
    pe_nnz = _check_matrix(_load_npz(graph_dir / "passage_entity.npz"), len(passages), len(entities), "passage_entity.npz", errors)
    se_nnz = _check_matrix(_load_npz(graph_dir / "sentence_entity.npz"), len(sentences), len(entities), "sentence_entity.npz", errors)
    if pe_nnz != pe_edges:
        errors.append(f"passage_entity nnz mismatch: matrix={pe_nnz} mentions={pe_edges}")
    if se_nnz != se_edges:
        errors.append(f"sentence_entity nnz mismatch: matrix={se_nnz} mentions={se_edges}")

    ss_pairs = set()
    for edge in sentence_sentence_edges:
        sid1 = str(edge.get("sid1"))
        sid2 = str(edge.get("sid2"))
        if sid1 not in sids or sid2 not in sids:
            errors.append(f"sentence_sentence edge missing sid: {edge}")
            continue
        if sid1 == sid2:
            errors.append(f"sentence_sentence self edge: {edge}")
            continue
        pair = tuple(sorted((sid1, sid2)))
        if pair in ss_pairs:
            errors.append(f"duplicate sentence_sentence undirected edge: {pair}")
        ss_pairs.add(pair)
    ss_matrix = _load_npz(graph_dir / "sentence_sentence.npz")
    ss_nnz = _check_matrix(ss_matrix, len(sentences), len(sentences), "sentence_sentence.npz", errors)
    if ss_nnz != len(ss_pairs) * 2:
        errors.append(f"sentence_sentence nnz mismatch: matrix={ss_nnz} edges={len(ss_pairs)}")

    synonym_pairs = set()
    for edge in entity_synonym_edges:
        eid1 = str(edge.get("eid1"))
        eid2 = str(edge.get("eid2"))
        if eid1 not in eids or eid2 not in eids:
            errors.append(f"entity_synonym edge missing eid: {edge}")
            continue
        if eid1 == eid2:
            errors.append(f"entity_synonym self edge: {edge}")
            continue
        pair = tuple(sorted((eid1, eid2)))
        if pair in synonym_pairs:
            errors.append(f"duplicate entity_synonym undirected edge: {pair}")
        synonym_pairs.add(pair)
        try:
            score = float(edge.get("score", 0.0))
            if score < 0.0 or score > 1.0001:
                errors.append(f"entity_synonym score out of range: {edge}")
        except Exception:
            errors.append(f"entity_synonym score invalid: {edge}")
    synonym_matrix = _load_npz(graph_dir / "entity_entity_synonym.npz")
    synonym_nnz = _check_matrix(synonym_matrix, len(entities), len(entities), "entity_entity_synonym.npz", errors)
    if synonym_nnz != len(synonym_pairs) * 2:
        errors.append(f"entity_entity_synonym nnz mismatch: matrix={synonym_nnz} edges={len(synonym_pairs)}")

    expected_counts = {
        "num_paragraphs": len(passages),
        "num_passages": len(passages),
        "num_sentences": len(sentences),
        "num_entities": len(entities),
        "num_mentions": len(mentions),
        "num_ps_edges": len(sentences),
        "num_pe_edges": pe_edges,
        "num_se_edges": se_edges,
        "num_ss_edges": len(ss_pairs),
        "num_sentence_sentence_edges": len(ss_pairs),
        "num_sentence_sentence_matrix_nnz": len(ss_pairs) * 2,
        "num_ee_sim_edges": len(synonym_pairs),
        "num_entity_synonym_edges": len(synonym_pairs),
        "num_entity_entity_synonym_matrix_nnz": len(synonym_pairs) * 2,
    }
    for key, actual in expected_counts.items():
        if manifest.get(key) != actual:
            errors.append(f"manifest {key} mismatch: manifest={manifest.get(key)} actual={actual}")

    for key in ["num_paragraphs", "num_sentences", "num_entities"]:
        if expected_counts[key] <= 0:
            errors.append(f"{key} must be > 0")

    if manifest.get("graph_storage") != "clean_tri_graph_tables":
        errors.append(f"manifest graph_storage must be clean_tri_graph_tables, got {manifest.get('graph_storage')}")
    if manifest.get("graph_directed") not in {False, 0, None}:
        errors.append("Harness-G clean graph should be undirected")

    if manifest.get("build_embeddings"):
        embedding_paths = {
            "passage": graph_dir / "passage_embeddings.npy",
            "sentence": graph_dir / "sentence_embeddings.npy",
            "entity": graph_dir / "entity_embeddings.npy",
            "config": graph_dir / "embedding_config.json",
        }
        for key, path in embedding_paths.items():
            if not path.exists():
                errors.append(f"Missing embedding file: {path.name}")
        if all(path.exists() for path in embedding_paths.values()):
            import numpy as np

            passage_embeddings = np.load(embedding_paths["passage"], mmap_mode="r")
            sentence_embeddings = np.load(embedding_paths["sentence"], mmap_mode="r")
            entity_embeddings = np.load(embedding_paths["entity"], mmap_mode="r")
            expected_rows = {
                "passage_embeddings.npy": (passage_embeddings, len(passages)),
                "sentence_embeddings.npy": (sentence_embeddings, len(sentences)),
                "entity_embeddings.npy": (entity_embeddings, len(entities)),
            }
            for filename, (array, expected) in expected_rows.items():
                if int(array.shape[0]) != expected:
                    errors.append(f"{filename} row mismatch: {array.shape[0]} != {expected}")
                if len(array.shape) != 2:
                    errors.append(f"{filename} must be 2D, got shape {array.shape}")
            embedding_dim = manifest.get("embedding_dim")
            if embedding_dim and int(passage_embeddings.shape[1]) != int(embedding_dim):
                errors.append(f"manifest embedding_dim mismatch: {embedding_dim} != {passage_embeddings.shape[1]}")

    # Exercise the runtime loader because the loader is what the stateful env uses.
    try:
        HarnessGGraphIndex.load(graph_dir)
    except Exception as exc:
        errors.append(f"HarnessGGraphIndex.load failed: {exc}")

    return {
        "ok": not errors,
        "errors": errors,
        "summary": {
            **expected_counts,
            "graph_version": manifest.get("graph_version"),
            "graph_storage": manifest.get("graph_storage"),
            "graph_type": manifest.get("graph_type"),
            "entity_extractor": manifest.get("entity_extractor"),
            "embedding_backend": manifest.get("embedding_backend"),
            "graph_dir": str(graph_dir),
        },
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate a Harness-G clean graph directory.")
    parser.add_argument("--graph_dir", required=True)
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    result = validate_graph(args.graph_dir)
    if not result["ok"]:
        print("[Harness-G] validation failed:")
        for error in result["errors"]:
            print(f"- {error}")
        return 1

    print("[Harness-G] validation passed")
    for key, value in result["summary"].items():
        print(f"{key}: {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
