import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

from harness_g.graph_builder import HarnessGGraphBuilder


CORPUS = """{"id": "1", "title": "Ada Lovelace", "contents": "Ada Lovelace was born in London. She worked with Charles Babbage on the Analytical Engine."}
{"id": "2", "title": "Charles Babbage", "contents": "Charles Babbage designed the Analytical Engine. He was born in London."}
"""


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
]


def test_graph_build_and_validate(tmp_path):
    corpus_path = tmp_path / "corpus.jsonl"
    output_dir = tmp_path / "harness_g_graph"
    corpus_path.write_text(CORPUS, encoding="utf-8")

    subprocess.run(
        [
            sys.executable,
            "scripts/build_harness_g_graph.py",
            "--corpus_path",
            str(corpus_path),
            "--output_dir",
            str(output_dir),
            "--use_spacy",
            "false",
        ],
        check=True,
    )

    for filename in REQUIRED_FILES:
        assert (output_dir / filename).exists(), filename

    manifest = json.loads((output_dir / "graph_manifest.json").read_text(encoding="utf-8"))
    assert manifest["graph_storage"] == "clean_tri_graph_tables"
    assert manifest["num_paragraphs"] > 0
    assert manifest["num_sentences"] > 0
    assert manifest["num_entities"] > 0
    assert manifest["num_mentions"] > 0
    assert manifest["graph_directed"] is False
    assert manifest["num_sentence_sentence_edges"] > 0
    assert manifest["num_entity_synonym_edges"] >= 0

    entities = pd.read_parquet(output_dir / "entities.parquet").to_dict("records")
    canonicals = {entity["canonical"] for entity in entities}
    assert {"ada lovelace", "charles babbage"} & canonicals

    subprocess.run(
        [
            sys.executable,
            "scripts/validate_harness_g_graph.py",
            "--graph_dir",
            str(output_dir),
        ],
        check=True,
    )


def test_entity_identity_is_case_insensitive(tmp_path):
    builder = HarnessGGraphBuilder(
        corpus_path=tmp_path / "corpus.jsonl",
        output_dir=tmp_path / "graph",
    )
    entity_ids = builder._register_entity_mentions(
        [
            ("Ada Lovelace", "PERSON"),
            ("ada lovelace", "ORG"),
            ("ADA LOVELACE", "PERSON"),
            ("Ada-Lovelace", "PERSON"),
        ],
        "s_000000",
        "p_000000",
    )

    assert entity_ids == ["e_000000"]
    assert len(builder.entities) == 1
    entity = builder.entities["e_000000"]
    assert entity["canonical"] == "ada lovelace"
    assert set(entity["labels"]) == {"PERSON", "ORG"}
    assert {"Ada Lovelace", "ada lovelace", "ADA LOVELACE", "Ada-Lovelace"} <= set(entity["surface_forms"])
