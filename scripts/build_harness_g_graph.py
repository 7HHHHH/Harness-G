#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from harness_g.corpus_loader import resolve_corpus_path
from harness_g.graph_builder import build_graph
from harness_g.utils import parse_bool


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build a Harness-G paragraph-sentence-entity graph.")
    parser.add_argument("--data_source", default="2WikiMultiHopQA")
    parser.add_argument("--corpus_path", default=None)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--max_docs", type=int, default=None)
    parser.add_argument("--use_spacy", type=parse_bool, default=False)
    parser.add_argument("--spacy_model", default="en_core_web_sm")
    parser.add_argument("--spacy_batch_size", type=int, default=256)
    parser.add_argument("--spacy_n_process", type=int, default=1)
    parser.add_argument("--spacy_gpu", type=parse_bool, default=False)
    parser.add_argument("--build_embeddings", type=parse_bool, default=False)
    parser.add_argument("--embedding_backend", default="lexical", choices=["lexical", "sentence_transformers", "bge", "auto"])
    parser.add_argument("--embedding_model_path", default=None)
    parser.add_argument("--embedding_batch_size", type=int, default=32)
    parser.add_argument("--embedding_device", default=None)
    parser.add_argument("--entity_sim_topm", type=int, default=5)
    parser.add_argument("--entity_sim_threshold", type=float, default=0.80)
    parser.add_argument("--build_sentence_edges", type=parse_bool, default=True)
    parser.add_argument("--build_entity_synonyms", type=parse_bool, default=True)
    parser.add_argument("--entity_synonym_topk", type=int, default=None)
    parser.add_argument("--entity_synonym_threshold", type=float, default=0.80)
    parser.add_argument("--entity_synonym_candidate_limit", type=int, default=256)
    parser.add_argument("--reuse_embeddings", type=parse_bool, default=True)
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    corpus_path, warnings = resolve_corpus_path(args.data_source, args.corpus_path)
    output_dir = Path(args.output_dir) if args.output_dir else Path("expr") / args.data_source / "harness_g_graph"

    manifest = build_graph(
        corpus_path=corpus_path,
        output_dir=output_dir,
        data_source=args.data_source,
        max_docs=args.max_docs,
        use_spacy=args.use_spacy,
        spacy_model=args.spacy_model,
        build_embeddings=args.build_embeddings,
        embedding_backend=args.embedding_backend,
        embedding_model_path=args.embedding_model_path,
        embedding_batch_size=args.embedding_batch_size,
        embedding_device=args.embedding_device,
        entity_sim_topm=args.entity_sim_topm,
        entity_sim_threshold=args.entity_sim_threshold,
        build_sentence_edges=args.build_sentence_edges,
        build_entity_synonyms=args.build_entity_synonyms,
        entity_synonym_topk=args.entity_synonym_topk,
        entity_synonym_threshold=args.entity_synonym_threshold,
        entity_synonym_candidate_limit=args.entity_synonym_candidate_limit,
        reuse_embeddings=args.reuse_embeddings,
        spacy_batch_size=args.spacy_batch_size,
        spacy_n_process=args.spacy_n_process,
        spacy_gpu=args.spacy_gpu,
    )

    print(f"[Harness-G] graph written to: {output_dir}")
    for warning in warnings:
        print(f"[Harness-G][WARN] {warning}")
    print(
        "[Harness-G] summary: "
        f"storage={manifest.get('graph_storage')} "
        f"passages={manifest['num_passages']} "
        f"sentences={manifest['num_sentences']} "
        f"entities={manifest['num_entities']} "
        f"mentions={manifest['num_mentions']} "
        f"pe_edges={manifest['num_pe_edges']} "
        f"se_edges={manifest['num_se_edges']} "
        f"ss_edges={manifest.get('num_sentence_sentence_edges')} "
        f"syn_edges={manifest.get('num_entity_synonym_edges')} "
        f"embedding_backend={manifest['embedding_backend']} "
        f"embedding_dim={manifest.get('embedding_dim')} "
        f"extractor={manifest['entity_extractor']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
