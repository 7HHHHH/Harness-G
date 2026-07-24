import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from .corpus_loader import iter_corpus_records, resolve_corpus_path
from .embeddings import (
    DEFAULT_QUERY_INSTRUCTION,
    TransformerEmbeddingModel,
    normalize_embedding_backend,
    resolve_embedding_device,
    resolve_embedding_model_path,
)
from .utils import (
    append_unique,
    canonicalize_entity,
    ensure_dir,
    extract_entities_rule_based,
    extract_entities_from_spacy_doc,
    extract_entities_spacy,
    is_junk_entity_surface,
    load_spacy_model,
    split_sentences,
    write_json,
)
from .text_utils import STOPWORDS


LEGACY_OUTPUTS = (
    "paragraph_store.jsonl",
    "sentence_store.jsonl",
    "entity_store.jsonl",
    "edges_ps.jsonl",
    "edges_se.jsonl",
    "edges_ee_sim.jsonl",
    "paragraph_to_sentences.json",
    "sentence_to_entities.json",
    "entity_to_sentences.json",
    "entity_to_paragraphs.json",
    "entity_to_similar_entities.json",
)

EMBEDDING_OUTPUTS = (
    "passage_embeddings.npy",
    "sentence_embeddings.npy",
    "entity_embeddings.npy",
    "embedding_config.json",
)


class HarnessGGraphBuilder:
    """Build a relation-free Harness-G graph in clean Tri-Graph storage."""

    def __init__(
        self,
        corpus_path: Path,
        output_dir: Path,
        data_source: str = "2WikiMultiHopQA",
        max_docs: Optional[int] = None,
        use_spacy: bool = False,
        spacy_model: str = "en_core_web_sm",
        embedding_backend: str = "lexical",
        build_embeddings: bool = False,
        embedding_model_path: Optional[str] = None,
        embedding_batch_size: int = 32,
        embedding_device: Optional[str] = None,
        entity_sim_topm: int = 5,
        entity_sim_threshold: float = 0.80,
        build_sentence_edges: bool = True,
        build_entity_synonyms: bool = True,
        entity_synonym_topk: Optional[int] = None,
        entity_synonym_threshold: Optional[float] = 0.80,
        entity_synonym_candidate_limit: int = 256,
        reuse_embeddings: bool = True,
        spacy_batch_size: int = 256,
        spacy_n_process: int = 1,
        spacy_gpu: bool = False,
    ) -> None:
        self.corpus_path = Path(corpus_path)
        self.output_dir = Path(output_dir)
        self.data_source = data_source
        self.max_docs = max_docs
        self.use_spacy = use_spacy
        self.spacy_model = spacy_model
        self.spacy_batch_size = int(spacy_batch_size)
        self.spacy_n_process = int(spacy_n_process)
        self.spacy_gpu = bool(spacy_gpu)
        self.nlp = load_spacy_model(spacy_model, use_gpu=spacy_gpu) if use_spacy else None
        self.entity_extractor = f"spacy:{spacy_model}" if self.nlp is not None else "rule_based"
        self.embedding_backend = normalize_embedding_backend(embedding_backend, build_embeddings)
        self.build_embeddings = bool(build_embeddings or self.embedding_backend == "bge_transformers")
        self.embedding_model_path = resolve_embedding_model_path(embedding_model_path) if self.build_embeddings else None
        self.embedding_batch_size = int(embedding_batch_size)
        self.embedding_device = resolve_embedding_device(embedding_device) if self.build_embeddings else None
        self.embedding_dim: Optional[int] = None
        self.build_sentence_edges = bool(build_sentence_edges)
        self.build_entity_synonyms = bool(build_entity_synonyms)
        self.entity_synonym_topk = int(entity_synonym_topk if entity_synonym_topk is not None else entity_sim_topm)
        self.entity_synonym_threshold = float(entity_synonym_threshold if entity_synonym_threshold is not None else entity_sim_threshold)
        self.entity_synonym_candidate_limit = int(entity_synonym_candidate_limit)
        self.reuse_embeddings = bool(reuse_embeddings)

        self.paragraphs: List[dict] = []
        self.sentences: List[dict] = []
        self.entities: Dict[str, dict] = {}
        self.entity_key_to_eid: Dict[str, str] = {}
        self.mentions: List[dict] = []
        self.sentence_sentence_edges: List[dict] = []
        self.entity_synonym_edges: List[dict] = []
        self.entity_synonym_method = "disabled"
        self._mention_keys = set()
        self.build_warnings: List[str] = []
        self.corpus_source_type = ""

    def build(self) -> dict:
        ensure_dir(self.output_dir)

        records_iter, warnings, source_type = iter_corpus_records(self.corpus_path, self.max_docs)
        self.build_warnings = warnings
        self.corpus_source_type = source_type

        paragraph_idx = 0
        for record in records_iter:
            pid = f"p_{paragraph_idx:06d}"
            sentence_texts = split_sentences(record["text"])
            if not sentence_texts:
                continue

            sent_ids: List[str] = []
            paragraph = {
                "pid": pid,
                "doc_id": record["doc_id"],
                "title": record["title"],
                "text": record["text"],
                "sent_ids": sent_ids,
            }
            self.paragraphs.append(paragraph)
            paragraph_idx += 1

            for sentence_text in sentence_texts:
                sid = f"s_{len(self.sentences):06d}"
                entity_ids = [] if self.nlp is not None else self._extract_and_register_entities(sentence_text, sid, pid)
                self.sentences.append(
                    {
                        "sid": sid,
                        "pid": pid,
                        "doc_id": record["doc_id"],
                        "title": record["title"],
                        "text": sentence_text,
                        "entity_ids": entity_ids,
                    }
                )
                sent_ids.append(sid)

        if self.nlp is not None:
            self._extract_spacy_entities_batched()
        self._register_title_anchors()
        if not self.entities:
            self.build_warnings.append("No entities extracted from corpus.")
        self._write_outputs()
        return self.manifest()

    def _extract_spacy_entities_batched(self) -> None:
        texts = [sentence["text"] for sentence in self.sentences]
        pipe_kwargs = {
            "batch_size": max(self.spacy_batch_size, 1),
            "n_process": max(self.spacy_n_process, 1),
        }
        for sentence, doc in zip(self.sentences, self.nlp.pipe(texts, **pipe_kwargs)):
            sentence["entity_ids"] = self._register_entity_mentions(
                extract_entities_from_spacy_doc(doc),
                sentence["sid"],
                sentence["pid"],
            )

    def _extract_and_register_entities(self, sentence_text: str, sid: str, pid: str) -> List[str]:
        mentions = extract_entities_spacy(sentence_text, self.nlp) if self.nlp is not None else extract_entities_rule_based(sentence_text)
        return self._register_entity_mentions(mentions, sid, pid)

    def _register_entity_mentions(self, mentions, sid: str, pid: str) -> List[str]:
        entity_ids: List[str] = []
        for surface, label in mentions:
            if is_junk_entity_surface(surface):
                continue
            canonical = canonicalize_entity(surface)
            if not canonical:
                continue
            entity_key = self._entity_identity_key(canonical)
            eid = self.entity_key_to_eid.get(entity_key)
            if eid is None:
                eid = f"e_{len(self.entity_key_to_eid):06d}"
                self.entity_key_to_eid[entity_key] = eid
                self.entities[eid] = {
                    "eid": eid,
                    "canonical": canonical,
                    "label": label,
                    "labels": [label],
                    "surface_forms": [],
                }

            entity = self.entities[eid]
            append_unique(entity.setdefault("labels", []), label)
            entity["label"] = self._preferred_entity_label(entity.get("labels", []))
            append_unique(entity["surface_forms"], surface)
            append_unique(entity_ids, eid)

            mention_key = (sid, eid, surface)
            if mention_key not in self._mention_keys:
                self._mention_keys.add(mention_key)
                self.mentions.append(
                    {
                        "mid": f"m_{len(self.mentions):06d}",
                        "pid": pid,
                        "sid": sid,
                        "eid": eid,
                        "surface": surface,
                        "canonical": canonical,
                        "label": label,
                    }
                )

        return entity_ids

    def _register_title_anchors(self) -> None:
        """Register each document title as a canonical entity anchor, attached to
        the paragraph's first sentence. With per-document passages the title is the
        natural canonical entity; it merges with an intact body mention via the
        shared identity key (tokenized canonical), so no special-casing is needed.
        """
        if not self.paragraphs:
            return
        sid_to_sentence = {sentence["sid"]: sentence for sentence in self.sentences}
        for paragraph in self.paragraphs:
            title = (paragraph.get("title") or "").strip()
            sent_ids = paragraph.get("sent_ids") or []
            if not title or not sent_ids:
                continue
            first_sid = sent_ids[0]
            eids = self._register_entity_mentions([(title, "ENTITY")], first_sid, paragraph["pid"])
            sentence = sid_to_sentence.get(first_sid)
            if sentence is not None:
                for eid in eids:
                    append_unique(sentence.setdefault("entity_ids", []), eid)
    def paragraph_to_sentences(self) -> Dict[str, List[str]]:
        return {paragraph["pid"]: list(paragraph["sent_ids"]) for paragraph in self.paragraphs}

    def sentence_to_entities(self) -> Dict[str, List[str]]:
        return {sentence["sid"]: list(sentence.get("entity_ids", [])) for sentence in self.sentences}

    def entity_to_sentences(self) -> Dict[str, List[str]]:
        mapping: Dict[str, List[str]] = {eid: [] for eid in self.entities}
        for mention in self.mentions:
            append_unique(mapping[mention["eid"]], mention["sid"])
        return mapping

    def entity_to_paragraphs(self) -> Dict[str, List[str]]:
        mapping: Dict[str, List[str]] = {eid: [] for eid in self.entities}
        for mention in self.mentions:
            append_unique(mapping[mention["eid"]], mention["pid"])
        return mapping

    def _edge_counts(self) -> Tuple[int, int, int]:
        ps_edges = len(self.sentences)
        pe_edges = len({(mention["pid"], mention["eid"]) for mention in self.mentions})
        se_edges = len({(mention["sid"], mention["eid"]) for mention in self.mentions})
        return ps_edges, pe_edges, se_edges

    def lexical_index(self) -> dict:
        return {
            "backend": "lexical",
            "stopwords": True,
            "storage_format": "clean_tri_graph_tables",
            "num_paragraphs": len(self.paragraphs),
            "num_sentences": len(self.sentences),
            "num_entities": len(self.entities),
        }

    def id_maps(self) -> dict:
        pids = [paragraph["pid"] for paragraph in self.paragraphs]
        sids = [sentence["sid"] for sentence in self.sentences]
        eids = list(self.entities.keys())
        return {
            "storage_format": "clean_tri_graph_tables",
            "idx_to_pid": pids,
            "idx_to_sid": sids,
            "idx_to_eid": eids,
            "pid_to_idx": {pid: idx for idx, pid in enumerate(pids)},
            "sid_to_idx": {sid: idx for idx, sid in enumerate(sids)},
            "eid_to_idx": {eid: idx for idx, eid in enumerate(eids)},
        }

    def manifest(self) -> dict:
        ps_edges, pe_edges, se_edges = self._edge_counts()
        ss_edges = len(self.sentence_sentence_edges)
        synonym_edges = len(self.entity_synonym_edges)
        return {
            "graph_version": "harness_g",
            "graph_storage": "clean_tri_graph_tables",
            "graph_type": "passage_sentence_entity_with_sentence_adjacency_and_entity_synonyms",
            "relation_extraction": False,
            "llm_graph_construction": False,
            "hyperedges": False,
            "sentence_adjacency_edges": bool(self.build_sentence_edges),
            "entity_similarity_edges": bool(self.build_entity_synonyms),
            "entity_synonym_edges": bool(self.build_entity_synonyms),
            "graph_directed": False,
            "corpus_path": str(self.corpus_path),
            "corpus_source_type": self.corpus_source_type,
            "data_source": self.data_source,
            "num_paragraphs": len(self.paragraphs),
            "num_passages": len(self.paragraphs),
            "num_sentences": len(self.sentences),
            "num_entities": len(self.entities),
            "num_mentions": len(self.mentions),
            "num_ps_edges": ps_edges,
            "num_pe_edges": pe_edges,
            "num_se_edges": se_edges,
            "num_ss_edges": ss_edges,
            "num_sentence_sentence_edges": ss_edges,
            "num_sentence_sentence_matrix_nnz": ss_edges * 2,
            "num_ee_sim_edges": synonym_edges,
            "num_entity_synonym_edges": synonym_edges,
            "num_entity_entity_synonym_matrix_nnz": synonym_edges * 2,
            "entity_extractor": self.entity_extractor,
            "embedding_backend": self.embedding_backend,
            "build_embeddings": bool(self.build_embeddings),
            "embedding_model_path": self.embedding_model_path,
            "embedding_dim": self.embedding_dim,
            "embedding_batch_size": self.embedding_batch_size if self.build_embeddings else None,
            "embedding_device": self.embedding_device,
            "query_instruction": DEFAULT_QUERY_INSTRUCTION if self.build_embeddings else None,
            "reuse_embeddings": self.reuse_embeddings if self.build_embeddings else None,
            "entity_synonym_method": self.entity_synonym_method,
            "entity_synonym_topk": self.entity_synonym_topk if self.build_entity_synonyms else None,
            "entity_synonym_threshold": self.entity_synonym_threshold if self.build_entity_synonyms else None,
            "entity_synonym_candidate_limit": self.entity_synonym_candidate_limit if self.build_entity_synonyms else None,
            "spacy_batch_size": self.spacy_batch_size,
            "spacy_n_process": self.spacy_n_process,
            "spacy_gpu": self.spacy_gpu,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }

    def build_report(self) -> str:
        manifest = self.manifest()
        lines = [
            "# Harness-G v2 Build Report",
            "",
            f"- data_source: {self.data_source}",
            f"- corpus_path: {self.corpus_path}",
            f"- corpus_source_type: {self.corpus_source_type}",
            f"- graph_version: {manifest['graph_version']}",
            f"- graph_storage: {manifest['graph_storage']}",
            f"- passages: {manifest['num_passages']}",
            f"- sentences: {manifest['num_sentences']}",
            f"- entities: {manifest['num_entities']}",
            f"- mentions: {manifest['num_mentions']}",
            f"- passage_entity_edges: {manifest['num_pe_edges']}",
            f"- sentence_entity_edges: {manifest['num_se_edges']}",
            f"- sentence_sentence_edges: {manifest['num_sentence_sentence_edges']}",
            f"- entity_synonym_edges: {manifest['num_entity_synonym_edges']}",
            f"- relation_extraction: {manifest['relation_extraction']}",
            f"- llm_graph_construction: {manifest['llm_graph_construction']}",
            f"- hyperedges: {manifest['hyperedges']}",
            f"- entity_extractor: {manifest['entity_extractor']}",
            f"- embedding_backend: {manifest['embedding_backend']}",
            f"- embedding_model_path: {manifest['embedding_model_path']}",
            f"- embedding_dim: {manifest['embedding_dim']}",
            "",
            "## Storage",
            "- passages.parquet",
            "- sentences.parquet",
            "- entities.parquet",
            "- mentions.parquet",
            "- passage_entity.npz",
            "- sentence_entity.npz",
            "- sentence_sentence_edges.parquet",
            "- sentence_sentence.npz",
            "- entity_synonym_edges.parquet",
            "- entity_entity_synonym.npz",
            "- passage_embeddings.npy" if self.build_embeddings else "- passage_embeddings.npy: disabled",
            "- sentence_embeddings.npy" if self.build_embeddings else "- sentence_embeddings.npy: disabled",
            "- entity_embeddings.npy" if self.build_embeddings else "- entity_embeddings.npy: disabled",
            "- id_maps.json",
            "",
            "## Warnings",
        ]
        if self.build_warnings:
            lines.extend(f"- {warning}" for warning in self.build_warnings[:100])
        else:
            lines.append("- none")
        lines.append("")
        return "\n".join(lines)

    def _write_outputs(self) -> None:
        self._remove_legacy_outputs()
        self._write_parquet(
            self.output_dir / "passages.parquet",
            self.paragraphs,
            ["pid", "doc_id", "title", "text", "sent_ids"],
        )
        sentence_rows = [
            {key: sentence[key] for key in ("sid", "pid", "doc_id", "title", "text")}
            for sentence in self.sentences
        ]
        self._write_parquet(
            self.output_dir / "sentences.parquet",
            sentence_rows,
            ["sid", "pid", "doc_id", "title", "text"],
        )
        self._write_parquet(
            self.output_dir / "entities.parquet",
            self.entities.values(),
            ["eid", "canonical", "label", "labels", "surface_forms"],
        )
        self._write_parquet(
            self.output_dir / "mentions.parquet",
            self.mentions,
            ["mid", "pid", "sid", "eid", "surface", "canonical", "label"],
        )
        id_maps = self.id_maps()
        self.sentence_sentence_edges = self._build_sentence_sentence_edges() if self.build_sentence_edges else []
        self._write_core_sparse_matrices(id_maps)
        self._write_sentence_sentence_outputs(id_maps)
        if self.build_embeddings:
            self._write_embeddings()
        else:
            self._remove_embedding_outputs()
        self.entity_synonym_edges = self._build_entity_synonym_edges(id_maps) if self.build_entity_synonyms else []
        self._write_entity_synonym_outputs(id_maps)
        write_json(self.output_dir / "id_maps.json", id_maps)
        write_json(self.output_dir / "lexical_index.json", self.lexical_index())
        write_json(self.output_dir / "graph_manifest.json", self.manifest())
        (self.output_dir / "build_report.md").write_text(self.build_report(), encoding="utf-8")

    def _remove_legacy_outputs(self) -> None:
        for filename in LEGACY_OUTPUTS:
            path = self.output_dir / filename
            if path.exists():
                path.unlink()

    def _remove_embedding_outputs(self) -> None:
        for filename in EMBEDDING_OUTPUTS:
            path = self.output_dir / filename
            if path.exists():
                path.unlink()

    def _write_parquet(self, path: Path, rows, columns: List[str]) -> None:
        try:
            import pandas as pd
        except Exception as exc:
            raise RuntimeError("pandas/pyarrow are required for clean Harness-G graph storage") from exc
        pd.DataFrame(list(rows), columns=columns).to_parquet(path, index=False)

    def _write_core_sparse_matrices(self, id_maps: dict) -> None:
        pid_to_idx = id_maps["pid_to_idx"]
        sid_to_idx = id_maps["sid_to_idx"]
        eid_to_idx = id_maps["eid_to_idx"]

        pe_counts = Counter((mention["pid"], mention["eid"]) for mention in self.mentions)
        pid_totals = Counter(mention["pid"] for mention in self.mentions)
        pe_rows, pe_cols, pe_data = [], [], []
        for (pid, eid), count in sorted(pe_counts.items()):
            pe_rows.append(pid_to_idx[pid])
            pe_cols.append(eid_to_idx[eid])
            pe_data.append(float(count) / max(float(pid_totals[pid]), 1.0))
        self._write_npz_matrix(
            self.output_dir / "passage_entity.npz",
            pe_rows,
            pe_cols,
            pe_data,
            (len(pid_to_idx), len(eid_to_idx)),
        )

        se_pairs = sorted({(mention["sid"], mention["eid"]) for mention in self.mentions})
        se_rows = [sid_to_idx[sid] for sid, _ in se_pairs]
        se_cols = [eid_to_idx[eid] for _, eid in se_pairs]
        self._write_npz_matrix(
            self.output_dir / "sentence_entity.npz",
            se_rows,
            se_cols,
            [1.0] * len(se_pairs),
            (len(sid_to_idx), len(eid_to_idx)),
        )

    def _build_sentence_sentence_edges(self) -> List[dict]:
        edges = []
        seen = set()
        for paragraph in self.paragraphs:
            sent_ids = [sid for sid in paragraph.get("sent_ids", []) if sid]
            for sid1, sid2 in zip(sent_ids, sent_ids[1:]):
                if sid1 == sid2:
                    continue
                key = tuple(sorted((sid1, sid2)))
                if key in seen:
                    continue
                seen.add(key)
                edges.append(
                    {
                        "sid1": sid1,
                        "sid2": sid2,
                        "relation": "adjacent",
                        "weight": 1.0,
                        "undirected": True,
                    }
                )
        return edges

    def _write_sentence_sentence_outputs(self, id_maps: dict) -> None:
        sid_to_idx = id_maps["sid_to_idx"]
        self._write_parquet(
            self.output_dir / "sentence_sentence_edges.parquet",
            self.sentence_sentence_edges,
            ["sid1", "sid2", "relation", "weight", "undirected"],
        )

        rows, cols, data = [], [], []
        for edge in self.sentence_sentence_edges:
            left = sid_to_idx[edge["sid1"]]
            right = sid_to_idx[edge["sid2"]]
            rows.extend([left, right])
            cols.extend([right, left])
            data.extend([float(edge.get("weight", 1.0)), float(edge.get("weight", 1.0))])
        self._write_npz_matrix(
            self.output_dir / "sentence_sentence.npz",
            rows,
            cols,
            data,
            (len(sid_to_idx), len(sid_to_idx)),
        )

    def _build_entity_synonym_edges(self, id_maps: dict) -> List[dict]:
        if len(self.entities) < 2:
            self.entity_synonym_method = "empty"
            return []

        entity_embeddings = self._load_entity_embeddings_for_synonyms(id_maps)
        if entity_embeddings is not None:
            self.entity_synonym_method = "hipporag_style_blocked_embedding_knn"
        else:
            self.entity_synonym_method = "blocked_lexical"

        features = {eid: self._entity_synonym_features(entity) for eid, entity in self.entities.items()}
        exact_index: Dict[str, List[str]] = {}
        acronym_index: Dict[str, List[str]] = {}
        token_index: Dict[str, List[str]] = {}
        for eid, feature in features.items():
            for key in feature["exact_keys"]:
                exact_index.setdefault(key, []).append(eid)
            for key in feature["explicit_acronyms"]:
                acronym_index.setdefault(key, []).append(eid)
            for key in feature["expansion_acronyms"]:
                acronym_index.setdefault(key, []).append(eid)
            for token in feature["tokens"]:
                token_index.setdefault(token, []).append(eid)

        eid_to_idx = id_maps["eid_to_idx"]
        edge_by_pair: Dict[Tuple[str, str], dict] = {}
        for eid, entity in self.entities.items():
            feature = features[eid]
            if self._short_entity(feature["canonical"]):
                continue
            if self._low_quality_synonym_feature(feature):
                continue
            candidates = self._entity_synonym_candidates(eid, feature, exact_index, acronym_index, token_index)
            if not candidates:
                continue

            scored = []
            candidate_eids = [candidate_eid for candidate_eid, _ in candidates]
            embedding_scores = self._score_entity_embedding_candidates(eid, candidate_eids, entity_embeddings, eid_to_idx)
            for candidate_eid, reason in candidates:
                other = self.entities.get(candidate_eid)
                other_feature = features.get(candidate_eid)
                if other is None or other_feature is None or self._short_entity(other_feature["canonical"]):
                    continue
                if self._low_quality_synonym_feature(other_feature):
                    continue
                if feature["canonical"] == other_feature["canonical"]:
                    continue
                if not self._entity_labels_compatible(entity.get("label", ""), other.get("label", ""), reason):
                    continue
                lexical_overlap = self._token_jaccard(feature["tokens"], other_feature["tokens"])
                if reason == "acronym" and not self._valid_acronym_pair(feature, other_feature):
                    continue
                if reason == "token" and lexical_overlap < 0.80:
                    continue

                score = embedding_scores.get(candidate_eid)
                if score is None:
                    score = lexical_overlap
                if score < self.entity_synonym_threshold:
                    continue
                score = max(0.0, min(float(score), 1.0))
                scored.append((candidate_eid, score, reason, lexical_overlap))

            scored.sort(key=lambda item: (-item[1], item[0]))
            for candidate_eid, score, reason, lexical_overlap in scored[: self.entity_synonym_topk]:
                left, right = sorted((eid, candidate_eid))
                if left == right:
                    continue
                pair = (left, right)
                existing = edge_by_pair.get(pair)
                if existing is not None and existing["score"] >= score:
                    continue
                edge_by_pair[pair] = {
                    "eid1": left,
                    "eid2": right,
                    "score": round(score, 6),
                    "method": self.entity_synonym_method,
                    "reason": reason,
                    "lexical_overlap": round(float(lexical_overlap), 6),
                    "canonical1": self.entities[left].get("canonical", ""),
                    "canonical2": self.entities[right].get("canonical", ""),
                    "undirected": True,
                }

        return sorted(edge_by_pair.values(), key=lambda row: (row["eid1"], row["eid2"]))

    def _write_entity_synonym_outputs(self, id_maps: dict) -> None:
        eid_to_idx = id_maps["eid_to_idx"]
        self._write_parquet(
            self.output_dir / "entity_synonym_edges.parquet",
            self.entity_synonym_edges,
            ["eid1", "eid2", "score", "method", "reason", "lexical_overlap", "canonical1", "canonical2", "undirected"],
        )

        rows, cols, data = [], [], []
        for edge in self.entity_synonym_edges:
            left = eid_to_idx[edge["eid1"]]
            right = eid_to_idx[edge["eid2"]]
            rows.extend([left, right])
            cols.extend([right, left])
            data.extend([float(edge.get("score", 1.0)), float(edge.get("score", 1.0))])
        self._write_npz_matrix(
            self.output_dir / "entity_entity_synonym.npz",
            rows,
            cols,
            data,
            (len(eid_to_idx), len(eid_to_idx)),
        )

    @staticmethod
    def _write_npz_matrix(path: Path, rows: List[int], cols: List[int], data: List[float], shape: Tuple[int, int]) -> None:
        import numpy as np

        np.savez_compressed(
            path,
            row=np.asarray(rows, dtype="int64"),
            col=np.asarray(cols, dtype="int64"),
            data=np.asarray(data, dtype="float32"),
            shape=np.asarray(shape, dtype="int64"),
        )

    def _write_embeddings(self) -> None:
        import numpy as np

        reusable = {
            "passage": self._try_reuse_embedding("passage", len(self.paragraphs)),
            "sentence": self._try_reuse_embedding("sentence", len(self.sentences)),
            "entity": self._try_reuse_embedding("entity", len(self.entities)),
        }
        reusable_dims = {
            int(array.shape[1])
            for array in reusable.values()
            if array is not None and len(array.shape) == 2
        }
        if all(array is not None for array in reusable.values()) and len(reusable_dims) == 1:
            self.embedding_dim = next(iter(reusable_dims))
            self._write_embedding_config()
            return

        encoder = TransformerEmbeddingModel(self.embedding_model_path, device=self.embedding_device)
        self.embedding_dim = encoder.embedding_dim

        if reusable["passage"] is None or int(reusable["passage"].shape[1]) != self.embedding_dim:
            passage_texts = [
                f"{paragraph.get('title', '')}\n{paragraph.get('text', '')}".strip()
                for paragraph in self.paragraphs
            ]
            np.save(
                self.output_dir / "passage_embeddings.npy",
                encoder.encode(passage_texts, batch_size=self.embedding_batch_size, show_progress=True),
            )

        if reusable["sentence"] is None or int(reusable["sentence"].shape[1]) != self.embedding_dim:
            sentence_texts = [
                f"{sentence.get('title', '')}\n{sentence.get('text', '')}".strip()
                for sentence in self.sentences
            ]
            np.save(
                self.output_dir / "sentence_embeddings.npy",
                encoder.encode(sentence_texts, batch_size=self.embedding_batch_size, show_progress=True),
            )

        if reusable["entity"] is None or int(reusable["entity"].shape[1]) != self.embedding_dim:
            entity_texts = [
                self._entity_embedding_text(entity)
                for entity in self.entities.values()
            ]
            np.save(
                self.output_dir / "entity_embeddings.npy",
                encoder.encode(entity_texts, batch_size=self.embedding_batch_size, show_progress=True),
            )

        self._write_embedding_config()

    def _try_reuse_embedding(self, kind: str, expected_rows: int):
        if not self.reuse_embeddings:
            return None
        path = self.output_dir / f"{kind}_embeddings.npy"
        config_path = self.output_dir / "embedding_config.json"
        if not path.exists() or not config_path.exists():
            return None
        try:
            import numpy as np

            with config_path.open("r", encoding="utf-8") as f:
                import json

                config = json.load(f)
            if config.get("embedding_model_path") != self.embedding_model_path:
                return None
            array = np.load(path, mmap_mode="r")
            if len(array.shape) != 2 or int(array.shape[0]) != expected_rows:
                return None
            return array
        except Exception:
            return None

    def _write_embedding_config(self) -> None:
        write_json(
            self.output_dir / "embedding_config.json",
            {
                "embedding_backend": self.embedding_backend,
                "embedding_model_path": self.embedding_model_path,
                "embedding_dim": self.embedding_dim,
                "embedding_device": self.embedding_device,
                "embedding_batch_size": self.embedding_batch_size,
                "normalized": True,
                "pooling": "cls",
                "query_instruction": DEFAULT_QUERY_INSTRUCTION,
                "passage_text": "title + newline + text",
                "sentence_text": "title + newline + text",
                "entity_text": "canonical + surface forms",
            },
        )

    def _load_entity_embeddings_for_synonyms(self, id_maps: dict):
        if not self.build_embeddings:
            return None
        path = self.output_dir / "entity_embeddings.npy"
        if not path.exists():
            return None
        try:
            import numpy as np

            array = np.load(path, mmap_mode="r")
            if len(array.shape) != 2 or int(array.shape[0]) != len(id_maps["eid_to_idx"]):
                return None
            return array
        except Exception:
            return None

    @staticmethod
    def _entity_embedding_text(entity: dict) -> str:
        forms = [str(item) for item in entity.get("surface_forms", []) if str(item)]
        pieces = [str(entity.get("canonical", ""))]
        pieces.extend(forms[:4])
        return " | ".join(piece for piece in pieces if piece)

    @staticmethod
    def _entity_synonym_features(entity: dict) -> dict:
        texts = [str(entity.get("canonical", "") or "")]
        texts.extend(str(item) for item in entity.get("surface_forms", []) if str(item))
        canonical_raw_tokens = re.findall(r"[a-z0-9]+", str(entity.get("canonical", "") or "").lower())
        exact_keys: Set[str] = set()
        tokens: Set[str] = set()
        explicit_acronyms: Set[str] = set()
        expansion_acronyms: Set[str] = set()
        for text in texts:
            normalized = " ".join(re.findall(r"[a-z0-9]+", text.lower()))
            compact = "".join(re.findall(r"[a-z0-9]+", text.lower()))
            if normalized:
                exact_keys.add(normalized)
            if compact:
                exact_keys.add(compact)
            if compact and compact.isalpha() and len(compact) >= 3 and re.fullmatch(r"[A-Z.\- ]{3,}", str(text).strip()):
                explicit_acronyms.add(compact)
            raw_tokens = [token for token in re.findall(r"[a-z0-9]+", text.lower()) if token not in STOPWORDS]
            tokens.update(raw_tokens)
            if len(raw_tokens) >= 2:
                acronym = "".join(token[0] for token in raw_tokens if token)
                if len(acronym) >= 2:
                    expansion_acronyms.add(acronym)
        return {
            "canonical": str(entity.get("canonical", "") or ""),
            "canonical_raw_tokens": canonical_raw_tokens,
            "exact_keys": exact_keys,
            "tokens": tokens,
            "explicit_acronyms": explicit_acronyms,
            "expansion_acronyms": expansion_acronyms,
        }

    def _entity_synonym_candidates(
        self,
        eid: str,
        feature: dict,
        exact_index: Dict[str, List[str]],
        acronym_index: Dict[str, List[str]],
        token_index: Dict[str, List[str]],
    ) -> List[Tuple[str, str]]:
        candidates: Dict[str, str] = {}
        for key in feature["exact_keys"]:
            for other in exact_index.get(key, []):
                if other != eid:
                    candidates[other] = "exact"
        for key in feature["explicit_acronyms"] | feature["expansion_acronyms"]:
            for other in acronym_index.get(key, []):
                if other != eid and candidates.get(other) != "exact":
                    candidates[other] = "acronym"

        token_votes = Counter()
        for token in feature["tokens"]:
            posting = token_index.get(token, [])
            if len(posting) > max(self.entity_synonym_candidate_limit * 8, 2048):
                continue
            for other in posting:
                if other != eid and other not in candidates:
                    token_votes[other] += 1
        for other, _ in token_votes.most_common(max(self.entity_synonym_candidate_limit - len(candidates), 0)):
            candidates[other] = "token"

        priority = {"exact": 0, "acronym": 1, "token": 2}
        ordered = sorted(candidates.items(), key=lambda item: (priority.get(item[1], 9), item[0]))
        return ordered[: self.entity_synonym_candidate_limit]

    @staticmethod
    def _score_entity_embedding_candidates(eid: str, candidate_eids: List[str], entity_embeddings, eid_to_idx: dict) -> Dict[str, float]:
        if entity_embeddings is None or eid not in eid_to_idx:
            return {}
        candidate_indices = [int(eid_to_idx[candidate]) for candidate in candidate_eids if candidate in eid_to_idx]
        if not candidate_indices:
            return {}
        import numpy as np

        target = entity_embeddings[int(eid_to_idx[eid])]
        matrix = entity_embeddings[np.asarray(candidate_indices, dtype="int64")]
        scores = np.asarray(matrix @ target, dtype="float32")
        scored = {}
        pos = 0
        for candidate in candidate_eids:
            if candidate not in eid_to_idx:
                continue
            scored[candidate] = float(scores[pos])
            pos += 1
        return scored

    @staticmethod
    def _short_entity(canonical: str) -> bool:
        return len(re.sub(r"[^a-z0-9]", "", str(canonical).lower())) <= 2

    @staticmethod
    def _low_quality_synonym_feature(feature: dict) -> bool:
        raw_tokens = list(feature.get("canonical_raw_tokens") or [])
        if not raw_tokens:
            return True
        if raw_tokens[-1] in STOPWORDS:
            return True
        if len(raw_tokens) <= 4 and len(set(raw_tokens)) < len(raw_tokens):
            return True
        content_tokens = [token for token in raw_tokens if token not in STOPWORDS]
        return not content_tokens

    @staticmethod
    def _entity_identity_key(canonical: str) -> str:
        tokens = re.findall(r"[a-z0-9]+", str(canonical).lower())
        return " ".join(tokens) if tokens else str(canonical).strip().lower()

    @staticmethod
    def _token_jaccard(left: Set[str], right: Set[str]) -> float:
        if not left or not right:
            return 0.0
        overlap = len(left & right)
        if overlap == 0:
            return 0.0
        return float(overlap / len(left | right))

    @staticmethod
    def _entity_labels_compatible(left: str, right: str, reason: str) -> bool:
        if reason == "acronym":
            return True
        left = str(left or "")
        right = str(right or "")
        return not left or not right or left == right or left == "ENTITY" or right == "ENTITY"

    @staticmethod
    def _valid_acronym_pair(left: dict, right: dict) -> bool:
        return bool(
            left["explicit_acronyms"].intersection(right["expansion_acronyms"])
            or right["explicit_acronyms"].intersection(left["expansion_acronyms"])
        )

    @staticmethod
    def _preferred_entity_label(labels: List[str]) -> str:
        priority = {
            "PERSON": 0,
            "ORG": 1,
            "GPE": 2,
            "LOC": 3,
            "FAC": 4,
            "WORK_OF_ART": 5,
            "EVENT": 6,
            "NORP": 7,
            "DATE": 8,
            "ENTITY": 9,
        }
        cleaned = [str(label) for label in labels if str(label)]
        if not cleaned:
            return "ENTITY"
        return sorted(cleaned, key=lambda label: (priority.get(label, 99), label))[0]


def build_graph(
    corpus_path: Optional[Path] = None,
    output_dir: Optional[Path] = None,
    data_source: str = "2WikiMultiHopQA",
    max_docs: Optional[int] = None,
    use_spacy: bool = False,
    spacy_model: str = "en_core_web_sm",
    build_embeddings: bool = False,
    embedding_backend: str = "lexical",
    embedding_model_path: Optional[str] = None,
    embedding_batch_size: int = 32,
    embedding_device: Optional[str] = None,
    entity_sim_topm: int = 5,
    entity_sim_threshold: float = 0.80,
    build_sentence_edges: bool = True,
    build_entity_synonyms: bool = True,
    entity_synonym_topk: Optional[int] = None,
    entity_synonym_threshold: Optional[float] = 0.80,
    entity_synonym_candidate_limit: int = 256,
    reuse_embeddings: bool = True,
    spacy_batch_size: int = 256,
    spacy_n_process: int = 1,
    spacy_gpu: bool = False,
) -> dict:
    resolved_corpus_path = Path(corpus_path) if corpus_path is not None else resolve_corpus_path(data_source)[0]
    resolved_output_dir = Path(output_dir) if output_dir is not None else Path("expr") / data_source / "harness_g_graph"
    builder = HarnessGGraphBuilder(
        corpus_path=resolved_corpus_path,
        output_dir=resolved_output_dir,
        data_source=data_source,
        max_docs=max_docs,
        use_spacy=use_spacy,
        spacy_model=spacy_model,
        build_embeddings=build_embeddings,
        embedding_backend=embedding_backend,
        embedding_model_path=embedding_model_path,
        embedding_batch_size=embedding_batch_size,
        embedding_device=embedding_device,
        entity_sim_topm=entity_sim_topm,
        entity_sim_threshold=entity_sim_threshold,
        build_sentence_edges=build_sentence_edges,
        build_entity_synonyms=build_entity_synonyms,
        entity_synonym_topk=entity_synonym_topk,
        entity_synonym_threshold=entity_synonym_threshold,
        entity_synonym_candidate_limit=entity_synonym_candidate_limit,
        reuse_embeddings=reuse_embeddings,
        spacy_batch_size=spacy_batch_size,
        spacy_n_process=spacy_n_process,
        spacy_gpu=spacy_gpu,
    )
    return builder.build()
