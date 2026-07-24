import json
import math
import os
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple, Union

import numpy as np

from .embeddings import DEFAULT_QUERY_INSTRUCTION, TransformerEmbeddingModel, resolve_embedding_device
from .text_utils import cosine_counter
from .utils import append_unique, read_json, read_jsonl, tokenize_for_search


class HarnessGGraphIndex:
    """In-memory lexical index over the clean Harness-G Tri-Graph."""

    def __init__(self) -> None:
        self.graph_dir: Optional[Path] = None
        self.manifest: dict = {}
        self.paragraphs: Dict[str, dict] = {}
        self.sentences: Dict[str, dict] = {}
        self.entities: Dict[str, dict] = {}
        self.mentions: List[dict] = []
        self.paragraph_to_sentences: Dict[str, List[str]] = {}
        self.sentence_to_entities: Dict[str, List[str]] = {}
        self.entity_to_sentences: Dict[str, List[str]] = {}
        self.entity_to_paragraphs: Dict[str, List[str]] = {}
        self.entity_to_similar_entities: Dict[str, List[dict]] = {}
        self.sentence_to_neighbor_sentences: Dict[str, List[str]] = {}
        self.entity_to_synonym_entities: Dict[str, List[dict]] = {}
        self.id_maps: dict = {}
        self.embedding_config: dict = {}
        self.passage_embeddings = None
        self.sentence_embeddings = None
        self.entity_embeddings = None
        self._embedding_encoder = None
        self._query_embedding_cache: Dict[str, np.ndarray] = {}
        self._dense_disabled_reason: Optional[str] = None
        self._paragraph_search_cache: Dict[Tuple[str, int], List[dict]] = {}
        self._global_sentence_search_cache: Dict[Tuple[str, int], List[dict]] = {}
        self._entity_search_cache: Dict[Tuple[str, int], List[dict]] = {}
        self._hybrid_initial_cache: Dict[Tuple[str, int, int, int, int, int, int], List[dict]] = {}
        self._local_context_cache: Dict[str, List[dict]] = {}
        self._bridge_candidate_cache: Dict[Tuple[Tuple[str, ...], str, Tuple[str, ...], int], List[dict]] = {}
        self._bridge_entity_cache: Dict[Tuple[str, str, str, str, Tuple[str, ...], int], List[dict]] = {}
        self._expand_entity_cache: Dict[Tuple[str, str, str, int], List[dict]] = {}
        self._lookup_entity_cache: Dict[Tuple[str, str, str, str, str, int], List[dict]] = {}
        self._similar_entities_cache: Dict[Tuple[str, int], List[dict]] = {}
        self._paragraph_vectors: Dict[str, Counter] = {}
        self._paragraph_title_vectors: Dict[str, Counter] = {}
        self._sentence_vectors: Dict[str, Counter] = {}
        self._entity_token_index: Dict[str, List[str]] = {}

    @classmethod
    def load(cls, graph_dir: Union[str, Path]) -> "HarnessGGraphIndex":
        index = cls()
        index._load(graph_dir)
        return index

    def _load(self, graph_dir: Union[str, Path]) -> None:
        self.graph_dir = Path(graph_dir)
        self.manifest = read_json(self.graph_dir / "graph_manifest.json")
        if (self.graph_dir / "passages.parquet").exists():
            self._load_clean_tables()
        else:
            self._load_legacy_json()
        self._load_dense_embeddings()
        self._build_lexical_vectors()

    def _load_clean_tables(self) -> None:
        passages = self._read_parquet_records(self.graph_dir / "passages.parquet")
        sentences = self._read_parquet_records(self.graph_dir / "sentences.parquet")
        entities = self._read_parquet_records(self.graph_dir / "entities.parquet")
        mentions = self._read_parquet_records(self.graph_dir / "mentions.parquet")

        self.paragraphs = {}
        for row in passages:
            row["pid"] = str(row["pid"])
            row["doc_id"] = str(row.get("doc_id", ""))
            row["title"] = str(row.get("title", "") or "")
            row["text"] = str(row.get("text", "") or "")
            row["sent_ids"] = self._as_list(row.get("sent_ids"))
            self.paragraphs[row["pid"]] = row

        self.sentences = {}
        for row in sentences:
            row["sid"] = str(row["sid"])
            row["pid"] = str(row["pid"])
            row["doc_id"] = str(row.get("doc_id", ""))
            row["title"] = str(row.get("title", "") or "")
            row["text"] = str(row.get("text", "") or "")
            row["entity_ids"] = []
            self.sentences[row["sid"]] = row

        self.entities = {}
        for row in entities:
            row["eid"] = str(row["eid"])
            row["canonical"] = str(row.get("canonical", "") or "")
            row["label"] = str(row.get("label", "") or "")
            row["labels"] = self._as_list(row.get("labels")) or ([row["label"]] if row["label"] else [])
            row["surface_forms"] = self._as_list(row.get("surface_forms"))
            row["mention_sids"] = []
            row["mention_pids"] = []
            self.entities[row["eid"]] = row

        self.mentions = []
        self.paragraph_to_sentences = {pid: [] for pid in self.paragraphs}
        for sid, sentence in self.sentences.items():
            append_unique(self.paragraph_to_sentences.setdefault(sentence["pid"], []), sid)
        for pid, paragraph in self.paragraphs.items():
            if paragraph.get("sent_ids"):
                self.paragraph_to_sentences[pid] = [sid for sid in paragraph["sent_ids"] if sid in self.sentences]
            else:
                paragraph["sent_ids"] = self.paragraph_to_sentences.get(pid, [])

        self.sentence_to_entities = {sid: [] for sid in self.sentences}
        self.entity_to_sentences = {eid: [] for eid in self.entities}
        self.entity_to_paragraphs = {eid: [] for eid in self.entities}
        for row in mentions:
            mention = {
                "mid": str(row.get("mid", f"m_{len(self.mentions):06d}")),
                "pid": str(row["pid"]),
                "sid": str(row["sid"]),
                "eid": str(row["eid"]),
                "surface": str(row.get("surface", "") or ""),
                "canonical": str(row.get("canonical", "") or ""),
                "label": str(row.get("label", "") or ""),
            }
            if mention["pid"] not in self.paragraphs or mention["sid"] not in self.sentences or mention["eid"] not in self.entities:
                continue
            self.mentions.append(mention)
            append_unique(self.sentence_to_entities[mention["sid"]], mention["eid"])
            append_unique(self.entity_to_sentences[mention["eid"]], mention["sid"])
            append_unique(self.entity_to_paragraphs[mention["eid"]], mention["pid"])

        for sid, entity_ids in self.sentence_to_entities.items():
            self.sentences[sid]["entity_ids"] = list(entity_ids)
        for eid, entity in self.entities.items():
            entity["mention_sids"] = list(self.entity_to_sentences.get(eid, []))
            entity["mention_pids"] = list(self.entity_to_paragraphs.get(eid, []))

        id_maps_path = self.graph_dir / "id_maps.json"
        self.id_maps = read_json(id_maps_path) if id_maps_path.exists() else {}
        self.entity_to_similar_entities = {}
        self._load_clean_edge_tables()

    def _load_clean_edge_tables(self) -> None:
        self.sentence_to_neighbor_sentences = {sid: [] for sid in self.sentences}
        ss_path = self.graph_dir / "sentence_sentence_edges.parquet"
        if ss_path.exists():
            for row in self._read_parquet_records(ss_path):
                sid1 = str(row.get("sid1", ""))
                sid2 = str(row.get("sid2", ""))
                if sid1 in self.sentences and sid2 in self.sentences and sid1 != sid2:
                    append_unique(self.sentence_to_neighbor_sentences.setdefault(sid1, []), sid2)
                    append_unique(self.sentence_to_neighbor_sentences.setdefault(sid2, []), sid1)

        self.entity_to_synonym_entities = {eid: [] for eid in self.entities}
        synonym_path = self.graph_dir / "entity_synonym_edges.parquet"
        if synonym_path.exists():
            for row in self._read_parquet_records(synonym_path):
                eid1 = str(row.get("eid1", ""))
                eid2 = str(row.get("eid2", ""))
                if eid1 not in self.entities or eid2 not in self.entities or eid1 == eid2:
                    continue
                score = float(row.get("score", 0.0) or 0.0)
                method = str(row.get("method", "") or "")
                reason = str(row.get("reason", "") or "")
                self.entity_to_synonym_entities.setdefault(eid1, []).append(
                    {**self.entities[eid2], "score": round(score, 6), "method": method, "reason": reason}
                )
                self.entity_to_synonym_entities.setdefault(eid2, []).append(
                    {**self.entities[eid1], "score": round(score, 6), "method": method, "reason": reason}
                )
        for eid in self.entity_to_synonym_entities:
            self.entity_to_synonym_entities[eid].sort(key=lambda item: (-float(item.get("score", 0.0)), item.get("eid", "")))

    def _load_legacy_json(self) -> None:
        self.paragraphs = {row["pid"]: row for row in read_jsonl(self.graph_dir / "paragraph_store.jsonl")}
        self.sentences = {row["sid"]: row for row in read_jsonl(self.graph_dir / "sentence_store.jsonl")}
        self.entities = {row["eid"]: row for row in read_jsonl(self.graph_dir / "entity_store.jsonl")}
        self.paragraph_to_sentences = read_json(self.graph_dir / "paragraph_to_sentences.json")
        self.sentence_to_entities = read_json(self.graph_dir / "sentence_to_entities.json")
        self.entity_to_sentences = read_json(self.graph_dir / "entity_to_sentences.json")
        self.entity_to_paragraphs = read_json(self.graph_dir / "entity_to_paragraphs.json")
        similar_path = self.graph_dir / "entity_to_similar_entities.json"
        self.entity_to_similar_entities = read_json(similar_path) if similar_path.exists() else {}
        self.sentence_to_neighbor_sentences = {sid: [] for sid in self.sentences}
        self.entity_to_synonym_entities = {}
        self.id_maps = {
            "pid_to_idx": {pid: idx for idx, pid in enumerate(self.paragraphs)},
            "sid_to_idx": {sid: idx for idx, sid in enumerate(self.sentences)},
            "eid_to_idx": {eid: idx for idx, eid in enumerate(self.entities)},
        }
        self.mentions = []
        for eid, sids in self.entity_to_sentences.items():
            for sid in sids:
                sentence = self.sentences.get(sid, {})
                self.mentions.append(
                    {
                        "mid": f"m_{len(self.mentions):06d}",
                        "pid": sentence.get("pid", ""),
                        "sid": sid,
                        "eid": eid,
                        "surface": (self.entities.get(eid, {}).get("surface_forms") or [""])[0],
                        "canonical": self.entities.get(eid, {}).get("canonical", ""),
                        "label": self.entities.get(eid, {}).get("label", ""),
                    }
                )

    def _build_lexical_vectors(self) -> None:
        self._paragraph_vectors = {
            pid: Counter(tokenize_for_search(f"{p.get('title', '')} {p.get('text', '')}"))
            for pid, p in self.paragraphs.items()
        }
        self._paragraph_title_vectors = {
            pid: Counter(tokenize_for_search(p.get("title", "")))
            for pid, p in self.paragraphs.items()
        }
        self._sentence_vectors = {
            sid: Counter(tokenize_for_search(f"{s.get('title', '')} {s.get('text', '')}"))
            for sid, s in self.sentences.items()
        }
        self._entity_token_index = {}
        for eid, entity in self.entities.items():
            for token in set(tokenize_for_search(entity.get("canonical", ""))):
                self._entity_token_index.setdefault(token, []).append(eid)

    def score(self, query: str, text: str) -> float:
        return self._score(Counter(tokenize_for_search(query)), Counter(tokenize_for_search(text)))

    def search_paragraphs(self, query: str, topk: int = 5) -> List[dict]:
        cache_key = (str(query or ""), int(topk))
        if cache_key in self._paragraph_search_cache:
            return self._clone_rows(self._paragraph_search_cache[cache_key])
        if self._dense_available("passage"):
            query_embedding = self._embed_query(query)
            if query_embedding is not None:
                scores = np.asarray(self.passage_embeddings @ query_embedding, dtype="float32")
                rows = []
                idx_to_pid = self.id_maps.get("idx_to_pid") or list(self.paragraphs.keys())
                for idx in np.argsort(-scores)[:topk]:
                    pid = idx_to_pid[int(idx)]
                    paragraph = self.paragraphs.get(pid)
                    if paragraph is not None:
                        rows.append({**paragraph, "score": round(float(scores[int(idx)]), 6)})
                self._paragraph_search_cache[cache_key] = self._clone_rows(rows)
                return rows

        query_vector = Counter(tokenize_for_search(query))
        scored = []
        for pid, paragraph in self.paragraphs.items():
            score = self._score(query_vector, self._paragraph_vectors[pid])
            title_score = self._score(query_vector, self._paragraph_title_vectors[pid])
            total_score = score + 0.25 * title_score
            scored.append({**paragraph, "score": round(float(total_score), 6)})
        scored.sort(key=lambda item: (-item["score"], item["pid"]))
        rows = scored[:topk]
        self._paragraph_search_cache[cache_key] = self._clone_rows(rows)
        return rows

    def rank_sentences(self, query: str, sids: Iterable[str], topk: int = 8) -> List[dict]:
        if self._dense_available("sentence"):
            query_embedding = self._embed_query(query)
            sid_to_idx = self.id_maps.get("sid_to_idx", {})
            candidate = []
            seen = set()
            for sid in sids:
                if sid in seen or sid not in self.sentences or sid not in sid_to_idx:
                    continue
                seen.add(sid)
                candidate.append((sid, int(sid_to_idx[sid])))
            if query_embedding is not None and candidate:
                indices = np.asarray([idx for _, idx in candidate], dtype="int64")
                scores = np.asarray(self.sentence_embeddings[indices] @ query_embedding, dtype="float32")
                order = np.argsort(-scores)[:topk]
                rows = []
                for pos in order:
                    sid, _ = candidate[int(pos)]
                    sentence = self.sentences[sid]
                    rows.append(
                        {
                            **sentence,
                            "score": round(float(scores[int(pos)]), 6),
                            "title": sentence.get("title", ""),
                            "entity_ids": list(self.sentence_to_entities.get(sid, [])),
                        }
                    )
                return rows

        query_vector = Counter(tokenize_for_search(query))
        scored = []
        seen = set()
        for sid in sids:
            if sid in seen or sid not in self.sentences:
                continue
            seen.add(sid)
            sentence = self.sentences[sid]
            score = self._score(query_vector, self._sentence_vectors[sid])
            scored.append(
                {
                    **sentence,
                    "score": round(float(score), 6),
                    "title": sentence.get("title", ""),
                    "entity_ids": list(self.sentence_to_entities.get(sid, [])),
                }
            )
        scored.sort(key=lambda item: (-item["score"], item["sid"]))
        return scored[:topk]

    def search_sentences_global(self, query: str, topk: int = 8) -> List[dict]:
        """Search all sentence embeddings, falling back to lexical ranking."""

        cache_key = (str(query or ""), int(topk))
        if cache_key in self._global_sentence_search_cache:
            return self._clone_rows(self._global_sentence_search_cache[cache_key])

        if self._dense_available("sentence"):
            query_embedding = self._embed_query(query)
            if query_embedding is not None:
                scores = np.asarray(self.sentence_embeddings @ query_embedding, dtype="float32")
                idx_to_sid = self.id_maps.get("idx_to_sid") or list(self.sentences.keys())
                rows = []
                for idx in np.argsort(-scores)[:topk]:
                    sid = idx_to_sid[int(idx)]
                    sentence = self.sentences.get(sid)
                    if sentence is None:
                        continue
                    rows.append(
                        {
                            **sentence,
                            "score": round(float(scores[int(idx)]), 6),
                            "source": "global_sentence_dense",
                            "entity_ids": list(self.sentence_to_entities.get(sid, [])),
                        }
                    )
                self._global_sentence_search_cache[cache_key] = self._clone_rows(rows)
                return rows

        rows = self.rank_sentences(query, self.sentences.keys(), topk=topk)
        for row in rows:
            row["source"] = "global_sentence_lexical"
        self._global_sentence_search_cache[cache_key] = self._clone_rows(rows)
        return rows

    def search_sentences_global_batch(self, queries: Iterable[str], topk: int = 8) -> Dict[str, List[dict]]:
        unique_queries = [str(query or "") for query in dict.fromkeys(queries)]
        results: Dict[str, List[dict]] = {}
        missing = []
        for query in unique_queries:
            cache_key = (query, int(topk))
            if cache_key in self._global_sentence_search_cache:
                results[query] = self._clone_rows(self._global_sentence_search_cache[cache_key])
            else:
                missing.append(query)

        if missing and self._dense_available("sentence"):
            query_embeddings = self._embed_queries(missing)
            if query_embeddings:
                valid_queries = []
                valid_embeddings = []
                for query, embedding in zip(missing, query_embeddings):
                    if embedding is not None:
                        valid_queries.append(query)
                        valid_embeddings.append(embedding)
                if valid_embeddings:
                    query_matrix = np.stack(valid_embeddings, axis=1)
                    score_matrix = np.asarray(self.sentence_embeddings @ query_matrix, dtype="float32")
                    idx_to_sid = self.id_maps.get("idx_to_sid") or list(self.sentences.keys())
                    for col, query in enumerate(valid_queries):
                        scores = score_matrix[:, col]
                        candidate_count = min(int(topk), len(scores))
                        if candidate_count <= 0:
                            rows = []
                        else:
                            candidate_idx = np.argpartition(-scores, candidate_count - 1)[:candidate_count]
                            candidate_idx = candidate_idx[np.argsort(-scores[candidate_idx])]
                            rows = []
                            for idx in candidate_idx:
                                sid = idx_to_sid[int(idx)]
                                sentence = self.sentences.get(sid)
                                if sentence is None:
                                    continue
                                rows.append(
                                    {
                                        **sentence,
                                        "score": round(float(scores[int(idx)]), 6),
                                        "source": "global_sentence_dense",
                                        "entity_ids": list(self.sentence_to_entities.get(sid, [])),
                                    }
                                )
                        cache_key = (query, int(topk))
                        self._global_sentence_search_cache[cache_key] = self._clone_rows(rows)
                        results[query] = rows

        for query in unique_queries:
            if query not in results:
                results[query] = self.search_sentences_global(query, topk=topk)
        return results

    def search_entities(self, query: str, topk: int = 8) -> List[dict]:
        """Search entity embeddings, falling back to lexical entity text."""

        cache_key = (str(query or ""), int(topk))
        if cache_key in self._entity_search_cache:
            return self._clone_rows(self._entity_search_cache[cache_key])

        if self._dense_available("entity"):
            query_embedding = self._embed_query(query)
            if query_embedding is not None:
                scores = np.asarray(self.entity_embeddings @ query_embedding, dtype="float32")
                idx_to_eid = self.id_maps.get("idx_to_eid") or list(self.entities.keys())
                rows = []
                for idx in np.argsort(-scores)[:topk]:
                    eid = idx_to_eid[int(idx)]
                    entity = self.entities.get(eid)
                    if entity is None:
                        continue
                    rows.append(
                        {
                            **entity,
                            "score": round(float(scores[int(idx)]), 6),
                            "source": "entity_dense",
                        }
                    )
                self._entity_search_cache[cache_key] = self._clone_rows(rows)
                return rows

        query_vector = Counter(tokenize_for_search(query))
        scored = []
        for eid, entity in self.entities.items():
            entity_text = self._entity_search_text(entity)
            score = self._score(query_vector, Counter(tokenize_for_search(entity_text)))
            scored.append({**entity, "score": round(float(score), 6), "source": "entity_lexical"})
        scored.sort(key=lambda item: (-item["score"], item["eid"]))
        rows = scored[:topk]
        self._entity_search_cache[cache_key] = self._clone_rows(rows)
        return rows

    def search_entities_batch(self, queries: Iterable[str], topk: int = 8) -> Dict[str, List[dict]]:
        unique_queries = [str(query or "") for query in dict.fromkeys(queries)]
        results: Dict[str, List[dict]] = {}
        missing = []
        for query in unique_queries:
            cache_key = (query, int(topk))
            if cache_key in self._entity_search_cache:
                results[query] = self._clone_rows(self._entity_search_cache[cache_key])
            else:
                missing.append(query)

        if missing and self._dense_available("entity"):
            query_embeddings = self._embed_queries(missing)
            if query_embeddings:
                valid_queries = []
                valid_embeddings = []
                for query, embedding in zip(missing, query_embeddings):
                    if embedding is not None:
                        valid_queries.append(query)
                        valid_embeddings.append(embedding)
                if valid_embeddings:
                    query_matrix = np.stack(valid_embeddings, axis=1)
                    score_matrix = np.asarray(self.entity_embeddings @ query_matrix, dtype="float32")
                    idx_to_eid = self.id_maps.get("idx_to_eid") or list(self.entities.keys())
                    for col, query in enumerate(valid_queries):
                        scores = score_matrix[:, col]
                        candidate_count = min(int(topk), len(scores))
                        if candidate_count <= 0:
                            rows = []
                        else:
                            candidate_idx = np.argpartition(-scores, candidate_count - 1)[:candidate_count]
                            candidate_idx = candidate_idx[np.argsort(-scores[candidate_idx])]
                            rows = []
                            for idx in candidate_idx:
                                eid = idx_to_eid[int(idx)]
                                entity = self.entities.get(eid)
                                if entity is None:
                                    continue
                                rows.append(
                                    {
                                        **entity,
                                        "score": round(float(scores[int(idx)]), 6),
                                        "source": "entity_dense",
                                    }
                                )
                        cache_key = (query, int(topk))
                        self._entity_search_cache[cache_key] = self._clone_rows(rows)
                        results[query] = rows

        for query in unique_queries:
            if query not in results:
                results[query] = self.search_entities(query, topk=topk)
        return results

    def hybrid_initial_retrieve(
        self,
        query: str,
        paragraph_topk: int = 20,
        high_conf_chunk_k: int = 5,
        sentence_topk: int = 8,
        entity_topk: int = 8,
        topk: int = 8,
        rrf_k: int = 60,
    ) -> List[dict]:
        """Fuse paragraph-local, global sentence, and entity-mention retrieval."""

        cache_key = (
            str(query or ""),
            int(paragraph_topk),
            int(high_conf_chunk_k),
            int(sentence_topk),
            int(entity_topk),
            int(topk),
            int(rrf_k),
        )
        if cache_key in self._hybrid_initial_cache:
            return self._clone_rows(self._hybrid_initial_cache[cache_key])

        paragraphs = self.search_paragraphs(query, topk=paragraph_topk)
        high_conf_pids = [paragraph["pid"] for paragraph in paragraphs[:high_conf_chunk_k]]
        paragraph_sentences = self.get_sentences_for_paragraphs(high_conf_pids)
        paragraph_ranked = self.rank_sentences(
            query,
            [sentence["sid"] for sentence in paragraph_sentences],
            topk=sentence_topk,
        )
        for row in paragraph_ranked:
            row["source"] = "paragraph_sentence"

        global_ranked = self.search_sentences_global(query, topk=sentence_topk)

        entity_ranked = self.search_entities(query, topk=entity_topk)
        entity_sids = []
        entity_score_by_sid: Dict[str, float] = {}
        entity_source_by_sid: Dict[str, Set[str]] = {}
        for entity in entity_ranked:
            eid = entity.get("eid")
            if not eid:
                continue
            for sid in self.entity_to_sentences.get(eid, []):
                if sid not in self.sentences:
                    continue
                append_unique(entity_sids, sid)
                entity_score_by_sid[sid] = max(entity_score_by_sid.get(sid, 0.0), float(entity.get("score", 0.0)))
                entity_source_by_sid.setdefault(sid, set()).add(str(entity.get("source", "entity")))

        mention_ranked = self.rank_sentences(query, entity_sids, topk=sentence_topk)
        for row in mention_ranked:
            sid = row["sid"]
            row["score"] = round(float(row.get("score", 0.0)) + 0.25 * entity_score_by_sid.get(sid, 0.0), 6)
            sources = sorted(entity_source_by_sid.get(sid, set()))
            row["source"] = "entity_mention" + (f":{','.join(sources)}" if sources else "")

        rows = self._rrf_fuse_sentences(
            [
                ("paragraph_sentence", paragraph_ranked),
                ("global_sentence", global_ranked),
                ("entity_mention", mention_ranked),
            ],
            topk=topk,
            rrf_k=rrf_k,
        )
        self._hybrid_initial_cache[cache_key] = self._clone_rows(rows)
        return rows

    def hybrid_initial_retrieve_batch(
        self,
        queries: Iterable[str],
        paragraph_topk: int = 20,
        high_conf_chunk_k: int = 5,
        sentence_topk: int = 8,
        entity_topk: int = 8,
        topk: int = 8,
        rrf_k: int = 60,
    ) -> Dict[str, List[dict]]:
        unique_queries = [str(query or "") for query in dict.fromkeys(queries)]
        results: Dict[str, List[dict]] = {}
        missing = []
        for query in unique_queries:
            cache_key = (
                query,
                int(paragraph_topk),
                int(high_conf_chunk_k),
                int(sentence_topk),
                int(entity_topk),
                int(topk),
                int(rrf_k),
            )
            if cache_key in self._hybrid_initial_cache:
                results[query] = self._clone_rows(self._hybrid_initial_cache[cache_key])
            else:
                missing.append(query)

        if not missing:
            return results

        global_by_query = self.search_sentences_global_batch(missing, topk=sentence_topk)
        entities_by_query = self.search_entities_batch(missing, topk=entity_topk)

        for query in missing:
            paragraphs = self.search_paragraphs(query, topk=paragraph_topk)
            high_conf_pids = [paragraph["pid"] for paragraph in paragraphs[:high_conf_chunk_k]]
            paragraph_sentences = self.get_sentences_for_paragraphs(high_conf_pids)
            paragraph_ranked = self.rank_sentences(
                query,
                [sentence["sid"] for sentence in paragraph_sentences],
                topk=sentence_topk,
            )
            for row in paragraph_ranked:
                row["source"] = "paragraph_sentence"

            entity_sids = []
            entity_score_by_sid: Dict[str, float] = {}
            entity_source_by_sid: Dict[str, Set[str]] = {}
            for entity in entities_by_query.get(query, []):
                eid = entity.get("eid")
                if not eid:
                    continue
                for sid in self.entity_to_sentences.get(eid, []):
                    if sid not in self.sentences:
                        continue
                    append_unique(entity_sids, sid)
                    entity_score_by_sid[sid] = max(entity_score_by_sid.get(sid, 0.0), float(entity.get("score", 0.0)))
                    entity_source_by_sid.setdefault(sid, set()).add(str(entity.get("source", "entity")))

            mention_ranked = self.rank_sentences(query, entity_sids, topk=sentence_topk)
            for row in mention_ranked:
                sid = row["sid"]
                row["score"] = round(float(row.get("score", 0.0)) + 0.25 * entity_score_by_sid.get(sid, 0.0), 6)
                sources = sorted(entity_source_by_sid.get(sid, set()))
                row["source"] = "entity_mention" + (f":{','.join(sources)}" if sources else "")

            rows = self._rrf_fuse_sentences(
                [
                    ("paragraph_sentence", paragraph_ranked),
                    ("global_sentence", global_by_query.get(query, [])),
                    ("entity_mention", mention_ranked),
                ],
                topk=topk,
                rrf_k=rrf_k,
            )
            cache_key = (
                query,
                int(paragraph_topk),
                int(high_conf_chunk_k),
                int(sentence_topk),
                int(entity_topk),
                int(topk),
                int(rrf_k),
            )
            self._hybrid_initial_cache[cache_key] = self._clone_rows(rows)
            results[query] = rows
        return results

    def get_local_context(self, sid: str) -> List[dict]:
        """Return adjacent paragraph sentences and sentence-graph neighbors."""

        if sid in self._local_context_cache:
            return self._clone_rows(self._local_context_cache[sid])
        if sid not in self.sentences:
            return []
        sentence = self.sentences[sid]
        pid = sentence.get("pid", "")
        ordered_sids = self.paragraph_to_sentences.get(pid, [])
        rows_by_sid: Dict[str, dict] = {}

        def add_context(context_sid: str, source: str, score: float) -> None:
            if context_sid == sid or context_sid not in self.sentences:
                return
            existing = rows_by_sid.get(context_sid)
            if existing is None:
                rows_by_sid[context_sid] = {
                    **self.sentences[context_sid],
                    "score": round(float(score), 6),
                    "source": source,
                    "context_of_sid": sid,
                    "entity_ids": list(self.sentence_to_entities.get(context_sid, [])),
                }
                return
            sources = set(str(existing.get("source", "")).split(","))
            sources.add(source)
            existing["source"] = ",".join(sorted(src for src in sources if src))
            existing["score"] = round(max(float(existing.get("score", 0.0)), float(score)), 6)

        if sid in ordered_sids:
            pos = ordered_sids.index(sid)
            if pos > 0:
                add_context(ordered_sids[pos - 1], "paragraph_prev", 1.0)
            if pos + 1 < len(ordered_sids):
                add_context(ordered_sids[pos + 1], "paragraph_next", 1.0)

        for neighbor_sid in self.sentence_to_neighbor_sentences.get(sid, []):
            add_context(neighbor_sid, "sentence_graph_neighbor", 0.9)

        rows = list(rows_by_sid.values())
        rows.sort(key=lambda item: (-float(item.get("score", 0.0)), item.get("sid", "")))
        self._local_context_cache[sid] = self._clone_rows(rows)
        return rows

    def propose_bridge_entities(
        self,
        source_eids: Iterable[str],
        question: str,
        selected_sids: Iterable[str],
        topm: int = 5,
    ) -> List[dict]:
        """Propose source->target bridge entities using structure first, then semantic rank fusion."""

        source_eid_list = [eid for eid in dict.fromkeys(source_eids) if eid in self.entities]
        if not source_eid_list:
            return []
        selected_sid_tuple = tuple(sorted(sid for sid in selected_sids if sid in self.sentences))
        cache_key = (tuple(source_eid_list), str(question or ""), selected_sid_tuple, int(topm))
        if cache_key in self._bridge_candidate_cache:
            return self._clone_rows(self._bridge_candidate_cache[cache_key])

        selected_sid_set = set(selected_sid_tuple)
        semantic_entity_rank = {
            entity["eid"]: rank
            for rank, entity in enumerate(self.search_entities(question, topk=max(topm * 4, 16)), start=1)
            if entity.get("eid")
        }
        rows = []
        source_set = set(source_eid_list)
        source_priority = {
            "selected_sentence": 0,
            "co_sentence": 1,
            "sentence_neighbor": 2,
            "synonym_entity": 3,
        }

        for source_eid in source_eid_list:
            candidate_sources: Dict[str, Set[str]] = {}
            candidate_reasons: Dict[str, Set[str]] = {}
            candidate_priority: Dict[str, int] = {}

            def add_candidate(target_eid: str, reason: str) -> None:
                if target_eid not in self.entities or target_eid == source_eid:
                    return
                if target_eid in source_set and reason not in {"selected_sentence", "co_sentence"}:
                    return
                candidate_sources.setdefault(target_eid, set()).add(reason)
                candidate_reasons.setdefault(target_eid, set()).add(reason)
                candidate_priority[target_eid] = min(
                    candidate_priority.get(target_eid, 99),
                    source_priority.get(reason, 99),
                )

            for selected_sid in selected_sid_set:
                selected_entities = self.sentence_to_entities.get(selected_sid, [])
                if source_eid not in selected_entities:
                    continue
                for target_eid in selected_entities:
                    add_candidate(target_eid, "selected_sentence")

            source_sids = set(self.entity_to_sentences.get(source_eid, [])[:32])
            for sid in source_sids:
                for target_eid in self.sentence_to_entities.get(sid, []):
                    add_candidate(target_eid, "co_sentence")
                for neighbor_sid in self.sentence_to_neighbor_sentences.get(sid, []):
                    for target_eid in self.sentence_to_entities.get(neighbor_sid, []):
                        add_candidate(target_eid, "sentence_neighbor")

            for similar in self.get_similar_entities(source_eid, topm=topm):
                target_eid = similar.get("eid")
                add_candidate(target_eid, "synonym_entity")

            structural_order = sorted(
                candidate_sources,
                key=lambda eid: (
                    candidate_priority.get(eid, 99),
                    -len(candidate_sources.get(eid, set())),
                    semantic_entity_rank.get(eid, 10**9),
                    eid,
                ),
            )
            structural_rank = {eid: rank for rank, eid in enumerate(structural_order, start=1)}
            for target_eid in structural_order:
                target_entity = self.entities[target_eid]
                score = 1.0 / float(60 + structural_rank[target_eid])
                if target_eid in semantic_entity_rank:
                    score += 1.0 / float(60 + semantic_entity_rank[target_eid])
                rows.append(
                    {
                        **target_entity,
                        "source_eid": source_eid,
                        "target_eid": target_eid,
                        "score": round(float(score), 6),
                        "structural_rank": structural_rank[target_eid],
                        "semantic_rank": semantic_entity_rank.get(target_eid),
                        "source": "bridge_candidate:" + ",".join(sorted(candidate_reasons.get(target_eid, set()))),
                    }
                )

        rows.sort(key=lambda item: (-float(item.get("score", 0.0)), item.get("source_eid", ""), item.get("target_eid", "")))
        rows = rows[:topm]
        self._bridge_candidate_cache[cache_key] = self._clone_rows(rows)
        return rows

    def bridge_entity(
        self,
        source_eid: str,
        target_eid: str,
        question: str,
        selected_sids: Iterable[str],
        bridge_query: Optional[str] = None,
        topk: int = 6,
    ) -> List[dict]:
        """Retrieve bridge evidence by fusing graph-structure rank with query semantic rank."""

        if target_eid not in self.entities:
            return []
        selected_sid_tuple = tuple(sorted(sid for sid in selected_sids if sid in self.sentences))
        bridge_query_key = str(bridge_query or "")
        cache_key = (
            str(source_eid or ""),
            str(target_eid or ""),
            str(question or ""),
            bridge_query_key,
            selected_sid_tuple,
            int(topk),
        )
        if cache_key in self._bridge_entity_cache:
            return self._clone_rows(self._bridge_entity_cache[cache_key])
        selected_sid_set = set(selected_sid_tuple)
        selected_text = " ".join(self.sentences[sid].get("text", "") for sid in selected_sid_set)
        source_surface = self.entities.get(source_eid, {}).get("canonical", source_eid)
        target_surface = self.entities.get(target_eid, {}).get("canonical", target_eid)
        semantic_query = " ".join(str(bridge_query or "").split())
        if not semantic_query:
            semantic_query = " ".join(part for part in [question, source_surface, target_surface, selected_text] if part).strip()

        candidate_sids: List[str] = []
        sid_sources: Dict[str, Set[str]] = {}
        sid_priority: Dict[str, int] = {}
        source_priority = {
            "bridge_target_mention": 0,
            "bridge_source_neighbor": 1,
            "bridge_local_context": 2,
        }

        def add_sid(candidate_sid: str, source: str) -> None:
            if candidate_sid not in self.sentences:
                return
            append_unique(candidate_sids, candidate_sid)
            sid_sources.setdefault(candidate_sid, set()).add(source)
            sid_priority[candidate_sid] = min(sid_priority.get(candidate_sid, 99), source_priority.get(source, 99))

        for sid in self.entity_to_sentences.get(target_eid, [])[:80]:
            add_sid(sid, "bridge_target_mention")
            for context in self.get_local_context(sid):
                add_sid(context["sid"], "bridge_local_context")

        if source_eid in self.entities:
            for sid in self.entity_to_sentences.get(source_eid, [])[:32]:
                for neighbor_sid in self.sentence_to_neighbor_sentences.get(sid, []):
                    add_sid(neighbor_sid, "bridge_source_neighbor")

        if not candidate_sids:
            return []

        structural_order = sorted(candidate_sids, key=lambda sid: (sid_priority.get(sid, 99), sid))
        structural_rank = {sid: rank for rank, sid in enumerate(structural_order, start=1)}
        semantic_ranked = self.rank_sentences(semantic_query, candidate_sids, topk=len(candidate_sids))
        semantic_rank = {row["sid"]: rank for rank, row in enumerate(semantic_ranked, start=1)}
        rows = []
        for sid in candidate_sids:
            sources = sid_sources.get(sid, set()) or {"bridge_entity"}
            score = 1.0 / float(60 + structural_rank.get(sid, len(candidate_sids) + 1))
            if sid in semantic_rank:
                score += 1.0 / float(60 + semantic_rank[sid])
            rows.append(
                {
                    **self.sentences[sid],
                    "score": round(float(score), 6),
                    "source": ",".join(sorted(sources)),
                    "bridge_source_eid": source_eid,
                    "bridge_target_eid": target_eid,
                    "bridge_query": semantic_query,
                    "structural_rank": structural_rank.get(sid),
                    "semantic_rank": semantic_rank.get(sid),
                    "entity_ids": list(self.sentence_to_entities.get(sid, [])),
                }
            )
        rows.sort(key=lambda item: (-float(item.get("score", 0.0)), item.get("sid", "")))
        rows = rows[:topk]
        self._bridge_entity_cache[cache_key] = self._clone_rows(rows)
        return rows

    def get_sentences_for_paragraphs(self, pids: Iterable[str]) -> List[dict]:
        sentences = []
        for pid in pids:
            for sid in self.paragraph_to_sentences.get(pid, []):
                sentence = self.sentences.get(sid)
                if sentence is not None:
                    sentences.append(sentence)
        return sentences

    def get_entities_for_sentence(self, sid: str) -> List[dict]:
        return [
            self.entities[eid]
            for eid in self.sentence_to_entities.get(sid, [])
            if eid in self.entities
        ]

    def get_sentences_for_entity(self, eid: str) -> List[dict]:
        return [
            self.sentences[sid]
            for sid in self.entity_to_sentences.get(eid, [])
            if sid in self.sentences
        ]

    def get_paragraphs_for_entity(self, eid: str) -> List[dict]:
        return [
            self.paragraphs[pid]
            for pid in self.entity_to_paragraphs.get(eid, [])
            if pid in self.paragraphs
        ]

    def get_neighbor_sentences(self, sid: str) -> List[dict]:
        return [
            self.sentences[neighbor_sid]
            for neighbor_sid in self.sentence_to_neighbor_sentences.get(sid, [])
            if neighbor_sid in self.sentences
        ]

    def get_similar_entities(self, eid: str, topm: int = 5) -> List[dict]:
        """Return graph-built synonym entities for ``eid``.

        Harness-G LOOKUP should be a graph operation: synonym expansion is
        allowed only through ``entity_synonym_edges.parquet`` loaded at graph
        load time. Runtime legacy/KNN/lexical fallbacks make the candidate pool
        depend on non-graph side channels and are intentionally not used here.
        """
        cache_key = (str(eid or ""), int(topm))
        if cache_key in self._similar_entities_cache:
            return self._clone_rows(self._similar_entities_cache[cache_key])
        rows = (self.entity_to_synonym_entities.get(eid, []) or [])[:topm]
        self._similar_entities_cache[cache_key] = self._clone_rows(rows)
        return rows

    def expand_entity(self, eid: str, q0: str, qh: str, topk: int = 6) -> List[dict]:
        cache_key = (str(eid or ""), str(q0 or ""), str(qh or ""), int(topk))
        if cache_key in self._expand_entity_cache:
            return self._clone_rows(self._expand_entity_cache[cache_key])
        candidate_sids, similar_eids, neighbor_sids = self._expand_candidate_sids(eid)
        rows = self._score_entity_candidates(
            eid, q0, qh, candidate_sids, similar_eids, neighbor_sids, topk
        )
        self._expand_entity_cache[cache_key] = self._clone_rows(rows)
        return rows

    def _expand_candidate_sids(self, eid: str) -> Tuple[Set[str], List[str], Set[str]]:
        """Return ``(candidate_sids, similar_eids, neighbor_sids)`` for an entity.

        Shared by :meth:`expand_entity` and :meth:`lookup_entity` so the V3
        ``LOOKUP`` candidate pool is graph-only: target mentions, graph-built
        synonym-entity mentions, and sentence-graph neighbors of those mentions.
        Paragraph-local context and runtime similarity fallbacks are intentionally
        excluded from LOOKUP."""

        target_sids = set(self.entity_to_sentences.get(eid, [])[:80])
        similar_eids = [entity.get("eid") for entity in self.get_similar_entities(eid, topm=5) if entity.get("eid")]
        similar_sids: Set[str] = set()
        for sim_eid in similar_eids:
            similar_sids.update(self.entity_to_sentences.get(sim_eid, [])[:40])

        candidate_sids = set(target_sids) | set(similar_sids)
        neighbor_sids: Set[str] = set()
        for sid in list(candidate_sids):
            neighbor_sids.update(self.sentence_to_neighbor_sentences.get(sid, []))
        candidate_sids.update(neighbor_sids)
        return candidate_sids, similar_eids, neighbor_sids

    def _score_entity_candidates(
        self,
        eid: str,
        q0: str,
        qh: str,
        candidate_sids: Iterable[str],
        similar_eids: Iterable[str],
        neighbor_sids: Iterable[str],
        topk: int,
    ) -> List[dict]:
        """Rank candidate sentences with the q0/qh/qmix dense+lexical score and
        the entity-structure prior. Shared by EXPAND and LOOKUP."""

        candidate_sids = set(candidate_sids)
        similar_eids = list(similar_eids)
        neighbor_sids = set(neighbor_sids)
        if self._dense_available("sentence"):
            dense_ranked = self._expand_entity_dense(eid, q0, qh, candidate_sids, similar_eids, neighbor_sids, topk)
            if dense_ranked is not None:
                return dense_ranked

        qmix = f"{q0} {qh}".strip()
        scored = []
        for sid in sorted(candidate_sids):
            sentence = self.sentences.get(sid)
            if not sentence:
                continue
            text = f"{sentence.get('title', '')} {sentence.get('text', '')}"
            base = (
                0.20 * self.score(q0, text)
                + 0.50 * self.score(qh, text)
                + 0.30 * self.score(qmix, text)
            )
            sentence_entities = set(self.sentence_to_entities.get(sid, []))
            if eid in sentence_entities:
                bonus = 0.10
                entity_source = "target"
            elif sentence_entities.intersection(similar_eids):
                bonus = 0.00
                entity_source = "similar"
            elif sid in neighbor_sids:
                bonus = 0.00
                entity_source = "sentence_neighbor"
            else:
                bonus = 0.00
                entity_source = "graph_candidate"
            scored.append(
                {
                    **sentence,
                    "score": round(float(base + bonus), 6),
                    "entity_ids": list(self.sentence_to_entities.get(sid, [])),
                    "entity_source": entity_source,
                    "expanded_eid": eid,
                }
            )
        scored.sort(key=lambda item: (-item["score"], item["sid"]))
        return scored[:topk]

    def lookup_entity(
        self,
        target_eid: str,
        mixquery: str,
        topk: int = 6,
        anchor_eid: Optional[str] = None,
    ) -> List[dict]:
        """V3 information-target retrieval over graph edges for ``target_eid``.

        The candidate pool contains target mentions, graph-built synonym-entity
        mentions, and sentence-graph neighbors of those mention sentences. The
        optional ``anchor_eid`` is retained as metadata for downstream credit
        assignment, but it no longer expands retrieval candidates.

        Retrieval is driven by a single ``mixquery`` (the question fused with
        the previously selected evidence text) rather than the legacy
        ``q0``/``qh``/``qmix`` three-way blend. The model no longer writes a
        ``|| need_query``; the environment builds ``mixquery`` and this is the
        only query used to rank the candidate pool. The entity-structure bonus
        (target mention +0.10) is retained since it does not depend on the
        query.
        """

        if target_eid not in self.entities:
            return []
        cache_key = (
            str(target_eid or ""),
            str(mixquery or ""),
            str(anchor_eid or ""),
            int(topk),
        )
        if cache_key in self._lookup_entity_cache:
            return self._clone_rows(self._lookup_entity_cache[cache_key])

        candidate_sids, similar_eids, neighbor_sids = self._expand_candidate_sids(target_eid)

        rows = self._score_lookup_candidates(
            target_eid, mixquery, candidate_sids, similar_eids, neighbor_sids, topk
        )
        for row in rows:
            row["lookup_target_eid"] = target_eid
            if anchor_eid:
                row["lookup_anchor_eid"] = anchor_eid
        self._lookup_entity_cache[cache_key] = self._clone_rows(rows)
        return rows

    def _expand_entity_dense(
        self,
        eid: str,
        q0: str,
        qh: str,
        candidate_sids: Iterable[str],
        similar_eids: List[str],
        neighbor_sids: Iterable[str],
        topk: int,
    ) -> Optional[List[dict]]:
        qmix = f"{q0} {qh}".strip()
        q0_emb = self._embed_query(q0)
        qh_emb = self._embed_query(qh)
        qmix_emb = self._embed_query(qmix)
        sid_to_idx = self.id_maps.get("sid_to_idx", {})
        candidate = [
            (sid, int(sid_to_idx[sid]))
            for sid in sorted(candidate_sids)
            if sid in self.sentences and sid in sid_to_idx
        ]
        if q0_emb is None or qh_emb is None or qmix_emb is None or not candidate:
            return None

        indices = np.asarray([idx for _, idx in candidate], dtype="int64")
        sentence_matrix = self.sentence_embeddings[indices]
        scores = (
            0.20 * np.asarray(sentence_matrix @ q0_emb, dtype="float32")
            + 0.50 * np.asarray(sentence_matrix @ qh_emb, dtype="float32")
            + 0.30 * np.asarray(sentence_matrix @ qmix_emb, dtype="float32")
        )
        rows = []
        similar_eid_set = set(similar_eids)
        neighbor_sid_set = set(neighbor_sids)
        for pos, (sid, _) in enumerate(candidate):
            sentence = self.sentences[sid]
            sentence_entities = set(self.sentence_to_entities.get(sid, []))
            if eid in sentence_entities:
                bonus = 0.10
                entity_source = "target"
            elif sentence_entities.intersection(similar_eid_set):
                bonus = 0.00
                entity_source = "similar"
            elif sid in neighbor_sid_set:
                bonus = 0.00
                entity_source = "sentence_neighbor"
            else:
                bonus = 0.00
                entity_source = "graph_candidate"
            rows.append(
                {
                    **sentence,
                    "score": round(float(scores[pos] + bonus), 6),
                    "entity_ids": list(self.sentence_to_entities.get(sid, [])),
                    "entity_source": entity_source,
                    "expanded_eid": eid,
                }
            )
        rows.sort(key=lambda item: (-item["score"], item["sid"]))
        return rows[:topk]

    def _score_lookup_candidates(
        self,
        eid: str,
        mixquery: str,
        candidate_sids: Iterable[str],
        similar_eids: Iterable[str],
        neighbor_sids: Iterable[str],
        topk: int,
    ) -> List[dict]:
        """Rank LOOKUP candidate sentences with a single ``mixquery`` signal.

        Unlike :meth:`_score_entity_candidates` (used by EXPAND), this scores
        the candidate pool with only ``mixquery`` — the question fused with the
        previously selected evidence text — rather than the ``q0``/``qh``/
        ``qmix`` three-way blend. The entity-structure prior (target mention
        +0.10, similar/neighbor/graph_candidate 0.00) is retained because it
        does not depend on the query and keeps target-mention sentences
        reachable even when ``mixquery`` is short.
        """

        candidate_sids = set(candidate_sids)
        similar_eids = list(similar_eids)
        neighbor_sids = set(neighbor_sids)
        if self._dense_available("sentence"):
            dense_ranked = self._lookup_dense(
                eid, mixquery, candidate_sids, similar_eids, neighbor_sids, topk
            )
            if dense_ranked is not None:
                return dense_ranked

        scored = []
        for sid in sorted(candidate_sids):
            sentence = self.sentences.get(sid)
            if not sentence:
                continue
            text = f"{sentence.get('title', '')} {sentence.get('text', '')}"
            base = self.score(mixquery, text)
            sentence_entities = set(self.sentence_to_entities.get(sid, []))
            if eid in sentence_entities:
                bonus = 0.10
                entity_source = "target"
            elif sentence_entities.intersection(similar_eids):
                bonus = 0.00
                entity_source = "similar"
            elif sid in neighbor_sids:
                bonus = 0.00
                entity_source = "sentence_neighbor"
            else:
                bonus = 0.00
                entity_source = "graph_candidate"
            scored.append(
                {
                    **sentence,
                    "score": round(float(base + bonus), 6),
                    "entity_ids": list(self.sentence_to_entities.get(sid, [])),
                    "entity_source": entity_source,
                    "expanded_eid": eid,
                }
            )
        scored.sort(key=lambda item: (-item["score"], item["sid"]))
        return scored[:topk]

    def _lookup_dense(
        self,
        eid: str,
        mixquery: str,
        candidate_sids: Iterable[str],
        similar_eids: List[str],
        neighbor_sids: Iterable[str],
        topk: int,
    ) -> Optional[List[dict]]:
        """Single-query dense rank for LOOKUP (mirrors ``_expand_entity_dense``
        but uses only ``mixquery`` instead of the q0/qh/qmix blend)."""

        mix_emb = self._embed_query(mixquery)
        sid_to_idx = self.id_maps.get("sid_to_idx", {})
        candidate = [
            (sid, int(sid_to_idx[sid]))
            for sid in sorted(candidate_sids)
            if sid in self.sentences and sid in sid_to_idx
        ]
        if mix_emb is None or not candidate:
            return None

        indices = np.asarray([idx for _, idx in candidate], dtype="int64")
        sentence_matrix = self.sentence_embeddings[indices]
        scores = np.asarray(sentence_matrix @ mix_emb, dtype="float32")
        rows = []
        similar_eid_set = set(similar_eids)
        neighbor_sid_set = set(neighbor_sids)
        for pos, (sid, _) in enumerate(candidate):
            sentence = self.sentences[sid]
            sentence_entities = set(self.sentence_to_entities.get(sid, []))
            if eid in sentence_entities:
                bonus = 0.10
                entity_source = "target"
            elif sentence_entities.intersection(similar_eid_set):
                bonus = 0.00
                entity_source = "similar"
            elif sid in neighbor_sid_set:
                bonus = 0.00
                entity_source = "sentence_neighbor"
            else:
                bonus = 0.00
                entity_source = "graph_candidate"
            rows.append(
                {
                    **sentence,
                    "score": round(float(scores[pos] + bonus), 6),
                    "entity_ids": list(self.sentence_to_entities.get(sid, [])),
                    "entity_source": entity_source,
                    "expanded_eid": eid,
                }
            )
        rows.sort(key=lambda item: (-item["score"], item["sid"]))
        return rows[:topk]

    def _rrf_fuse_sentences(
        self,
        ranked_lists: List[Tuple[str, List[dict]]],
        topk: int,
        rrf_k: int,
    ) -> List[dict]:
        fused: Dict[str, dict] = {}
        for list_source, rows in ranked_lists:
            for rank, row in enumerate(rows, start=1):
                sid = row.get("sid")
                if not sid or sid not in self.sentences:
                    continue
                item = fused.setdefault(
                    sid,
                    {
                        **self.sentences[sid],
                        "sid": sid,
                        "score": 0.0,
                        "source": set(),
                        "retrieval_scores": {},
                        "entity_ids": list(self.sentence_to_entities.get(sid, [])),
                    },
                )
                item["score"] += 1.0 / float(rrf_k + rank)
                row_source = str(row.get("source") or list_source)
                item["source"].add(row_source)
                item["retrieval_scores"][list_source] = float(row.get("score", 0.0))

        rows = []
        for sid, item in fused.items():
            sources = ",".join(sorted(item.pop("source")))
            rows.append(
                {
                    **item,
                    "score": round(float(item["score"]), 6),
                    "source": sources,
                    "hybrid_retrieval": True,
                }
            )
        rows.sort(key=lambda item: (-float(item.get("score", 0.0)), item.get("sid", "")))
        return rows[:topk]

    def _entity_search_text(self, entity: dict) -> str:
        parts = [str(entity.get("canonical", "") or ""), str(entity.get("label", "") or "")]
        parts.extend(str(item) for item in entity.get("surface_forms", []) if str(item))
        return " ".join(part for part in parts if part)

    def _entity_query_score(self, query: str, eid: str) -> float:
        if eid not in self.entities:
            return 0.0
        if self._dense_available("entity"):
            query_embedding = self._embed_query(query)
            eid_to_idx = self.id_maps.get("eid_to_idx", {})
            if query_embedding is not None and eid in eid_to_idx:
                return float(self.entity_embeddings[int(eid_to_idx[eid])] @ query_embedding)
        return self.score(query, self._entity_search_text(self.entities[eid]))

    def _entity_pair_score(self, source_eid: str, target_eid: str) -> float:
        if source_eid not in self.entities or target_eid not in self.entities:
            return 0.0
        if self._dense_available("entity"):
            eid_to_idx = self.id_maps.get("eid_to_idx", {})
            if source_eid in eid_to_idx and target_eid in eid_to_idx:
                return float(
                    self.entity_embeddings[int(eid_to_idx[source_eid])]
                    @ self.entity_embeddings[int(eid_to_idx[target_eid])]
                )
        return self.score(self._entity_search_text(self.entities[source_eid]), self._entity_search_text(self.entities[target_eid]))

    @staticmethod
    def _clone_rows(rows: List[dict]) -> List[dict]:
        cloned = []
        for row in rows:
            item = dict(row)
            if isinstance(item.get("entity_ids"), list):
                item["entity_ids"] = list(item["entity_ids"])
            if isinstance(item.get("retrieval_scores"), dict):
                item["retrieval_scores"] = dict(item["retrieval_scores"])
            cloned.append(item)
        return cloned

    def _load_dense_embeddings(self) -> None:
        if not self.graph_dir:
            return
        if not self.manifest.get("build_embeddings"):
            return
        config_path = self.graph_dir / "embedding_config.json"
        if config_path.exists():
            self.embedding_config = read_json(config_path)
        else:
            self.embedding_config = {}

        paths = {
            "passage": self.graph_dir / "passage_embeddings.npy",
            "sentence": self.graph_dir / "sentence_embeddings.npy",
            "entity": self.graph_dir / "entity_embeddings.npy",
        }
        if not all(path.exists() for path in paths.values()):
            return
        mmap_mode = "r" if os.environ.get("HARNESS_G_EMBEDDING_MMAP", "").lower() in {"1", "true", "yes"} else None
        self.passage_embeddings = np.load(paths["passage"], mmap_mode=mmap_mode)
        self.sentence_embeddings = np.load(paths["sentence"], mmap_mode=mmap_mode)
        self.entity_embeddings = np.load(paths["entity"], mmap_mode=mmap_mode)

        expected = {
            "passage": len(self.paragraphs),
            "sentence": len(self.sentences),
            "entity": len(self.entities),
        }
        actual = {
            "passage": int(self.passage_embeddings.shape[0]),
            "sentence": int(self.sentence_embeddings.shape[0]),
            "entity": int(self.entity_embeddings.shape[0]),
        }
        for key, expected_rows in expected.items():
            if actual[key] != expected_rows:
                self._dense_disabled_reason = f"{key} embedding row mismatch: {actual[key]} != {expected_rows}"
                self.passage_embeddings = None
                self.sentence_embeddings = None
                self.entity_embeddings = None
                return

    def _dense_available(self, kind: str) -> bool:
        if self._dense_disabled_reason:
            return False
        if kind == "passage":
            return self.passage_embeddings is not None
        if kind == "sentence":
            return self.sentence_embeddings is not None
        if kind == "entity":
            return self.entity_embeddings is not None
        return False

    def _embed_query(self, query: str) -> Optional[np.ndarray]:
        cache_key = str(query or "")
        if cache_key in self._query_embedding_cache:
            return self._query_embedding_cache[cache_key]
        encoder = self._get_embedding_encoder()
        if encoder is None:
            return None
        instruction = self.embedding_config.get("query_instruction", DEFAULT_QUERY_INSTRUCTION)
        embedding = encoder.encode([cache_key], batch_size=1, query_instruction=instruction)[0]
        self._query_embedding_cache[cache_key] = embedding
        return embedding

    def _embed_queries(self, queries: Iterable[str]) -> List[Optional[np.ndarray]]:
        query_list = [str(query or "") for query in queries]
        missing = [query for query in query_list if query not in self._query_embedding_cache]
        if missing:
            encoder = self._get_embedding_encoder()
            if encoder is None:
                return [self._query_embedding_cache.get(query) for query in query_list]
            instruction = self.embedding_config.get("query_instruction", DEFAULT_QUERY_INSTRUCTION)
            batch_size = int(self.embedding_config.get("embedding_batch_size") or 32)
            embeddings = encoder.encode(missing, batch_size=max(batch_size, 1), query_instruction=instruction)
            for query, embedding in zip(missing, embeddings):
                self._query_embedding_cache[query] = embedding
        return [self._query_embedding_cache.get(query) for query in query_list]

    def _get_embedding_encoder(self):
        if self._embedding_encoder is not None:
            return self._embedding_encoder
        model_path = (
            os.environ.get("HARNESS_G_EMBEDDING_MODEL_PATH")
            or self.embedding_config.get("embedding_model_path")
            or self.manifest.get("embedding_model_path")
        )
        if not model_path:
            self._dense_disabled_reason = "missing embedding model path"
            return None
        try:
            self._embedding_encoder = TransformerEmbeddingModel(
                model_path,
                device=resolve_embedding_device(os.environ.get("HARNESS_G_EMBEDDING_DEVICE") or self.embedding_config.get("embedding_device")),
            )
        except Exception as exc:
            self._dense_disabled_reason = f"failed to load embedding model: {exc}"
            return None
        return self._embedding_encoder

    @staticmethod
    def _score(query_vector: Counter, doc_vector: Counter) -> float:
        return cosine_counter(query_vector, doc_vector)

    @staticmethod
    def _read_parquet_records(path: Path) -> List[dict]:
        try:
            import pandas as pd
        except Exception as exc:
            raise RuntimeError("pandas/pyarrow are required to load clean Harness-G graph storage") from exc
        return pd.read_parquet(path).to_dict("records")

    @staticmethod
    def _as_list(value) -> List[str]:
        if value is None:
            return []
        if isinstance(value, float) and math.isnan(value):
            return []
        if isinstance(value, list):
            return [str(item) for item in value]
        if isinstance(value, tuple):
            return [str(item) for item in value]
        if hasattr(value, "tolist") and not isinstance(value, (str, bytes, dict)):
            return [str(item) for item in value.tolist()]
        if isinstance(value, str):
            stripped = value.strip()
            if not stripped:
                return []
            if stripped.startswith("["):
                try:
                    loaded = json.loads(stripped)
                    if isinstance(loaded, list):
                        return [str(item) for item in loaded]
                except Exception:
                    pass
            return [stripped]
        return [str(value)]
