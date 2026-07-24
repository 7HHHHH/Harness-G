from pathlib import Path

from harness_g.graph_builder import build_graph
from harness_g.graph_index import HarnessGGraphIndex


CORPUS = """{"id": "1", "title": "Ada Lovelace", "contents": "Ada Lovelace was born in London. She worked with Charles Babbage on the Analytical Engine."}
{"id": "2", "title": "Charles Babbage", "contents": "Charles Babbage designed the Analytical Engine. He was born in London."}
"""

SAME_SENTENCE_CORPUS = """{"id": "1", "title": "Bridge Test", "contents": "Alpha Person met Beta Person."}
"""

NEIGHBOR_CORPUS = """{"id": "1", "title": "Neighbor Test", "contents": "Alpha Person stood alone. Beta Person stood nearby."}
"""


def _eid_by_canonical(index, canonical):
    return next(eid for eid, entity in index.entities.items() if entity.get("canonical") == canonical)


def test_graph_index_search_and_accessors(tmp_path):
    corpus_path = tmp_path / "corpus.jsonl"
    graph_dir = tmp_path / "harness_g_graph"
    corpus_path.write_text(CORPUS, encoding="utf-8")
    build_graph(corpus_path=corpus_path, output_dir=graph_dir)

    index = HarnessGGraphIndex.load(graph_dir)
    paragraphs = index.search_paragraphs("Ada Lovelace birthplace", topk=2)
    assert paragraphs

    sentences = index.get_sentences_for_paragraphs([paragraph["pid"] for paragraph in paragraphs])
    ranked = index.rank_sentences("Ada Lovelace birthplace", [sentence["sid"] for sentence in sentences], topk=2)
    assert ranked
    assert index.get_neighbor_sentences(ranked[0]["sid"])

    entities = index.get_entities_for_sentence(ranked[0]["sid"])
    assert isinstance(entities, list)
    assert index.expand_entity(entities[0]["eid"], "Where was Ada Lovelace born?", "Ada Lovelace birthplace", topk=3)


def test_graph_index_v2_hybrid_context_and_bridge(tmp_path):
    corpus_path = tmp_path / "corpus.jsonl"
    graph_dir = tmp_path / "harness_g_graph"
    corpus_path.write_text(CORPUS, encoding="utf-8")
    build_graph(corpus_path=corpus_path, output_dir=graph_dir)

    index = HarnessGGraphIndex.load(graph_dir)
    global_sentences = index.search_sentences_global("Analytical Engine designer", topk=2)
    assert global_sentences
    assert global_sentences[0]["source"].startswith("global_sentence")

    entities = index.search_entities("Charles Babbage", topk=3)
    assert entities
    assert any("babbage" in entity.get("canonical", "") for entity in entities)

    hybrid = index.hybrid_initial_retrieve("Who designed the Analytical Engine?", topk=3)
    assert hybrid
    assert all("source" in sentence for sentence in hybrid)
    hybrid_batch = index.hybrid_initial_retrieve_batch(
        ["Who designed the Analytical Engine?", "Who designed the Analytical Engine?"],
        topk=3,
    )
    assert list(hybrid_batch) == ["Who designed the Analytical Engine?"]
    assert [row["sid"] for row in hybrid_batch["Who designed the Analytical Engine?"]] == [row["sid"] for row in hybrid]

    selected_sid = next(sid for sid, sentence in index.sentences.items() if "worked with Charles Babbage" in sentence["text"])
    context = index.get_local_context(selected_sid)
    assert context
    assert any("paragraph_" in sentence["source"] or "sentence_graph_neighbor" in sentence["source"] for sentence in context)

    source_eids = index.sentence_to_entities[selected_sid]
    bridge_candidates = index.propose_bridge_entities(source_eids, "Who designed the Analytical Engine?", [selected_sid], topm=3)
    assert bridge_candidates
    target = bridge_candidates[0]
    bridged = index.bridge_entity(
        target["source_eid"],
        target["target_eid"],
        "Who designed the Analytical Engine?",
        [selected_sid],
        bridge_query="Analytical Engine designer",
        topk=3,
    )
    assert bridged
    assert all("bridge_" in sentence["source"] for sentence in bridged)
    assert all(sentence.get("bridge_query") == "Analytical Engine designer" for sentence in bridged)


def test_bridge_candidates_allow_selected_same_sentence_frontier_pairs(tmp_path):
    corpus_path = tmp_path / "corpus.jsonl"
    graph_dir = tmp_path / "harness_g_graph"
    corpus_path.write_text(SAME_SENTENCE_CORPUS, encoding="utf-8")
    build_graph(corpus_path=corpus_path, output_dir=graph_dir)

    index = HarnessGGraphIndex.load(graph_dir)
    selected_sid = next(iter(index.sentences))
    alpha = _eid_by_canonical(index, "alpha person")
    beta = _eid_by_canonical(index, "beta person")

    candidates = index.propose_bridge_entities([alpha, beta], "Alpha Beta bridge", [selected_sid], topm=10)
    pairs = {(row["source_eid"], row["target_eid"]) for row in candidates}
    assert (alpha, beta) in pairs
    assert (beta, alpha) in pairs
    assert all(source != target for source, target in pairs)
    assert any("selected_sentence" in row["source"] or "co_sentence" in row["source"] for row in candidates)


def test_bridge_candidates_still_filter_frontier_noise_from_neighbor_sources(tmp_path):
    corpus_path = tmp_path / "corpus.jsonl"
    graph_dir = tmp_path / "harness_g_graph"
    corpus_path.write_text(NEIGHBOR_CORPUS, encoding="utf-8")
    build_graph(corpus_path=corpus_path, output_dir=graph_dir)

    index = HarnessGGraphIndex.load(graph_dir)
    alpha = _eid_by_canonical(index, "alpha person")
    beta = _eid_by_canonical(index, "beta person")
    alpha_sid = next(sid for sid, eids in index.sentence_to_entities.items() if alpha in eids)

    candidates = index.propose_bridge_entities([alpha, beta], "nearby bridge", [alpha_sid], topm=10)
    frontier = {alpha, beta}
    assert not any(row["source_eid"] in frontier and row["target_eid"] in frontier for row in candidates)
