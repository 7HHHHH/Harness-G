from harness_g.env import HarnessGEpisode
from harness_g.graph_builder import build_graph
from harness_g.graph_index import HarnessGGraphIndex


CORPUS = """{"id": "1", "title": "Ada Lovelace", "contents": "Ada Lovelace was born in London. She worked with Charles Babbage on the Analytical Engine."}
{"id": "2", "title": "Charles Babbage", "contents": "Charles Babbage designed the Analytical Engine. He was born in London."}
"""


def _episode(tmp_path):
    corpus_path = tmp_path / "corpus.jsonl"
    graph_dir = tmp_path / "harness_g_graph"
    corpus_path.write_text(CORPUS, encoding="utf-8")
    build_graph(corpus_path=corpus_path, output_dir=graph_dir)
    return HarnessGEpisode("s1", "Where was Ada Lovelace born?", HarnessGGraphIndex.load(graph_dir))


def test_env_state_machine(tmp_path, monkeypatch):
    monkeypatch.delenv("HARNESS_G_SHOW_RAW_SCORE", raising=False)
    ep = _episode(tmp_path)
    obs = ep.step("INIT")
    assert "mode: harness_g" in obs
    assert "visible_sentences:" in obs
    assert "score:" not in obs
    assert "A0 = SELECT" in obs
    types = {action["type"] for action in ep.current_action_map.values()}
    assert types <= {"SELECT", "ANSWER_WITH", "LOOKUP"}
    assert ep.snc_seen_sids == ep.current_visible_sids
    initial_seen = list(ep.snc_seen_sids)

    obs = ep.step("A0")
    assert "frontier_entities:" in obs
    assert "available_actions:" in obs
    assert any(action["type"] == "SELECT" for action in ep.current_action_map.values())
    assert any(action["type"] == "ANSWER_WITH" for action in ep.current_action_map.values())
    assert any(action["type"] == "LOOKUP" for action in ep.current_action_map.values())
    assert any(action["type"] == "ANSWER" for action in ep.current_action_map.values())
    assert ep.metrics.lookup_action_available_count > 0
    assert ep.metrics.answer_with_action_available_count > 0
    assert ep.metrics.stop_action_available_count > 0
    answer_id = next(aid for aid, action in ep.current_action_map.items() if action["type"] == "ANSWER")
    assert answer_id == list(ep.current_action_map.keys())[-1]
    assert answer_id != "A0"
    assert len(ep.selected_evidence) == 1

    lookup_id = next(aid for aid, action in ep.current_action_map.items() if action["type"] == "LOOKUP")
    obs = ep.step(lookup_id)
    assert "event: LOOKUP" in obs
    assert "new_visible_sentences:" in obs
    assert ep.metrics.lookup_count == 1
    assert ep.snc_seen_sids[: len(initial_seen)] == initial_seen
    assert len(ep.snc_seen_sids) == len(set(ep.snc_seen_sids))

    obs = ep.step(f"{lookup_id} || Ada Lovelace birthplace")
    assert "event: INVALID_ACTION" in obs

    obs = ep.step("Ada Lovelace birthplace")
    assert "event: INVALID_ACTION" in obs

    answer_id = next(aid for aid, action in ep.current_action_map.items() if action["type"] == "ANSWER")
    obs = ep.step(answer_id)
    assert "Now provide the final answer" in obs


def test_score_visible_ablation_flag(tmp_path, monkeypatch):
    monkeypatch.setenv("HARNESS_G_SHOW_RAW_SCORE", "1")
    ep = _episode(tmp_path)
    obs = ep.step("INIT")
    assert "score:" in obs





