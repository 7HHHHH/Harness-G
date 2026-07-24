import json

from harness_g.graph_builder import build_graph
from agent.tool.tools.harness_g_tool import HarnessGTool
from scripts import run_harness_g_api


CORPUS = """{"id": "1", "title": "Ada Lovelace", "contents": "Ada Lovelace was born in London in 1815 while collaborating with Charles Babbage on the Analytical Engine across many mathematical notes."}
{"id": "2", "title": "Charles Babbage", "contents": "Charles Babbage designed the Analytical Engine. He was born in London."}
"""


def test_stateful_harness_g_api_sessions(tmp_path):
    from fastapi.testclient import TestClient

    corpus_path = tmp_path / "corpus.jsonl"
    graph_dir = tmp_path / "harness_g_graph"
    corpus_path.write_text(CORPUS, encoding="utf-8")
    build_graph(corpus_path=corpus_path, output_dir=graph_dir)

    run_harness_g_api.configure_api(data_source="test", graph_dir=graph_dir)
    run_harness_g_api.reset_sessions()
    client = TestClient(run_harness_g_api.app)

    health = client.get("/health").json()
    assert health["status"] == "ok"

    r = client.post("/harness_g_step", json={"requests": [{"session_id": "s1", "question": "Where was Ada Lovelace born?", "action": "INIT"}]}).json()
    obs = r[0]["observation"]
    assert "[HARNESS_G_OBS]" in obs
    assert not obs.lstrip().startswith('{"results"')
    assert "A0 = SELECT" in obs

    r = client.post(
        "/harness_g_step",
        json={
            "requests": [
                {"session_id": "batch_a", "question": "Where was Ada Lovelace born?", "action": "INIT"},
                {"session_id": "batch_b", "question": "Where was Ada Lovelace born?", "action": "INIT"},
            ]
        },
    ).json()
    assert len(r) == 2
    assert all("A0 = SELECT" in item["observation"] for item in r)

    r = client.post("/harness_g_step", json={"requests": [{"session_id": "s1", "question": "Where was Ada Lovelace born?", "action": "A0"}]}).json()
    obs = r[0]["observation"]
    assert "frontier_entities:" in obs

    r = client.post("/harness_g_step", json={"requests": [{"session_id": "s2", "question": "Where was Ada Lovelace born?", "action": "A0"}]}).json()
    obs = r[0]["observation"]
    assert "visible_sentences:" in obs

    r = client.post("/harness_g_step", json={"requests": [{"session_id": "s1", "question": "Where was Ada Lovelace born?", "action": "Ada Lovelace birthplace"}]}).json()
    obs = r[0]["observation"]
    assert "INVALID_ACTION" in obs



def test_batched_init_nav_events_keep_their_own_session_ids(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    monkeypatch.setenv("HARNESS_G_ACTION_V3", "1")
    monkeypatch.setenv("HARNESS_G_ANSWER_WITH", "0")
    monkeypatch.setenv("HARNESS_G_INTERFACE_FREEQ", "0")
    monkeypatch.setenv("HARNESS_G_RUN_ID", "unit_batched_init")
    nav_path = tmp_path / "nav_events.jsonl"
    monkeypatch.setenv("HARNESS_G_NAV_EVENTS_PATH", str(nav_path))

    corpus_path = tmp_path / "corpus.jsonl"
    graph_dir = tmp_path / "harness_g_graph"
    corpus_path.write_text(CORPUS, encoding="utf-8")
    build_graph(corpus_path=corpus_path, output_dir=graph_dir)

    run_harness_g_api.configure_api(data_source="test", graph_dir=graph_dir)
    run_harness_g_api.reset_sessions()
    client = TestClient(run_harness_g_api.app)

    session_ids = ["batch_init_a", "batch_init_b", "batch_init_c"]
    response = client.post(
        "/harness_g_step",
        json={
            "requests": [
                {
                    "session_id": session_id,
                    "question": "Where was Ada Lovelace born?",
                    "action": "INIT",
                }
                for session_id in session_ids
            ]
        },
    ).json()

    assert len(response) == len(session_ids)
    events = [json.loads(line) for line in nav_path.read_text(encoding="utf-8").splitlines()]
    assert [event["action_type"] for event in events] == ["INIT"] * len(session_ids)
    assert [event["session_id"] for event in events] == session_ids
    assert {event["session_id"] for event in events} == set(session_ids)


def test_nav_events_are_exactly_once_and_include_select_lookup(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    monkeypatch.setenv("HARNESS_G_ACTION_V3", "1")
    monkeypatch.setenv("HARNESS_G_ANSWER_WITH", "0")
    monkeypatch.setenv("HARNESS_G_INTERFACE_FREEQ", "0")
    monkeypatch.setenv("HARNESS_G_RUN_ID", "unit_nav")
    nav_path = tmp_path / "nav_events.jsonl"
    monkeypatch.setenv("HARNESS_G_NAV_EVENTS_PATH", str(nav_path))

    corpus_path = tmp_path / "corpus.jsonl"
    graph_dir = tmp_path / "harness_g_graph"
    corpus_path.write_text(CORPUS, encoding="utf-8")
    build_graph(corpus_path=corpus_path, output_dir=graph_dir)

    run_harness_g_api.configure_api(data_source="test", graph_dir=graph_dir, max_turns=8)
    run_harness_g_api.reset_sessions()
    client = TestClient(run_harness_g_api.app)

    client.post("/harness_g_step", json={"requests": [{"session_id": "nav_s1", "question": "Where was Ada Lovelace born?", "action": "INIT"}]}).json()
    # A repeated INIT request on an initialized session must not re-log stale INIT.
    client.post("/harness_g_step", json={"requests": [{"session_id": "nav_s1", "question": "Where was Ada Lovelace born?", "action": "INIT"}]}).json()

    episode = run_harness_g_api._SESSIONS["nav_s1"]
    select_id = next(aid for aid, a in episode.current_action_map.items() if a["type"] == "SELECT")
    client.post("/harness_g_step", json={"requests": [{"session_id": "nav_s1", "question": "Where was Ada Lovelace born?", "action": select_id}]}).json()
    lookup_id = next(aid for aid, a in episode.current_action_map.items() if a["type"] == "LOOKUP")
    client.post("/harness_g_step", json={"requests": [{"session_id": "nav_s1", "question": "Where was Ada Lovelace born?", "action": lookup_id}]}).json()

    events = [json.loads(line) for line in nav_path.read_text(encoding="utf-8").splitlines()]
    assert [event["action_type"] for event in events] == ["INIT", "SELECT", "LOOKUP"]
    assert sum(event["action_type"] == "INIT" for event in events) == 1
    select_event = events[1]
    assert select_event["query_text"] is None
    assert select_event["result_ids"] == [episode.selected_evidence[0]]
    lookup_event = events[2]
    assert lookup_event["query_text"] == episode._lookup_history[-1]["mixquery"]
    # Mixquery contract: the full question is preserved as a prefix and the
    # total length stays within the mixquery word budget.
    question_words = "Where was Ada Lovelace born?".split()
    query_words = lookup_event["query_text"].split()
    assert query_words[: len(question_words)] == question_words
    assert len(query_words) <= max(episode.mixquery_max_words, len(question_words))
    assert lookup_event["result_ids"]


def test_harness_g_tool_unwraps_legacy_json_observation():
    raw = "[HARNESS_G_OBS]\navailable_actions:\nA0 = SELECT S0\n[/HARNESS_G_OBS]"
    assert HarnessGTool._unwrap_api_result({"results": raw}) == raw
    assert HarnessGTool._unwrap_api_result('{"results": "[HARNESS_G_OBS]\\navailable_actions:\\nA0 = SELECT S0\\n[/HARNESS_G_OBS]"}') == raw
