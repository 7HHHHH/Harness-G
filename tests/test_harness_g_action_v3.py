"""Regression tests for the discrete action space.

Covers protocol parsing, env menu construction, the LOOKUP retrieval union,
ANSWER_WITH harvest+answer semantics, SNC previews, and the reward bridge.
"""

import pytest

from harness_g.env import HarnessGEpisode
from harness_g.graph_builder import build_graph
from harness_g.graph_index import HarnessGGraphIndex
from harness_g.protocol import parse_harness_g_action
from harness_g.snc_preview import preview_candidate
from harness_g.utils import is_bad_lookup_target
from verl.utils.reward_score.harness_g_qa import analyze


CORPUS = """{"id": "1", "title": "Ada Lovelace", "contents": "Ada Lovelace was born in London in 1815. She worked with Charles Babbage on the Analytical Engine."}
{"id": "2", "title": "Charles Babbage", "contents": "Charles Babbage designed the Analytical Engine. He was British and was born in London."}
"""


@pytest.fixture
def v3_env(tmp_path, monkeypatch):
    corpus_path = tmp_path / "corpus.jsonl"
    graph_dir = tmp_path / "harness_g_graph"
    corpus_path.write_text(CORPUS, encoding="utf-8")
    build_graph(corpus_path=corpus_path, output_dir=graph_dir)
    index = HarnessGGraphIndex.load(graph_dir)
    return lambda: HarnessGEpisode("v3", "Where was Ada Lovelace born?", index, max_turns=8)


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------
def test_v3_protocol_accepts_new_actions():
    action_map = {
        "A0": {"type": "SELECT"},
        "A1": {"type": "LOOKUP"},
        "A2": {"type": "ANSWER_WITH"},
        "A3": {"type": "ANSWER"},
    }
    assert parse_harness_g_action("A1", action_map, initialized=True)["is_valid"]
    # V3 LOOKUP no longer accepts a || need query (the env builds mixquery).
    assert not parse_harness_g_action("A1 || place of death", action_map, initialized=True)["is_valid"]
    assert parse_harness_g_action("A2", action_map, initialized=True)["is_valid"]
    assert parse_harness_g_action("A3", action_map, initialized=True)["is_valid"]
    # ANSWER / ANSWER_WITH must not carry a || query
    assert not parse_harness_g_action("A2 || x", action_map, initialized=True)["is_valid"]
    assert not parse_harness_g_action("A3 || x", action_map, initialized=True)["is_valid"]
    # In V3 mode no action accepts || (the env builds the LOOKUP query itself);
    # even a legacy EXPAND action is rejected when the V3 flag is on.
    legacy = {"A1": {"type": "EXPAND_ENTITY"}}
    assert not parse_harness_g_action("A1 || x", legacy, initialized=True)["is_valid"]



def test_bad_lookup_target_filter():
    assert is_bad_lookup_target("1815", "DATE")
    assert is_bad_lookup_target("1815", "")
    assert is_bad_lookup_target("American", "NORP")
    assert is_bad_lookup_target("british", "")  # nationality adjective surface
    assert is_bad_lookup_target("Leslie H", "PERSON")  # split-name fragment
    assert not is_bad_lookup_target("Ada Lovelace", "PERSON")
    assert not is_bad_lookup_target("London", "GPE")


# ---------------------------------------------------------------------------
# Env menus
# ---------------------------------------------------------------------------
def test_v3_menu_only_contains_v3_actions(v3_env):
    ep = v3_env()
    ep.step("INIT")
    init_types = {a["type"] for a in ep.current_action_map.values()}
    assert init_types <= {"SELECT", "ANSWER_WITH", "LOOKUP"}
    assert not (init_types & {"BRIDGE_ENTITY", "EXPAND_ENTITY", "OPEN_CONTEXT", "REWRITE_QUERY", "STOP"})

    select_id = next(aid for aid, a in ep.current_action_map.items() if a["type"] == "SELECT")
    ep.step(select_id)
    sel_types = {a["type"] for a in ep.current_action_map.values()}
    assert sel_types <= {"SELECT", "ANSWER_WITH", "LOOKUP", "ANSWER"}
    assert "ANSWER" in sel_types
    assert "LOOKUP" in sel_types
    assert not (sel_types & {"BRIDGE_ENTITY", "EXPAND_ENTITY", "OPEN_CONTEXT", "REWRITE_QUERY", "STOP"})


def test_v3_lookup_step_emits_marker_and_metric(v3_env):
    ep = v3_env()
    ep.step("INIT")
    select_id = next(aid for aid, a in ep.current_action_map.items() if a["type"] == "SELECT")
    ep.step(select_id)
    lookup_id = next(aid for aid, a in ep.current_action_map.items() if a["type"] == "LOOKUP")
    obs = ep.step(lookup_id)
    assert "event: LOOKUP" in obs
    assert "new_visible_sentences:" in obs
    assert ep.metrics.lookup_count == 1


def test_v3_lookup_dedup_counts_repeat(v3_env):
    ep = v3_env()
    ep.step("INIT")
    eid = next(iter(ep.entities))
    ep._lookup(eid)
    ep._lookup(eid)
    assert ep.metrics.lookup_count == 2
    assert ep.metrics.duplicate_lookup_count == 1


def test_v3_lookup_target_not_reoffered_after_use(v3_env):
    ep = v3_env()
    ep.step("INIT")
    select_id = next(aid for aid, a in ep.current_action_map.items() if a["type"] == "SELECT")
    ep.step(select_id)
    lookup_id, lookup_action = next(
        (aid, a) for aid, a in ep.current_action_map.items() if a["type"] == "LOOKUP"
    )
    used_eid = lookup_action["eid"]
    ep.step(lookup_id)
    # the just-looked-up entity must not be offered as a LOOKUP target again
    reoffered = [a for a in ep.current_action_map.values() if a["type"] == "LOOKUP" and a["eid"] == used_eid]
    assert not reoffered
    assert used_eid in ep._looked_up_eids


def test_v3_lookup_targets_never_bad(v3_env):
    ep = v3_env()
    ep.step("INIT")
    select_id = next(aid for aid, a in ep.current_action_map.items() if a["type"] == "SELECT")
    ep.step(select_id)
    for action in ep.current_action_map.values():
        if action["type"] == "LOOKUP":
            assert not ep._is_bad_lookup_target(ep.entities[action["eid"]])
            assert not is_bad_lookup_target(action.get("entity_surface", ""), ep.entities[action["eid"]].get("label", ""))


def test_v3_answer_with_harvests_and_terminates(v3_env):
    ep = v3_env()
    ep.step("INIT")
    aw_id, aw_action = next(
        (aid, a) for aid, a in ep.current_action_map.items() if a["type"] == "ANSWER_WITH"
    )
    target_sid = aw_action["sids"][0]
    obs = ep.step(aw_id)
    assert ep.stopped
    assert target_sid in ep.selected_evidence
    assert "selected_evidence:" in obs
    assert "Now provide the final answer" in obs
    assert ep.metrics.answer_with_count == 1
    # episode is terminal: further steps echo the stopped observation
    assert "event: ANSWER" in ep.step("A0")



def test_v3_answer_action_terminates(v3_env):
    ep = v3_env()
    ep.step("INIT")
    select_id = next(aid for aid, a in ep.current_action_map.items() if a["type"] == "SELECT")
    ep.step(select_id)
    answer_id = next(aid for aid, a in ep.current_action_map.items() if a["type"] == "ANSWER")
    obs = ep.step(answer_id)
    assert ep.stopped
    assert ep.metrics.answer_count == 1
    assert "Now provide the final answer" in obs


# ---------------------------------------------------------------------------
# graph_index.lookup_entity union
# ---------------------------------------------------------------------------
def test_lookup_entity_union_reaches_source_neighbor():
    """LOOKUP candidate pool is graph-only: target mentions + their sentence-graph
    neighbors. The anchor entity's neighbors are NOT merged into the pool (the
    anchor is retained as metadata only)."""
    idx = HarnessGGraphIndex()
    idx.entities = {"e_src": {"canonical": "Source", "surface_forms": ["Source"]},
                    "e_tgt": {"canonical": "Target", "surface_forms": ["Target"]}}
    idx.sentences = {
        "s_src": {"sid": "s_src", "title": "Src", "text": "Source meets Target", "pid": "p1"},
        "s_nb": {"sid": "s_nb", "title": "Br", "text": "the bridge fact about Target", "pid": "p2"},
        "s_tgt": {"sid": "s_tgt", "title": "Tg", "text": "Target detail", "pid": "p3"},
    }
    idx.entity_to_sentences = {"e_src": ["s_src"], "e_tgt": ["s_tgt"]}
    idx.sentence_to_entities = {"s_src": ["e_src"], "s_nb": [], "s_tgt": ["e_tgt"]}
    idx.sentence_to_neighbor_sentences = {"s_src": ["s_nb"]}

    expand_cand, _, _ = idx._expand_candidate_sids("e_tgt")
    assert "s_nb" not in expand_cand

    rows = idx.lookup_entity("e_tgt", "Target bridge mixquery", topk=5, anchor_eid="e_src")
    sids = {row["sid"] for row in rows}
    assert "s_tgt" in sids  # target mention always reachable
    # anchor-only neighbor s_nb is NOT merged: it is reachable only via the
    # anchor entity's own LOOKUP, not via the target's.
    assert "s_nb" not in sids


def test_lookup_entity_superset_of_expand(v3_env):
    ep = v3_env()
    ep.step("INIT")
    eid = next(iter(ep.entities))
    expand_sids = {r["sid"] for r in ep.graph_index.expand_entity(eid, "q", f"{eid} q", topk=20)}
    # LOOKUP shares the same graph candidate pool as EXPAND; with topk=20 over a
    # small corpus both return the whole pool, so expand_sids is a subset.
    lookup_sids = {r["sid"] for r in ep.graph_index.lookup_entity(eid, f"{eid} q", topk=20)}
    assert expand_sids <= lookup_sids


# ---------------------------------------------------------------------------
# SNC previews
# ---------------------------------------------------------------------------
def test_v3_snc_preview_lookup_and_answer_with(v3_env):
    ep = v3_env()
    ep.step("INIT")
    select_id = next(aid for aid, a in ep.current_action_map.items() if a["type"] == "SELECT")
    ep.step(select_id)

    lookup_id = next(aid for aid, a in ep.current_action_map.items() if a["type"] == "LOOKUP")
    lookup_preview = preview_candidate(ep, lookup_id)
    assert ep.current_action_map[lookup_id]["eid"] in lookup_preview["used_entity_ids"]

    aw_id = next((aid for aid, a in ep.current_action_map.items() if a["type"] == "ANSWER_WITH"), None)
    if aw_id is not None:
        aw_preview = preview_candidate(ep, aw_id)
        assert aw_preview["surfaced_sids"] == ep.current_action_map[aw_id]["sids"]
        assert aw_preview["used_entity_ids"] == frozenset()

    answer_id = next(aid for aid, a in ep.current_action_map.items() if a["type"] == "ANSWER")
    answer_preview = preview_candidate(ep, answer_id)
    assert answer_preview["surfaced_sids"] == []  # answering surfaces no new evidence (zero IG)


# ---------------------------------------------------------------------------
# Reward bridge
# ---------------------------------------------------------------------------
def test_reward_answer_with_is_select_and_stop():
    rollout = """
<|im_start|>assistant
<query>{"query": "INIT"}</query><|im_end|>
<|im_start|>user
[HARNESS_G_OBS]
available_actions:
A0 = SELECT S0
A1 = LOOKUP E0 | entity: Ada Lovelace | via: from S0
[/HARNESS_G_OBS]<|im_end|>
<|im_start|>assistant
<query>{"query": "A1"}</query><|im_end|>
<|im_start|>user
[HARNESS_G_OBS]
event: LOOKUP
new_visible_sentences:
S1 | Ada Lovelace was born in London.
available_actions:
A0 = SELECT S1
A1 = ANSWER_WITH S1 | evidence: Ada Lovelace was born in London.
A2 = ANSWER
[/HARNESS_G_OBS]<|im_end|>
<|im_start|>assistant
<query>{"query": "A1"}</query><|im_end|>
<|im_start|>user
[HARNESS_G_OBS]
event: ANSWER_WITH
selected_evidence:
S1 | Ada Lovelace was born in London.
[/HARNESS_G_OBS]<|im_end|>
<|im_start|>assistant
<answer>London</answer><|im_end|>
"""
    metrics = analyze(rollout, ["London"])
    assert metrics["lookup_count"] == 1
    assert metrics["answer_with_count"] == 1
    assert metrics["answer_after_stop"] == 1
    assert metrics["qa_reward_eligible"] == 1
    assert metrics["selected_coverage"] > 0.0
    assert metrics["answer_f1"] == 1.0
    assert metrics["total_reward"] > 0.0
