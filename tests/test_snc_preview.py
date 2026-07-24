from __future__ import annotations

from copy import deepcopy
from types import SimpleNamespace

from harness_g.snc_preview import compact_keywords, preview_candidate, preview_frontier


class FakeGraphIndex:
    def __init__(self):
        self.sentinel = {"committed": False}
        self.bridge_calls = []
        self.expand_calls = []
        self.lookup_calls = []
        self.context_calls = []
        self.rewrite_calls = []
        self.entity_calls = []
        self.entities_by_sid = {
            "bridge-1": [{"eid": "bridge-entity-1"}],
            "bridge-2": [{"eid": "bridge-entity-2"}],
            "expand-1": [{"eid": "expand-entity"}],
            "select-1": [{"eid": "select-entity"}],
            "context-1": [{"eid": "context-entity"}],
            "rewrite-1": [{"eid": "rewrite-entity"}],
            "lookup-q1": [{"eid": "lookup-entity-q1"}],
            "lookup-q2": [{"eid": "lookup-entity-q2"}],
            "lookup-default": [{"eid": "lookup-entity-default"}],
        }

    def get_local_context(self, sid):
        self.context_calls.append(sid)
        return [{"sid": "context-1"}]

    def bridge_entity(
        self,
        source_eid,
        target_eid,
        question,
        selected_evidence,
        bridge_query=None,
        topk=None,
    ):
        self.bridge_calls.append(
            (source_eid, target_eid, question, selected_evidence, bridge_query, topk)
        )
        return [{"sid": "bridge-1"}, {"sid": "bridge-2"}]

    def expand_entity(self, eid, q0, qh, topk=None):
        self.expand_calls.append((eid, q0, qh, topk))
        return [{"sid": "expand-1"}]

    def lookup_entity(self, target_eid, mixquery, topk=6, anchor_eid=None):
        self.lookup_calls.append(
            {"target_eid": target_eid, "mixquery": mixquery, "topk": topk,
             "anchor_eid": anchor_eid}
        )
        # Return different sids depending on the mixquery so tests can detect
        # whether the env-built mixquery actually reached lookup_entity.
        if "selected" in mixquery or "already" in mixquery:
            return [{"sid": "lookup-mix"}]
        return [{"sid": "lookup-default"}]

    def hybrid_initial_retrieve(
        self,
        query,
        *,
        paragraph_topk,
        high_conf_chunk_k,
        sentence_topk,
        entity_topk,
        topk,
    ):
        self.rewrite_calls.append(
            {
                "query": query,
                "paragraph_topk": paragraph_topk,
                "high_conf_chunk_k": high_conf_chunk_k,
                "sentence_topk": sentence_topk,
                "entity_topk": entity_topk,
                "topk": topk,
            }
        )
        return [{"sid": "rewrite-1"}]

    def get_entities_for_sentence(self, sid):
        self.entity_calls.append(sid)
        return self.entities_by_sid.get(sid, [])


def make_episode():
    return SimpleNamespace(
        graph_index=FakeGraphIndex(),
        question="Where did Ada Lovelace publish notes about the Analytical Engine?",
        selected_evidence=[{"sid": "selected-1", "text": "already selected"}],
        visible_sentence_k=3,
        paragraph_topk=2,
        high_conf_chunk_k=2,
        bridge_entity_topm=2,
        expanded_visible_sentence_k=2,
        qh_max_words=6,
        current_action_map={
            "bridge": {
                "type": "BRIDGE_ENTITY",
                "source_eid": "src",
                "target_eid": "tgt",
            },
            "expand": {
                "type": "EXPAND_ENTITY",
                "eid": "ada",
                "expanded_entity": "Ada Lovelace",
            },
            "select": {"type": "SELECT", "sid": "select-1"},
            "stop": {"type": "STOP"},
            "open": {"type": "OPEN_CONTEXT", "sid": "select-1"},
            "rewrite": {"type": "REWRITE_QUERY"},
        },
    )




def test_select_preview_returns_single_sid():
    episode = make_episode()

    preview = preview_candidate(episode, "select")

    assert preview["surfaced_sids"] == ["select-1"]
    assert preview["used_entity_ids"] == frozenset()
    assert preview["surfaced_entity_ids"] == frozenset({"select-entity"})


def test_stop_preview_returns_empty_sets():
    episode = make_episode()

    preview = preview_candidate(episode, "stop")

    assert preview == {
        "surfaced_sids": [],
        "surfaced_entity_ids": frozenset(),
        "used_entity_ids": frozenset(),
    }


def test_preview_frontier_does_not_mutate_episode_or_graph_index_state():
    episode = make_episode()
    action_map_before = deepcopy(episode.current_action_map)
    selected_evidence_before = deepcopy(episode.selected_evidence)
    graph_sentinel_before = deepcopy(episode.graph_index.sentinel)

    previews = preview_frontier(episode)

    assert set(previews) == set(episode.current_action_map)
    assert episode.current_action_map == action_map_before
    assert episode.selected_evidence == selected_evidence_before
    assert episode.graph_index.sentinel == graph_sentinel_before
    assert episode.graph_index.bridge_calls == []
    assert episode.graph_index.expand_calls == []
    assert episode.graph_index.context_calls == []
    assert episode.graph_index.rewrite_calls == []


# ---------------------------------------------------------------------------
# Fix B tests: hop_query override must reach graph_index.lookup_entity and
# the constructed qh must match env._lookup (surface prefix always present).
# ---------------------------------------------------------------------------


def _make_lookup_episode():
    """Build an episode whose action_map has a LOOKUP action.

    The V3 LOOKUP no longer carries a ``hop_query`` (the env builds mixquery),
    so there is no action-map hop_query to seed.
    """
    episode = make_episode()
    episode.current_action_map["lookup"] = {
        "type": "LOOKUP",
        "eid": "ada",
        "expanded_entity": "Ada Lovelace",
        "entity_surface": "Ada Lovelace",
        "anchor_eid": None,
    }
    return episode


def test_preview_lookup_passes_env_mixquery_to_lookup_entity():
    """B1: the LOOKUP preview must call ``graph_index.lookup_entity`` with the
    env-built mixquery (question + selected evidence text), not a
    surface+question-keyword qh. ``hop_query_override`` is ignored for LOOKUP
    because the env no longer reads a model-written need_query."""
    episode = _make_lookup_episode()
    episode.qh_max_words = 16

    preview_with_override = preview_candidate(episode, "lookup", hop_query_override="place of death")
    preview_no_override = preview_candidate(episode, "lookup")

    # Both calls must reach lookup_entity with the SAME mixquery (override
    # ignored), and both must surface the mix-query result.
    assert preview_with_override["surfaced_sids"] == ["lookup-mix"]
    assert preview_no_override["surfaced_sids"] == ["lookup-mix"]
    assert len(episode.graph_index.lookup_calls) == 2
    assert episode.graph_index.lookup_calls[0]["mixquery"] == episode.graph_index.lookup_calls[1]["mixquery"]


def test_preview_lookup_mixquery_matches_env_mixquery_construction():
    """B2: the mixquery passed to ``graph_index.lookup_entity`` must equal what
    ``env._lookup`` produces — the full question plus SELECTed evidence text
    filling the rest of the ``mixquery_max_words`` budget. A preview that built
    a different query would retrieve a different sentence set than ``env.step``
    and silently corrupt taken_ig. This is the central Fix B correctness
    assertion for the mixquery path."""
    episode = _make_lookup_episode()
    episode.mixquery_max_words = 64

    preview_candidate(episode, "lookup")

    assert len(episode.graph_index.lookup_calls) == 1
    call = episode.graph_index.lookup_calls[0]
    # env._lookup_mixquery: full question + evidence within the word budget.
    question_words = episode.question.split()
    evidence_words = "already selected".split()[: max(64 - len(question_words), 0)]
    expected = " ".join(question_words + evidence_words)
    assert call["mixquery"] == expected
    assert call["anchor_eid"] is None


def test_preview_lookup_mixquery_budget_never_truncates_question():
    """The mixquery word budget caps only the evidence suffix: the question
    survives in full even when the budget is barely larger than the question,
    and the evidence is cut to whatever room is left."""
    episode = _make_lookup_episode()
    question_words = episode.question.split()
    episode.mixquery_max_words = len(question_words) + 1

    preview_candidate(episode, "lookup")

    call = episode.graph_index.lookup_calls[0]
    words = call["mixquery"].split()
    assert words[: len(question_words)] == question_words
    assert words[len(question_words):] == ["already"]


def test_preview_lookup_mixquery_falls_back_to_question_without_selected():
    """B2 complement: with no selected evidence, mixquery falls back to the
    full question (never truncated). Ensures the preview matches env._lookup
    when LOOKUP runs before any SELECT (e.g. straight after INIT)."""
    episode = _make_lookup_episode()
    episode.selected_evidence = []

    preview_candidate(episode, "lookup")

    call = episode.graph_index.lookup_calls[0]
    expected = " ".join(episode.question.split())
    assert call["mixquery"] == expected
