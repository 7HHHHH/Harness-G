from __future__ import annotations

from copy import deepcopy
from types import SimpleNamespace

from harness_g import snc_api
from harness_g.snc_api import build_snc_step_payload


class FakeGraphIndex:
    def __init__(self) -> None:
        self.sentences = {
            "S0": "Existing selected evidence.",
            "S1": "Bridge evidence.",
            "S2": "Expanded evidence.",
            "S3": "Selectable evidence.",
        }

    def get_entities_for_sentence(self, sid: str) -> list[str]:
        return {
            "S0": ["E0"],
            "S1": ["E1", "E2"],
            "S2": ["E1", "E3"],
            "S3": ["E4"],
        }.get(sid, [])

    def bridge_entity(self, source_eid: str, target_eid: str, **_: object) -> list[dict[str, str]]:
        return [{"sid": "S1", "source_eid": source_eid, "target_eid": target_eid}]

    def expand_entity(self, eid: str, **_: object) -> list[dict[str, str]]:
        return [{"sid": "S2", "eid": eid}]


def make_episode() -> SimpleNamespace:
    return SimpleNamespace(
        initialized=True,
        current_action_map={
            "A0": {
                "type": "BRIDGE_ENTITY",
                "source_eid": "E1",
                "target_eid": "E2",
                "score": 0.4,
            },
            "A1": {
                "type": "EXPAND_ENTITY",
                "eid": "E1",
                "expanded_entity": "Entity One",
                "score": 0.9,
            },
            "A2": {
                "type": "SELECT",
                "sid": "S3",
                "score": 0.1,
            },
        },
        selected_evidence=["S0"],
        snc_seen_sids=["S0", "S3"],
        question="Which answer is supported?",
        graph_index=FakeGraphIndex(),
        qh_max_words=16,
        bridge_k=2,
        expand_k=2,
        select_k=2,
        evidence_k=2,
        frontier_k=2,
        step=lambda _: (_ for _ in ()).throw(AssertionError("step must not be called")),
    )


def test_build_snc_step_payload_keeps_taken_separately_from_frontier(
    monkeypatch,
) -> None:
    """Fix B/C regression guard: taken is previewed separately and never
    appears in frontier. Old behaviour put taken in frontier and copied its
    default-query preview; that silently corrupted taken_ig once hop_query
    overrides were introduced."""
    monkeypatch.setenv("HARNESS_G_SNC_FRONTIER_TOPK", "2")
    monkeypatch.delenv("HARNESS_G_SNC_FRONTIER_STRATIFY", raising=False)

    episode = make_episode()
    selected_before = list(episode.selected_evidence)
    action_map_before = deepcopy(episode.current_action_map)

    def fake_parse_harness_g_action(raw_action: str, **_: object) -> SimpleNamespace:
        return SimpleNamespace(valid=True, action_id=raw_action)

    # Must accept **kwargs because build_snc_step_payload now calls
    # preview_candidate with hop_query_override=/bridge_query_override= for
    # the taken action. Frontier calls omit them.
    seen_override_calls: list[dict] = []
    seen_frontier_calls: list[str] = []

    def fake_preview_candidate(_: SimpleNamespace, action_id: str, **kwargs) -> dict[str, frozenset[str]]:
        if kwargs:
            seen_override_calls.append({"action_id": action_id, **kwargs})
        else:
            seen_frontier_calls.append(action_id)
        previews = {
            "A0": {
                "surfaced_sids": frozenset({"S1"}),
                "surfaced_entity_ids": frozenset({"E1", "E2"}),
                "used_entity_ids": frozenset({"E1", "E2"}),
            },
            "A1": {
                "surfaced_sids": frozenset({"S2"}),
                "surfaced_entity_ids": frozenset({"E1", "E3"}),
                "used_entity_ids": frozenset({"E1"}),
            },
            "A2": {
                "surfaced_sids": frozenset({"S3"}),
                # SELECT surfaces S3 which mentions E4, but queries no entity
                # itself — this is the enabling-signal case Fix A fixes.
                "surfaced_entity_ids": frozenset({"E4"}),
                "used_entity_ids": frozenset(),
            },
        }
        return previews[action_id]

    def fake_evidence_context_sids(selected_sids: list[str], surfaced_sids: list[str]) -> list[str]:
        return list(selected_sids) + list(surfaced_sids)

    def fake_build_answer_scoring_prompt(
        question: str,
        sids: list[str],
        graph_index: FakeGraphIndex,
    ) -> str:
        assert graph_index is episode.graph_index
        return f"{question} :: {'|'.join(sids)}"

    monkeypatch.setattr(snc_api, "parse_harness_g_action", fake_parse_harness_g_action)
    monkeypatch.setattr(snc_api, "preview_candidate", fake_preview_candidate)
    monkeypatch.setattr(snc_api, "build_answer_scoring_prompt", fake_build_answer_scoring_prompt)

    payload = build_snc_step_payload(episode, "A2")

    assert payload is not None
    assert payload["selected_sids_before"] == ["S0"]
    assert payload["seen_sids_before"] == ["S0", "S3"]
    assert payload["baseline_prompt"] == "Which answer is supported? :: S0|S3"
    assert payload["taken_action_id"] == "A2"
    # Fix B/C: taken must NOT be a frontier key.
    assert "A2" not in payload["frontier"]
    # Commit actions have no evidence frontier; they receive provenance/outcome
    # credit rather than being compared against retrieval actions.
    assert payload["frontier"] == {}
    # taken is built separately with action_id/action_type preserved.
    assert payload["taken"]["action_id"] == "A2"
    assert payload["taken"]["action_type"] == "SELECT"
    assert payload["taken"]["surfaced_sids"] == ["S3"]
    assert payload["taken"]["new_sids"] == []
    assert payload["taken"]["produced_sids"] == ["S3"]
    assert payload["taken"]["consumed_sids"] == ["S3"]
    assert payload["taken"]["is_information_action"] is False
    assert payload["taken"]["answer_prompt"] == payload["baseline_prompt"]
    # taken preview call must pass override slots (None here since raw has no ||);
    # frontier calls must not pass any override.
    assert seen_override_calls == [{"action_id": "A2", "hop_query_override": None, "bridge_query_override": None}]
    assert seen_frontier_calls == []
    assert episode.selected_evidence == selected_before
    assert episode.current_action_map == action_map_before



def test_build_snc_step_payload_invalid_action_returns_none(monkeypatch) -> None:

    assert build_snc_step_payload(make_episode(), "A2 || invalid select query") is None


# ---------------------------------------------------------------------------
# Fix B tests: taken action must be previewed with parsed hop_query and
# kept out of frontier so the trainer cannot reuse the frontier's
# default-query prompt score for the taken ref.
# ---------------------------------------------------------------------------


def _make_v3_episode() -> SimpleNamespace:
    """V3-style action map: SELECT / ANSWER_WITH / LOOKUP / ANSWER, no scores.

    Mirrors the real V3 menu (env.py _add_v3_actions) where none of the
    actions carry a ``score`` field, which is what triggers the Fix C
    insertion-order degeneration.
    """
    return SimpleNamespace(
        initialized=True,
        current_action_map={
            "A0": {"type": "SELECT", "sid": "S3"},
            "A1": {"type": "ANSWER_WITH", "sids": ["S3"], "sid": "S3"},
            "A2": {"type": "LOOKUP", "eid": "E5", "expanded_entity": "Entity Five",
                   "entity_surface": "Entity Five", "anchor_eid": None},
            "A3": {"type": "ANSWER"},
        },
        selected_evidence=["S0"],
        snc_seen_sids=["S0", "S3"],
        question="Which answer is supported?",
        graph_index=FakeGraphIndex(),
        qh_max_words=16,
        bridge_k=2,
        expand_k=2,
        select_k=2,
        evidence_k=2,
        frontier_k=2,
        step=lambda _: (_ for _ in ()).throw(AssertionError("step must not be called")),
    )


def _fake_parse_v3(raw_action: str, **_: object) -> dict:
    """Parse ``A2 || my query`` into a real-protocol-shaped dict."""
    raw = (raw_action or "").strip()
    if "||" in raw:
        aid, query = raw.split("||", 1)
        return {
            "is_valid": True,
            "action_id": aid.strip(),
            "hop_query": query.strip(),
            "semantic_action": None,
            "invalid_reason": None,
        }
    return {
        "is_valid": True,
        "action_id": raw,
        "hop_query": None,
        "semantic_action": None,
        "invalid_reason": None,
    }


def _install_v3_fakes(monkeypatch, episode) -> dict:
    """Install parse/preview/scoring fakes that record override usage.

    Returns a dict of recording lists so the test can assert on them.
    """
    preview_calls: list[dict] = {"taken": [], "frontier": []}

    def fake_preview_candidate(_, action_id, **kwargs):
        if kwargs:
            preview_calls["taken"].append({"action_id": action_id, **kwargs})
        else:
            preview_calls["frontier"].append(action_id)
        # Return different surfaced sids depending on whether a hop_query
        # override was passed, so taken vs frontier prompts diverge and a
        # collision would be detectable.
        if kwargs.get("hop_query_override") == "my query":
            sids = ["S-override"]
        else:
            sids = [f"S-{action_id}"]
        return {
            "surfaced_sids": sids,
            "surfaced_entity_ids": frozenset(),
            "used_entity_ids": frozenset(),
        }

    def fake_evidence_context_sids(selected_sids, surfaced_sids):
        return list(selected_sids) + list(surfaced_sids)

    def fake_build_answer_scoring_prompt(question, sids, graph_index):
        return f"{question} :: {'|'.join(sids)}"

    monkeypatch.setattr(snc_api, "parse_harness_g_action", _fake_parse_v3)
    monkeypatch.setattr(snc_api, "preview_candidate", fake_preview_candidate)
    monkeypatch.setattr(snc_api, "build_answer_scoring_prompt", fake_build_answer_scoring_prompt)
    return preview_calls


def test_build_snc_step_payload_taken_uses_parsed_hop_query(monkeypatch) -> None:
    """B3: ``A2 || my query`` must drive the taken preview with
    hop_query_override='my query', and the taken answer_prompt must come from
    the override-driven surfaced sids, not the action-map default query."""
    monkeypatch.setenv("HARNESS_G_SNC_FRONTIER_TOPK", "3")
    monkeypatch.delenv("HARNESS_G_SNC_FRONTIER_STRATIFY", raising=False)
    episode = _make_v3_episode()
    calls = _install_v3_fakes(monkeypatch, episode)

    payload = build_snc_step_payload(episode, "A2 || my query")

    assert payload is not None
    assert payload["taken_action_id"] == "A2"
    # taken previewed with the parsed hop_query override.
    assert calls["taken"] == [
        {"action_id": "A2", "hop_query_override": "my query", "bridge_query_override": "my query"}
    ]
    # frontier candidates previewed WITHOUT any override (not polluted).
    for call_entry in calls["frontier"]:
        assert call_entry != "A2"  # taken excluded
    # taken answer_prompt built from the override-driven surfaced sids.
    assert payload["taken"]["answer_prompt"] == "Which answer is supported? :: S0|S3|S-override"
    assert payload["taken"]["surfaced_sids"] == ["S-override"]


def test_build_snc_step_payload_frontier_candidates_use_default_query(monkeypatch) -> None:
    """B4: the taken hop_query must not leak into frontier candidate
    previews. Frontier LOOKUP candidates are previewed with no override."""
    monkeypatch.setenv("HARNESS_G_SNC_FRONTIER_TOPK", "3")
    monkeypatch.delenv("HARNESS_G_SNC_FRONTIER_STRATIFY", raising=False)
    episode = _make_v3_episode()
    calls = _install_v3_fakes(monkeypatch, episode)

    build_snc_step_payload(episode, "A2 || my query")

    # Every frontier call recorded no kwargs (empty kwargs -> recorded as id).
    assert all(isinstance(c, str) for c in calls["frontier"])
    # taken never appears among frontier calls.
    assert "A2" not in calls["frontier"]


def test_build_snc_step_payload_taken_excluded_from_frontier_prevents_score_ref_collision(
    monkeypatch,
) -> None:
    """B5: taken_action_id must NOT be a key in payload['frontier']. The
    trainer (snc_trainer.py:246-251) keys score refs by (sample, step,
    action_id); if taken shared a frontier id, the trainer would reuse the
    frontier's default-query prompt score for taken_ig and silently undo
    Fix B. This is the regression guard for that invariant."""
    monkeypatch.setenv("HARNESS_G_SNC_FRONTIER_TOPK", "4")
    monkeypatch.delenv("HARNESS_G_SNC_FRONTIER_STRATIFY", raising=False)
    episode = _make_v3_episode()
    _install_v3_fakes(monkeypatch, episode)

    payload = build_snc_step_payload(episode, "A2 || my query")

    assert payload is not None
    assert payload["taken_action_id"] == "A2"
    assert "A2" not in payload["frontier"]
    # Stronger: taken answer_prompt differs from any frontier candidate's,
    # so even if ids collided the scores would diverge — but ids must not
    # collide in the first place.
    taken_prompt = payload["taken"]["answer_prompt"]
    frontier_prompts = {c["answer_prompt"] for c in payload["frontier"].values()}
    assert taken_prompt not in frontier_prompts


# ---------------------------------------------------------------------------
# Fix C tests: stratified frontier must cover representative action types
# instead of degenerating to the first few menu entries (all SELECT) when V3
# actions carry no score. taken is always excluded regardless of stratify flag.
# ---------------------------------------------------------------------------


def _v3_action_map_no_scores() -> dict:
    """Insertion order: SELECT, SELECT, SELECT, SELECT, ANSWER_WITH, LOOKUP,
    ANSWER — mirrors the real V3 _add_v3_actions order (env.py:533-591)."""
    return {
        "A0": {"type": "SELECT", "sid": "S1"},
        "A1": {"type": "SELECT", "sid": "S2"},
        "A2": {"type": "SELECT", "sid": "S3"},
        "A3": {"type": "SELECT", "sid": "S4"},
        "A4": {"type": "ANSWER_WITH", "sids": ["S1"]},
        "A5": {"type": "LOOKUP", "eid": "E5", "expanded_entity": "Entity Five"},
        "A6": {"type": "ANSWER"},
    }


def test_frontier_stratified_contains_information_actions_only(monkeypatch) -> None:
    """The evidence frontier excludes commit/terminal actions whose scorer
    context cannot represent their transition semantics."""
    monkeypatch.setenv("HARNESS_G_SNC_FRONTIER_STRATIFY", "1")
    from harness_g.snc_api import _choose_frontier_action_ids

    chosen = _choose_frontier_action_ids(_v3_action_map_no_scores(), taken_action_id=None, limit=4)

    assert chosen == ["A5"]


def test_frontier_stratified_excludes_taken(monkeypatch) -> None:
    """C2: when taken is a LOOKUP and there is only one LOOKUP, the frontier
    must not contain it. taken exclusion is a Fix B invariant and must hold
    under stratification."""
    monkeypatch.setenv("HARNESS_G_SNC_FRONTIER_STRATIFY", "1")
    from harness_g.snc_api import _choose_frontier_action_ids

    chosen = _choose_frontier_action_ids(
        _v3_action_map_no_scores(), taken_action_id="A5", limit=4
    )

    assert "A5" not in chosen
    assert chosen == []


def test_frontier_stratify_disabled_uses_ranked_order_but_excludes_taken(monkeypatch) -> None:
    """C3: with stratify off, frontier falls back to ranked (insertion) order
    but STILL excludes taken. This is the ablation path; it must not restore
    the taken-in-frontier bug."""
    monkeypatch.setenv("HARNESS_G_SNC_FRONTIER_STRATIFY", "0")
    from harness_g.snc_api import _choose_frontier_action_ids

    action_map = _v3_action_map_no_scores()
    chosen = _choose_frontier_action_ids(action_map, taken_action_id="A0", limit=4)

    # Commit/terminal actions remain ineligible even on the unstratified
    # ablation path.
    assert chosen == ["A5"]
    assert "A0" not in chosen


def test_frontier_within_type_respects_score(monkeypatch) -> None:
    """C4: when a type bucket has multiple actions with scores, stratification
    picks the highest-scored one first (delegates to _ranked_action_items)."""
    monkeypatch.setenv("HARNESS_G_SNC_FRONTIER_STRATIFY", "1")
    from harness_g.snc_api import _choose_frontier_action_ids

    action_map = {
        "A0": {"type": "LOOKUP", "eid": "E1", "score": 0.1},
        "A1": {"type": "LOOKUP", "eid": "E2", "score": 0.9},
        "A2": {"type": "SELECT", "sid": "S1"},
    }
    chosen = _choose_frontier_action_ids(action_map, taken_action_id=None, limit=2)

    # LOOKUP bucket: A1 (0.9) beats A0 (0.1); SELECT is ineligible.
    assert chosen[0] == "A1"
    assert chosen == ["A1", "A0"]


def test_frontier_stratified_fill_ignores_commit_actions(monkeypatch) -> None:
    monkeypatch.setenv("HARNESS_G_SNC_FRONTIER_STRATIFY", "1")
    from harness_g.snc_api import _choose_frontier_action_ids

    action_map = {
        "A0": {"type": "SELECT", "sid": "S1"},
        "A1": {"type": "SELECT", "sid": "S2"},
        "A2": {"type": "SELECT", "sid": "S3"},
        "A3": {"type": "LOOKUP", "eid": "E5"},
    }
    chosen = _choose_frontier_action_ids(action_map, taken_action_id=None, limit=4)

    assert chosen == ["A3"]


def test_payload_dedupes_equivalent_preview_contexts_and_backfills(monkeypatch) -> None:
    monkeypatch.setenv("HARNESS_G_SNC_FRONTIER_TOPK", "2")
    episode = SimpleNamespace(
        initialized=True,
        current_action_map={
            "A0": {"type": "LOOKUP", "eid": "E0"},
            "A1": {"type": "LOOKUP", "eid": "E1"},
            "A2": {"type": "LOOKUP", "eid": "E2"},
            "A3": {"type": "LOOKUP", "eid": "E3"},
        },
        selected_evidence=[],
        snc_seen_sids=["S0"],
        question="Q?",
        graph_index=FakeGraphIndex(),
        qh_max_words=16,
    )

    def preview(_, action_id, **kwargs):
        sid = {"A0": "S-taken", "A1": "S1", "A2": "S1", "A3": "S2"}[action_id]
        return {
            "surfaced_sids": [sid],
            "surfaced_entity_ids": [],
            "used_entity_ids": [f"E-{action_id}"],
        }

    monkeypatch.setattr(snc_api, "parse_harness_g_action", _fake_parse_v3)
    monkeypatch.setattr(snc_api, "preview_candidate", preview)
    monkeypatch.setattr(
        snc_api,
        "build_answer_scoring_prompt",
        lambda question, sids, graph_index: f"{question} :: {'|'.join(sids)}",
    )

    payload = build_snc_step_payload(episode, "A0")

    assert list(payload["frontier"]) == ["A1", "A3"]
    assert payload["frontier_duplicate_contexts_skipped"] == 1
    assert len({item["answer_prompt"] for item in payload["frontier"].values()}) == 2
