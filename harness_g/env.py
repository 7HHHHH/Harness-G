import uuid
from typing import Dict, List, Optional, Set

from .formatting import actions_lines, entity_line, observation_block, selected_line, sentence_line
from .graph_index import HarnessGGraphIndex
from .metrics import HarnessGEpisodeMetrics
from .protocol import parse_harness_g_action
from .utils import is_bad_lookup_target

# Action-menu budgets for the discrete navigation action space: the post-SELECT
# menu offers up to SELECT_LOOKUP_K LOOKUP targets, post-navigation menus offer
# up to NAV_LOOKUP_K.
SELECT_LOOKUP_K = 8
NAV_LOOKUP_K = 4

# LOOKUP mixquery budget. Separate from qh_max_words (which caps model-written
# short queries): mixquery must hold the full question plus SELECTed evidence
# text, so 16 words would truncate even the question itself on longer
# comparison questions.
MIXQUERY_MAX_WORDS = 64


class HarnessGEpisode:
    def __init__(
        self,
        session_id: Optional[str],
        question: str,
        graph_index: HarnessGGraphIndex,
        paragraph_topk: int = 20,
        high_conf_chunk_k: int = 5,
        visible_sentence_k: int = 6,
        expanded_visible_sentence_k: int = 6,
        max_turns: int = 5,
        qh_max_words: int = 16,
        bridge_entity_topm: int = 5,
        mixquery_max_words: Optional[int] = None,
    ) -> None:
        self.session_id = session_id or str(uuid.uuid4())
        self.question = question or ""
        self.graph_index = graph_index
        self.paragraph_topk = paragraph_topk
        self.high_conf_chunk_k = high_conf_chunk_k
        self.visible_sentence_k = visible_sentence_k
        self.expanded_visible_sentence_k = expanded_visible_sentence_k
        self.max_turns = max_turns
        self.qh_max_words = qh_max_words
        if mixquery_max_words is None:
            mixquery_max_words = MIXQUERY_MAX_WORDS
        self.mixquery_max_words = max(int(mixquery_max_words), 1)
        self.bridge_entity_topm = bridge_entity_topm

        self.initialized = False
        self.stopped = False
        self.step_count = 0
        self.selected_evidence: List[str] = []
        self.current_visible_sids: List[str] = []
        self.current_display_sentence_map: Dict[str, str] = {}
        self._real_sid_to_display: Dict[str, str] = {}
        self._next_sentence_display_idx = 0
        self.current_display_entity_map: Dict[str, str] = {}
        self.current_action_map: Dict[str, dict] = {}
        self.frontier_entities: List[str] = []
        self.metrics = HarnessGEpisodeMetrics()
        self._visible_history: List[str] = []
        # Stable first-observation order used by SNC's cumulative evidence
        # state.  ``_visible_history`` is an LRU-style menu pool and reorders a
        # SID whenever it reappears, so it cannot define C_t without creating
        # artificial score deltas from context reordering.
        self.snc_seen_sids: List[str] = []
        self._snc_seen_sid_set: Set[str] = set()
        # Dedup repeated LOOKUPs of the same (target entity, need-query) and
        # keep a lightweight lookup trace.
        self._seen_lookup_keys: Set[tuple] = set()
        self._looked_up_eids: Set[str] = set()
        self._lookup_history: List[dict] = []
        self._last_nav_event: Optional[dict] = None

    def _record_available_action_counts(self) -> None:
        self.metrics.available_action_count += len(self.current_action_map)
        for action in self.current_action_map.values():
            action_type = action.get("type")
            if action_type == "LOOKUP":
                self.metrics.lookup_action_available_count += 1
            elif action_type == "ANSWER_WITH":
                self.metrics.answer_with_action_available_count += 1
            elif action_type == "ANSWER":
                self.metrics.stop_action_available_count += 1

    def step(self, raw_action: str) -> str:
        action = (raw_action or "").strip()
        if self.stopped:
            return self._format_stopped_observation(event="ANSWER")

        if not action and not self.initialized:
            action = "INIT"

        parsed = parse_harness_g_action(
            action,
            action_map=self.current_action_map,
            initialized=self.initialized,
            qh_max_words=self.qh_max_words,
        )

        if not parsed["is_valid"]:
            self.metrics.invalid_action_count += 1
            if parsed.get("is_natural_query"):
                self.metrics.natural_query_count += 1
            return self._format_invalid_observation(parsed)

        self.metrics.valid_action_id_count += 1
        self.metrics.step_count += 1
        self._last_nav_event = None
        semantic_action = parsed["semantic_action"]

        if semantic_action == "INIT":
            return self._init()
        if semantic_action == "SELECT":
            return self._select(self.current_action_map[parsed["action_id"]]["sid"])
        if semantic_action == "LOOKUP":
            mapped = self.current_action_map[parsed["action_id"]]
            return self._lookup(
                mapped["eid"],
                via=mapped.get("via"),
                anchor_eid=mapped.get("anchor_eid"),
            )
        if semantic_action == "ANSWER_WITH":
            mapped = self.current_action_map[parsed["action_id"]]
            sids = mapped.get("sids") or ([mapped["sid"]] if mapped.get("sid") else [])
            return self._answer_with(sids)
        if semantic_action == "ANSWER":
            return self._stop(event="ANSWER")

        parsed["is_valid"] = False
        parsed["invalid_reason"] = f"unsupported semantic action: {semantic_action}"
        self.metrics.invalid_action_count += 1
        return self._format_invalid_observation(parsed)

    def _init(self) -> str:
        ranked = self.graph_index.hybrid_initial_retrieve(
            self.question,
            paragraph_topk=self.paragraph_topk,
            high_conf_chunk_k=self.high_conf_chunk_k,
            sentence_topk=max(self.visible_sentence_k, 8),
            entity_topk=max(self.bridge_entity_topm, 8),
            topk=self.visible_sentence_k,
        )
        return self.init_with_ranked(ranked)

    def init_with_ranked(self, ranked: List[dict]) -> str:
        self.initialized = True
        self.step_count += 1
        self.current_visible_sids = [sentence["sid"] for sentence in ranked]
        init_sids = list(self.current_visible_sids)
        self._last_nav_event = {
            "action_type": "INIT",
            "query_text": self.question,
            "result_ids": init_sids,
            "new_result_ids": init_sids,
            "turn": self.step_count,
        }
        self._set_sentence_actions(ranked, include_answer=False)
        return self._format_initial_observation(ranked)

    def _select(self, sid: str) -> str:
        self.step_count += 1
        self.metrics.select_count += 1
        if sid not in self.selected_evidence:
            self.selected_evidence.append(sid)
        self.metrics.selected_sentence_count = len(self.selected_evidence)
        self._last_nav_event = {
            "action_type": "SELECT",
            "query_text": None,
            "result_ids": [sid],
            "new_result_ids": [],
            "turn": self.step_count,
        }

        entities = self.graph_index.get_entities_for_sentence(sid)
        self.frontier_entities = [entity["eid"] for entity in entities]
        self.current_display_entity_map = {}
        for eid in self.frontier_entities:
            self._ensure_display_eid(eid)

        bridge_candidates = self.graph_index.propose_bridge_entities(
            self.frontier_entities,
            self.question,
            self.selected_evidence,
            topm=self.bridge_entity_topm,
        )

        self.current_action_map = {}
        self._add_v3_actions(
            0,
            visible_sids=self.current_visible_sids,
            lookup_specs=self._lookup_specs_from_select(sid, bridge_candidates),
            include_answer=True,
            lookup_cap=SELECT_LOOKUP_K,
        )
        self._record_available_action_counts()
        return self._format_select_observation(entities, bridge_candidates)

    @property
    def entities(self) -> Dict[str, dict]:
        return self.graph_index.entities

    def _stop(self, event: str = "ANSWER") -> str:
        self.step_count += 1
        self.metrics.stop_count += 1
        if event == "ANSWER":
            self.metrics.answer_count += 1
        self.stopped = True
        self.current_action_map = {}
        return self._format_stopped_observation(event=event)

    # ------------------------------------------------------------------
    # Discrete navigation menu (SELECT / LOOKUP / ANSWER_WITH / ANSWER)
    # ------------------------------------------------------------------
    def _is_bad_lookup_target(self, entity: dict) -> bool:
        return is_bad_lookup_target(self._entity_surface(entity), entity.get("label", ""))

    def _evidence_preview(self, sentence: dict, max_words: int = 18) -> str:
        text = str(sentence.get("text") or "").strip().strip('"').strip()
        words = text.split()
        if len(words) > max_words:
            text = " ".join(words[:max_words]) + " ..."
        return text

    def _lookup_specs_from_select(self, sid: str, bridge_candidates: List[dict]) -> List[tuple]:
        """LOOKUP targets for the post-SELECT menu: frontier entities of the
        selected sentence (anchor=None, pure expand) plus bridge candidates
        (anchor=source entity, so ``lookup_entity`` merges bridge provenance)."""
        via_sid = self._display_sid(sid)
        order: List[str] = []
        by_eid: Dict[str, dict] = {}

        def upsert(eid: str, anchor: Optional[str], via: str, low: bool) -> None:
            if not eid or eid not in self.entities:
                return
            if eid not in by_eid:
                by_eid[eid] = {"eid": eid, "anchor": anchor, "via": via, "low": low}
                order.append(eid)
            else:
                spec = by_eid[eid]
                if anchor and not spec["anchor"]:
                    spec["anchor"] = anchor
                spec["low"] = spec["low"] and low

        for eid in self.frontier_entities:
            upsert(eid, None, f"from {via_sid}", False)
        for candidate in bridge_candidates:
            target_eid = candidate.get("target_eid") or candidate.get("eid")
            source_eid = candidate.get("source_eid")
            if not target_eid or target_eid not in self.entities:
                continue
            anchor = source_eid if source_eid in self.entities else None
            via = f"bridge from {self._entity_surface(self.entities[anchor])}" if anchor else "bridge candidate"
            upsert(target_eid, anchor, via, True)
        return [(by_eid[e]["eid"], by_eid[e]["anchor"], by_eid[e]["via"], by_eid[e]["low"]) for e in order]

    def _lookup_specs_from_visible(self, ranked: List[dict]) -> List[tuple]:
        """LOOKUP targets for post-navigation menus: entities surfaced in the
        newly visible sentences (anchor=None, expand-style)."""
        specs: List[tuple] = []
        seen: Set[str] = set()
        for sentence in ranked:
            sid = sentence["sid"]
            via_sid = self._display_sid(sid)
            for entity in self.graph_index.get_entities_for_sentence(sid):
                eid = entity.get("eid")
                if not eid or eid in seen:
                    continue
                seen.add(eid)
                specs.append((eid, None, f"from {via_sid}", False))
        return specs

    def _add_v3_actions(
        self,
        next_action: int,
        visible_sids: List[str],
        lookup_specs: List[tuple],
        include_answer: bool,
        lookup_cap: int,
    ) -> int:
        """Build the action menu: SELECT + ANSWER_WITH per visible sentence,
        deduped/filtered LOOKUP targets, and (optionally) ANSWER."""
        visible_sids = list(dict.fromkeys(visible_sids))

        for sid in visible_sids:
            if sid in self.selected_evidence:
                continue
            self.current_action_map[f"A{next_action}"] = {
                "type": "SELECT",
                "sid": sid,
                "display_sid": self._display_sid(sid),
                "source": "visible_sentence",
            }
            next_action += 1

        for sid in visible_sids:
            if sid in self.selected_evidence:
                continue
            sentence = self.graph_index.sentences.get(sid, {})
            self.current_action_map[f"A{next_action}"] = {
                "type": "ANSWER_WITH",
                "sids": [sid],
                "display_sids": [self._display_sid(sid)],
                "display_sid": self._display_sid(sid),
                "evidence_preview": self._evidence_preview(sentence),
                "source": "visible_sentence",
            }
            next_action += 1

        seen_eids: Set[str] = set()
        added_lookups = 0
        for eid, anchor_eid, via, low_priority in lookup_specs:
            if added_lookups >= lookup_cap:
                break
            if not eid or eid in seen_eids or eid not in self.entities:
                continue
            # Dedup across the episode: never re-offer a LOOKUP for an entity
            # already looked up, so the policy cannot loop on no-op lookups.
            if eid in self._looked_up_eids:
                continue
            entity = self.entities[eid]
            if self._is_bad_lookup_target(entity):
                self.metrics.bad_target_lookup_count += 1
                continue
            seen_eids.add(eid)
            self.current_action_map[f"A{next_action}"] = {
                "type": "LOOKUP",
                "eid": eid,
                "display_eid": self._ensure_display_eid(eid),
                "entity_surface": self._entity_surface(entity),
                "anchor_eid": anchor_eid,
                "via": via,
                "low_priority": bool(low_priority),
                "expanded_entity": self._entity_surface(entity),
                "source": "lookup_target",
            }
            next_action += 1
            added_lookups += 1

        if include_answer:
            self.current_action_map[f"A{next_action}"] = {"type": "ANSWER"}
            next_action += 1
        return next_action

    def _lookup(self, eid: str, via: Optional[str] = None, anchor_eid: Optional[str] = None) -> str:
        self.step_count += 1
        self.metrics.lookup_count += 1
        entity = self.entities.get(eid, {"canonical": eid, "surface_forms": [eid]})
        # LOOKUP does not take a model-written ``|| need_query``. The retrieval
        # query is a single ``mixquery`` built from the question plus the text
        # of every sentence SELECTed so far, so the next hop is found by what
        # the agent has already gathered rather than by a free-form query the
        # small model cannot reliably write.
        mixquery = self._lookup_mixquery()

        dedup_key = (eid, mixquery.strip().lower())
        if dedup_key in self._seen_lookup_keys:
            self.metrics.duplicate_lookup_count += 1
        self._seen_lookup_keys.add(dedup_key)
        self._looked_up_eids.add(eid)

        before_visible = set(self.current_visible_sids) | set(self.selected_evidence)
        ranked = self.graph_index.lookup_entity(
            eid,
            mixquery,
            topk=self.expanded_visible_sentence_k,
            anchor_eid=anchor_eid,
        )
        for row in ranked:
            row["source"] = row.get("source") or row.get("entity_source") or "lookup"
        self.current_visible_sids = [sentence["sid"] for sentence in ranked]
        if ranked:
            self.metrics.lookup_success_count += 1
        if any(sentence["sid"] not in before_visible for sentence in ranked):
            self.metrics.lookup_new_sid_count += 1
        self._lookup_history.append({"eid": eid, "mixquery": mixquery, "num_sentences": len(ranked)})
        after_sids = [sentence["sid"] for sentence in ranked]
        new_sids = [s for s in after_sids if s not in before_visible]
        self._last_nav_event = {
            "action_type": "LOOKUP",
            "query_text": mixquery,
            "result_ids": after_sids,
            "new_result_ids": new_sids,
            "turn": self.step_count,
        }
        self._set_sentence_actions(ranked, include_answer=True)
        return self._format_lookup_observation(entity, mixquery, ranked, via)

    def _lookup_mixquery(self) -> str:
        """Build the LOOKUP retrieval query: question + text of all SELECTed
        evidence sentences. Falls back to the question alone when nothing has
        been SELECTed yet (e.g. a LOOKUP straight after INIT).

        The question is never truncated; SELECTed evidence fills whatever is
        left of the ``mixquery_max_words`` budget. Must stay in lock-step with
        ``snc_preview._lookup_mixquery``."""
        question = (self.question or "").strip()
        parts: List[str] = []
        for sid in self.selected_evidence:
            sentence = self.graph_index.sentences.get(sid)
            if not sentence:
                continue
            text = str(sentence.get("text") or "").strip()
            if text:
                parts.append(text)
        question_words = question.split()
        evidence_budget = max(self.mixquery_max_words - len(question_words), 0)
        evidence_words = " ".join(parts).split()[:evidence_budget]
        return " ".join(question_words + evidence_words)

    def _answer_with(self, sids: List[str]) -> str:
        self.step_count += 1
        self.metrics.answer_with_count += 1
        for sid in sids:
            if sid and sid not in self.selected_evidence and sid in self.graph_index.sentences:
                self.selected_evidence.append(sid)
        self.metrics.selected_sentence_count = len(self.selected_evidence)
        self.metrics.select_count += 1
        self.metrics.stop_count += 1
        self.stopped = True
        self.current_action_map = {}
        return self._format_stopped_observation(event="ANSWER_WITH")

    def _record_snc_seen_sids(self, ranked: List[dict]) -> None:
        """Append newly observed sentence ids in stable first-seen order."""

        for sentence in ranked:
            sid = sentence.get("sid")
            if not sid or sid in self._snc_seen_sid_set:
                continue
            self._snc_seen_sid_set.add(sid)
            self.snc_seen_sids.append(sid)

    def _set_sentence_actions(self, ranked: List[dict], include_answer: bool) -> None:
        self._record_snc_seen_sids(ranked)
        self.current_action_map = {}
        visible_sids = [sentence["sid"] for sentence in ranked]
        self._add_v3_actions(
            0,
            visible_sids=visible_sids,
            lookup_specs=self._lookup_specs_from_visible(ranked),
            include_answer=include_answer,
            lookup_cap=NAV_LOOKUP_K,
        )
        for sentence in ranked:
            if sentence["sid"] in self._visible_history:
                self._visible_history.remove(sentence["sid"])
            self._visible_history.append(sentence["sid"])
        self._record_available_action_counts()

    def _display_sid(self, sid: str) -> str:
        if sid not in self._real_sid_to_display:
            display = f"S{self._next_sentence_display_idx}"
            self._next_sentence_display_idx += 1
            self._real_sid_to_display[sid] = display
            self.current_display_sentence_map[display] = sid
        return self._real_sid_to_display[sid]

    def _ensure_display_eid(self, eid: str) -> str:
        for display_eid, real_eid in self.current_display_entity_map.items():
            if real_eid == eid:
                return display_eid
        display_eid = f"E{len(self.current_display_entity_map)}"
        self.current_display_entity_map[display_eid] = eid
        return display_eid

    def _entity_surface(self, entity: dict) -> str:
        forms = entity.get("surface_forms") or []
        if forms:
            return str(forms[0])
        return str(entity.get("canonical", ""))

    def _selected_lines(self) -> List[str]:
        if not self.selected_evidence:
            return ["- none"]
        lines = []
        for sid in self.selected_evidence:
            sentence = self.graph_index.sentences.get(sid)
            if not sentence:
                continue
            lines.append(selected_line(self._display_sid(sid), sentence))
        return lines or ["- none"]

    def _available_action_lines(self) -> List[str]:
        lines = actions_lines(self.current_action_map)
        return lines or ["- none"]

    def _entity_header_line(self, entity: dict) -> str:
        return f"{self._entity_surface(entity)} | {entity.get('label', 'ENTITY')}"

    def _format_initial_observation(self, ranked: List[dict]) -> str:
        lines = [
            f"step: {self.step_count}",
            "mode: harness_g",
            "graph_mode: offline_graph_index",
            "",
            "question:",
            self.question,
            "",
            "selected:",
            "- none",
            "",
            "visible_sentences:",
        ]
        lines.extend(sentence_line(self._display_sid(sentence["sid"]), sentence) for sentence in ranked)
        lines.extend(
            [
                "",
                "available_actions:",
                *self._available_action_lines(),
                "",
                "instruction:",
                "Choose exactly one available action id. Do not write a natural language search query.",
            ]
        )
        return observation_block(lines)

    def _format_select_observation(self, entities: List[dict], bridge_candidates: List[dict]) -> str:
        lines = [
            f"step: {self.step_count}",
            "mode: harness_g",
            "event: SELECT",
            "",
            "selected:",
            *self._selected_lines(),
            "",
            "available_actions:",
            *self._available_action_lines(),
            "",
            "frontier_entities:",
        ]
        if entities:
            for display_eid, eid in self.current_display_entity_map.items():
                if eid in self.frontier_entities:
                    lines.append(entity_line(display_eid, self.entities[eid]))
        else:
            lines.append("- none")

        lines.extend(["", "bridge_entities:"])
        if bridge_candidates:
            for candidate in bridge_candidates:
                target_eid = candidate.get("target_eid") or candidate.get("eid")
                source_eid = candidate.get("source_eid")
                if not target_eid or target_eid not in self.entities:
                    continue
                target_display = self._ensure_display_eid(target_eid)
                source_display = self._ensure_display_eid(source_eid) if source_eid in self.entities else "E?"
                target_line = entity_line(target_display, {**self.entities[target_eid], **candidate})
                lines.append(f"{source_display} -> {target_line}")
        else:
            lines.append("- none")

        lines.extend(
            [
                "",
                "instruction:",
                "Choose one action id: SELECT or ANSWER_WITH a useful sentence, LOOKUP an entity to find missing info, or ANSWER.",
            ]
        )
        return observation_block(lines)

    def _format_lookup_observation(
        self,
        entity: dict,
        mixquery: str,
        ranked: List[dict],
        via: Optional[str],
    ) -> str:
        lines = [
            f"step: {self.step_count}",
            "mode: harness_g",
            "event: LOOKUP",
            "",
            "target_entity:",
            self._entity_header_line(entity),
        ]
        if via:
            lines.extend(["", "via:", str(via)])
        lines.extend(
            [
                "",
                "mixed_query:",
                mixquery,
                "",
                "selected:",
                *self._selected_lines(),
                "",
                "new_visible_sentences:",
            ]
        )
        if ranked:
            lines.extend(sentence_line(self._display_sid(sentence["sid"]), sentence) for sentence in ranked)
        else:
            lines.append("- none")
        lines.extend(
            [
                "",
                "available_actions:",
                *self._available_action_lines(),
                "",
                "instruction:",
                "Select useful evidence, LOOKUP another entity, or ANSWER.",
            ]
        )
        return observation_block(lines)

    def _format_stopped_observation(self, event: str) -> str:
        lines = [
            f"step: {self.step_count}",
            "mode: harness_g",
            f"event: {event}",
            "",
            "selected_evidence:",
            *self._selected_lines(),
            "",
            "instruction:",
            "Now provide the final answer in <answer>...</answer> using only the selected evidence.",
        ]
        return observation_block(lines)

    def _format_invalid_observation(self, parsed: dict) -> str:
        lines = [
            f"step: {self.step_count}",
            "mode: harness_g",
            "event: INVALID_ACTION",
            "",
            f"invalid_action: {parsed.get('raw', '')}",
            f"invalid_reason: {parsed.get('invalid_reason')}",
            "",
            "selected:",
            *self._selected_lines(),
        ]
        if self.current_action_map:
            lines.extend(["", "available_actions:", *self._available_action_lines()])
        lines.extend(
            [
                "",
                "instruction:",
                "Choose exactly one available action id. Do not write a natural language search query.",
            ]
        )
        return observation_block(lines)
