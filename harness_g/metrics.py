from dataclasses import dataclass
from typing import Dict


@dataclass
class HarnessGEpisodeMetrics:
    valid_action_id_count: int = 0
    invalid_action_count: int = 0
    natural_query_count: int = 0
    select_count: int = 0
    stop_count: int = 0
    selected_sentence_count: int = 0
    answer_after_stop: int = 0
    lookup_count: int = 0
    lookup_success_count: int = 0
    lookup_new_sid_count: int = 0
    duplicate_lookup_count: int = 0
    bad_target_lookup_count: int = 0
    answer_with_count: int = 0
    answer_count: int = 0
    available_action_count: int = 0
    stop_action_available_count: int = 0
    lookup_action_available_count: int = 0
    answer_with_action_available_count: int = 0
    step_count: int = 0

    def to_dict(self) -> Dict[str, int]:
        return dict(self.__dict__)


def aggregate_episode_metrics(metrics_by_session: Dict[str, HarnessGEpisodeMetrics]) -> dict:
    totals = {}
    n = max(len(metrics_by_session), 1)
    for metrics in metrics_by_session.values():
        for key, value in metrics.to_dict().items():
            totals[key] = totals.get(key, 0) + value

    valid = totals.get("valid_action_id_count", 0)
    invalid = totals.get("invalid_action_count", 0)
    actions = max(valid + invalid, 1)
    sessions = list(metrics_by_session.values())
    stopped = sum(1 for metrics in sessions if metrics.stop_count > 0)
    selected_counts = [metrics.selected_sentence_count for metrics in sessions]
    available_count = totals.get("available_action_count", 0)
    lookup_count = totals.get("lookup_count", 0)
    answer_with_count = totals.get("answer_with_count", 0)

    return {
        **totals,
        "sessions": len(metrics_by_session),
        "valid_action_id_rate": valid / actions,
        "env_invalid_rate": invalid / actions,
        "natural_query_rate": totals.get("natural_query_count", 0) / actions,
        "select_rate": totals.get("select_count", 0) / actions,
        "stop_rate": totals.get("stop_count", 0) / actions,
        "no_stop_rate": 1.0 - (stopped / n),
        "avg_selected_sentence_count": sum(selected_counts) / max(len(selected_counts), 1),
        "selected_evidence_contains_gold_rate": 0.0,
        "answer_after_stop_rate": totals.get("answer_after_stop", 0) / n,
        "lookup_rate": lookup_count / actions,
        "lookup_success_rate": totals.get("lookup_success_count", 0) / max(lookup_count, 1),
        "lookup_new_sid_rate": totals.get("lookup_new_sid_count", 0) / max(lookup_count, 1),
        "duplicate_lookup_rate": totals.get("duplicate_lookup_count", 0) / max(lookup_count, 1),
        "answer_with_rate": answer_with_count / actions,
        "answer_rate": totals.get("answer_count", 0) / actions,
        "lookup_action_available_rate": totals.get("lookup_action_available_count", 0) / max(available_count, 1),
    }
