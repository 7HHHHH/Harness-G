import json
import os
import re
import string
from pathlib import Path
from typing import Dict, List, Tuple

from harness_g.protocol import parse_harness_g_action
from harness_g.text_utils import contains_any_answer, normalize_for_match


RUN_DIR = Path(os.environ.get("HARNESS_G_RUN_DIR", "runs/harness_g_run"))


def _answers(ground_truth) -> List[str]:
    if ground_truth is None:
        return []
    if hasattr(ground_truth, "tolist"):
        ground_truth = ground_truth.tolist()
    if isinstance(ground_truth, (list, tuple, set)):
        return [str(item) for item in ground_truth if item is not None]
    return [str(ground_truth)]


def _as_plain_data(value):
    return value.tolist() if hasattr(value, "tolist") else value


def _json_safe(value):
    value = _as_plain_data(value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    return str(value)


def _flatten_text_values(value) -> List[str]:
    value = _as_plain_data(value)
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        texts: List[str] = []
        for item in value.values():
            texts.extend(_flatten_text_values(item))
        return texts
    if isinstance(value, (list, tuple, set)):
        texts = []
        for item in value:
            texts.extend(_flatten_text_values(item))
        return texts
    return [str(value)]


def _supporting_evidence(extra_info) -> List[str]:
    extra_info = _as_plain_data(extra_info)
    if not isinstance(extra_info, dict):
        return []
    texts: List[str] = []
    for key in ("supporting_evidence", "supporting_evidences", "supporting_facts", "evidences", "evidence"):
        if key in extra_info:
            texts.extend(_flatten_text_values(extra_info.get(key)))
    return [text for text in texts if text and len(normalize_for_match(text)) >= 3]


def _evidence_targets(ground_truth, extra_info=None) -> List[str]:
    seen = set()
    targets = []
    for item in [*_answers(ground_truth), *_supporting_evidence(extra_info)]:
        normalized = normalize_for_match(item)
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        targets.append(str(item))
    return targets


def _extract_answer(solution_str: str) -> str:
    matches = re.findall(r"<answer>(.*?)</answer>", solution_str or "", re.DOTALL)
    return matches[-1].strip() if matches else ""


def _normalize(text: str) -> str:
    text = str(text).lower()
    text = "".join(ch for ch in text if ch not in set(string.punctuation))
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def _f1(prediction: str, answer: str) -> float:
    pred_tokens = _normalize(prediction).split()
    gold_tokens = _normalize(answer).split()
    if not pred_tokens or not gold_tokens:
        return float(pred_tokens == gold_tokens)
    common = {}
    for token in pred_tokens:
        common[token] = min(pred_tokens.count(token), gold_tokens.count(token))
    same = sum(common.values())
    if same == 0:
        return 0.0
    precision = same / len(pred_tokens)
    recall = same / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def answer_f1(prediction: str, ground_truth) -> float:
    answers = _answers(ground_truth)
    if not answers:
        return 0.0
    return max(_f1(prediction, answer) for answer in answers)


def answer_em(prediction: str, ground_truth) -> float:
    normalized = _normalize(prediction)
    return float(any(normalized == _normalize(answer) for answer in _answers(ground_truth)))


def _assistant_query_spans(solution_str: str) -> List[Tuple[int, str]]:
    spans = []
    assistant_blocks = list(re.finditer(r"<\|im_start\|>assistant\n(.*?)(?:<\|im_end\|>|$)", solution_str or "", re.DOTALL))
    if assistant_blocks:
        for block in assistant_blocks:
            for match in re.finditer(r"<query>(.*?)</query>", block.group(1), re.DOTALL):
                raw = match.group(1).strip()
                try:
                    raw = json.loads(raw).get("query", raw)
                except Exception:
                    pass
                spans.append((block.start(1) + match.start(), str(raw)))
        return spans

    text_without_obs = re.sub(r"\[HARNESS_G_OBS\].*?\[/HARNESS_G_OBS\]", "", solution_str or "", flags=re.DOTALL)
    for match in re.finditer(r"<query>(.*?)</query>", text_without_obs, re.DOTALL):
        raw = match.group(1).strip()
        try:
            raw = json.loads(raw).get("query", raw)
        except Exception:
            pass
        spans.append((match.start(), str(raw)))
    return spans


def _observation_spans(solution_str: str) -> List[Tuple[int, int, str]]:
    return [
        (match.start(), match.end(), _decode_observation_text(match.group(0)))
        for match in re.finditer(r"\[HARNESS_G_OBS\].*?\[/HARNESS_G_OBS\]", solution_str or "", re.DOTALL)
    ]


def _decode_observation_text(obs: str) -> str:
    text = str(obs or "")
    if "\\n" not in text and '\\"' not in text and "\\t" not in text:
        return text
    try:
        return json.loads(f'"{text}"')
    except Exception:
        return text.replace("\\n", "\n").replace("\\t", "\t").replace('\\"', '"')


def _observation_contents(solution_str: str) -> List[str]:
    return [obs for _, _, obs in _observation_spans(solution_str)]


def _preceding_observation(observations: List[Tuple[int, int, str]], pos: int) -> str:
    previous = ""
    for start, end, obs in observations:
        if end <= pos:
            previous = obs
        else:
            break
    return previous


def _following_observation(observations: List[Tuple[int, int, str]], pos: int) -> str:
    for start, _, obs in observations:
        if start >= pos:
            return obs
    return ""


def _action_map_from_obs(obs: str) -> Dict[str, dict]:
    action_map: Dict[str, dict] = {}
    obs = _decode_observation_text(obs)
    for line in (obs or "").splitlines():
        match = re.match(
            r"\s*(A\d+)\s*=\s*(SELECT|LOOKUP|ANSWER_WITH|ANSWER)\b(?:\s+([^|]+?))?(?:\s*\||$)",
            line,
        )
        if not match:
            continue
        action_id, action_type, target = match.groups()
        item = {"type": action_type}
        if action_type in ("SELECT", "ANSWER_WITH"):
            item["display_sid"] = (target or "").strip().split()[0] if target else None
        elif action_type == "LOOKUP":
            item["display_eid"] = (target or "").strip().split()[0] if target else None
        action_map[action_id] = item
    return action_map


def _available_action_counts(action_map: Dict[str, dict]) -> dict:
    counts = {
        "available_action_count": len(action_map),
        "stop_action_available_count": 0,
        "lookup_action_available_count": 0,
        "answer_with_action_available_count": 0,
    }
    for action in action_map.values():
        action_type = action.get("type")
        if action_type == "ANSWER":
            counts["stop_action_available_count"] += 1
        elif action_type == "LOOKUP":
            counts["lookup_action_available_count"] += 1
        elif action_type == "ANSWER_WITH":
            counts["answer_with_action_available_count"] += 1
    return counts


def _selected_lines(solution_str: str) -> List[str]:
    lines = []
    for obs in _observation_contents(solution_str):
        capture = False
        for raw_line in obs.splitlines():
            line = raw_line.strip()
            if line in {"selected:", "selected_evidence:"}:
                capture = True
                continue
            if capture and (not line or line.endswith(":")):
                capture = False
                continue
            if capture and re.match(r"S\d+\s+\|", line):
                lines.append(line)
    return list(dict.fromkeys(lines))


def _obs_contains_target(solution_str: str, marker: str, targets: List[str]) -> bool:
    for obs in _observation_contents(solution_str):
        if marker in obs and contains_any_answer(obs, targets):
            return True
    return False


def _single_obs_contains_target(obs: str, marker: str, targets: List[str]) -> bool:
    return bool(marker in (obs or "") and contains_any_answer(obs, targets))


def _target_matches_text(target: str, text: str) -> bool:
    """Token-boundary containment with a fuzzy fallback for sentence-length targets.

    Plain substring matching over-counts short targets ("no" inside "north") and
    under-counts gold sentences whose segmentation differs slightly from the
    graph corpus, so both cases are handled explicitly here.
    """
    norm_target = normalize_for_match(target)
    norm_text = normalize_for_match(text)
    if not norm_target or not norm_text:
        return False
    if f" {norm_target} " in f" {norm_text} ":
        return True
    target_tokens = norm_target.split()
    if len(target_tokens) >= 8:
        text_tokens = set(norm_text.split())
        hit = sum(1 for token in target_tokens if token in text_tokens)
        return hit / len(target_tokens) >= 0.8
    return False


def _obs_sentence_lines(obs: str) -> str:
    """Sentence-bearing lines (S0 | ...) of one observation, so gold-fact matching
    never fires on the echoed question or entity headers."""
    lines = [line for line in _decode_observation_text(obs).splitlines() if re.match(r"\s*S\d+\s+\|", line)]
    return "\n".join(lines)


def _match_new_facts(text: str, gold_facts: List[str], seen: set) -> List[int]:
    return [idx for idx, fact in enumerate(gold_facts) if idx not in seen and _target_matches_text(fact, text)]


def analyze(solution_str: str, ground_truth, extra_info=None) -> dict:
    observations = _observation_spans(solution_str)
    query_spans = _assistant_query_spans(solution_str)
    evidence_targets = _evidence_targets(ground_truth, extra_info)
    gold_facts = _supporting_evidence(extra_info) or [str(a) for a in _answers(ground_truth)]
    facts_seen: set = set()
    initialized = False
    saw_select = False
    saw_stop = False
    stop_pos = -1
    metrics = {
        "valid_action_id_count": 0,
        "invalid_action_count": 0,
        "natural_query_count": 0,
        "select_count": 0,
        "stop_count": 0,
        "selected_sentence_count": 0,
        "answer_after_stop": 0,
        "lookup_count": 0,
        "lookup_success_count": 0,
        "answer_with_count": 0,
        "answer_count": 0,
        "available_action_count": 0,
        "stop_action_available_count": 0,
        "lookup_action_available_count": 0,
        "answer_with_action_available_count": 0,
    }
    protocol_points = 0.0
    protocol_total = 0.0

    for pos, raw in query_spans:
        obs = _preceding_observation(observations, pos)
        next_obs = _following_observation(observations, pos)
        action_map = _action_map_from_obs(obs)
        for key, value in _available_action_counts(action_map).items():
            metrics[key] += value
        parsed = parse_harness_g_action(raw, action_map=action_map, initialized=initialized)
        protocol_total += 1.0
        if parsed["is_valid"]:
            protocol_points += 1.0
            metrics["valid_action_id_count"] += 1
        else:
            metrics["invalid_action_count"] += 1
            if parsed.get("is_natural_query"):
                metrics["natural_query_count"] += 1

        if parsed["semantic_action"] == "INIT" and parsed["is_valid"]:
            initialized = True
            protocol_points += 0.25
        elif parsed["semantic_action"] == "SELECT" and parsed["is_valid"]:
            saw_select = True
            metrics["select_count"] += 1
            protocol_points += 0.50
        elif parsed["semantic_action"] == "LOOKUP" and parsed["is_valid"]:
            metrics["lookup_count"] += 1
            protocol_points += 0.25
            if _single_obs_contains_target(next_obs, "new_visible_sentences:", evidence_targets):
                metrics["lookup_success_count"] += 1
        elif parsed["semantic_action"] == "ANSWER_WITH" and parsed["is_valid"]:
            # ANSWER_WITH harvests visible sentence(s) into selected evidence AND
            # ends the episode. Count it as both a SELECT (raises selected_coverage
            # via the echoed selected_evidence) and a STOP (so answer_after_stop /
            # qa_reward_eligible hold).
            saw_select = True
            saw_stop = True
            stop_pos = pos
            metrics["select_count"] += 1
            metrics["stop_count"] += 1
            metrics["answer_with_count"] += 1
            protocol_points += 0.50
        elif parsed["semantic_action"] == "ANSWER" and parsed["is_valid"]:
            saw_stop = True
            stop_pos = pos
            metrics["stop_count"] += 1
            metrics["answer_count"] += 1
            protocol_points += 0.50

        if parsed["is_valid"]:
            new_facts = _match_new_facts(_obs_sentence_lines(next_obs), gold_facts, facts_seen)
            if new_facts:
                facts_seen.update(new_facts)

    if query_spans and query_spans[0][1].strip().upper() == "INIT":
        protocol_points += 0.5
    if saw_select:
        protocol_points += 0.75
    if saw_stop:
        protocol_points += 0.75
    if metrics["natural_query_count"] == 0:
        protocol_points += 0.5

    answer = _extract_answer(solution_str)
    answer_pos = (solution_str or "").rfind("<answer>")
    answer_after_stop = bool(answer and saw_stop and answer_pos > stop_pos)
    metrics["answer_after_stop"] = int(answer_after_stop)
    selected = _selected_lines(solution_str)
    metrics["selected_sentence_count"] = len(selected)

    protocol_reward = min(protocol_points / max(protocol_total + 2.0, 1.0), 1.0)
    if metrics["invalid_action_count"] or metrics["natural_query_count"]:
        protocol_reward *= 0.5
    if answer and not answer_after_stop:
        protocol_reward *= 0.5
    if saw_stop and not saw_select:
        protocol_reward *= 0.6

    evidence_reward = 0.0
    if selected:
        evidence_reward += 0.45
    if contains_any_answer("\n".join(selected), evidence_targets):
        evidence_reward += 0.35
    if _obs_contains_target(solution_str, "new_visible_sentences:", evidence_targets):
        evidence_reward += 0.15
    evidence_reward = min(evidence_reward, 1.0)

    selected_text = "\n".join(selected)
    selected_fact_count = sum(1 for fact in gold_facts if _target_matches_text(fact, selected_text))
    visible_coverage = len(facts_seen) / len(gold_facts) if gold_facts else 0.0
    selected_coverage = selected_fact_count / len(gold_facts) if gold_facts else 0.0

    f1 = answer_f1(answer, ground_truth) if answer else 0.0
    em = answer_em(answer, ground_truth) if answer else 0.0
    qa_reward_eligible = bool(answer_after_stop and selected)
    answer_reward = f1 if qa_reward_eligible else 0.0

    total_reward = 0.38 * protocol_reward + 0.32 * evidence_reward + 0.25 * answer_reward
    if not selected:
        total_reward = min(total_reward, -0.02)
    if not answer:
        no_answer_cap = -0.02
        if metrics["valid_action_id_count"] > 0 and metrics["invalid_action_count"] == 0:
            no_answer_cap = 0.03
        if saw_select and metrics["invalid_action_count"] == 0:
            no_answer_cap = 0.28
        if saw_select and saw_stop and metrics["invalid_action_count"] == 0:
            no_answer_cap = 0.35
        total_reward = min(total_reward, no_answer_cap)
    if answer and not answer_after_stop:
        total_reward = min(total_reward, 0.0)
    if not query_spans:
        total_reward = min(total_reward, -0.05)
    if (solution_str or "").count("<answer>") != (solution_str or "").count("</answer>"):
        total_reward = min(total_reward, -1.0)
    if metrics["natural_query_count"]:
        total_reward -= 0.25

    metrics.update(
        {
            "answer_f1": f1,
            "answer_em": em,
            "protocol_reward": protocol_reward,
            "evidence_reward": evidence_reward,
            "answer_reward": answer_reward,
            "qa_reward_eligible": int(qa_reward_eligible),
            "supporting_evidence_count": len(_supporting_evidence(extra_info)),
            "gold_fact_count": len(gold_facts),
            "gold_fact_visible_count": len(facts_seen),
            "gold_fact_selected_count": selected_fact_count,
            "visible_coverage": visible_coverage,
            "selected_coverage": selected_coverage,
            "total_reward": total_reward,
        }
    )
    return metrics


def _write_metrics(metrics: dict) -> None:
    path = Path(os.environ.get("HARNESS_G_REWARD_METRICS_PATH", RUN_DIR / "reward_metrics.jsonl"))
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(_json_safe(metrics), ensure_ascii=False) + "\n")
    except Exception:
        return


def compute_score(solution_str, ground_truth, extra_info=None):
    metrics = analyze(solution_str, ground_truth, extra_info=extra_info)
    if extra_info:
        metrics["extra_info"] = _json_safe(extra_info)
    _write_metrics(metrics)
    return float(metrics["total_reward"])


def compute_score_format_answer(solution_str, ground_truth, extra_info=None):
    return compute_score(solution_str, ground_truth, extra_info=extra_info)


def compute_score_format(solution_str, extra_info=None):
    return float(analyze(solution_str, [], extra_info=extra_info)["protocol_reward"])


def _compute_outcome_score(solution_str, ground_truth, extra_info, metric_key, fn_name):
    """Sparse outcome-only reward (answer_em / answer_f1) that still writes the
    full per-episode diagnostics to reward_metrics.jsonl. ``total_reward`` in
    the record is the composite diagnostic; ``outcome_reward`` is what training saw."""
    metrics = analyze(solution_str, ground_truth, extra_info=extra_info)
    if extra_info:
        metrics["extra_info"] = _json_safe(extra_info)
    metrics["outcome_score_fn"] = fn_name
    metrics["outcome_reward"] = float(metrics[metric_key])
    _write_metrics(metrics)
    return float(metrics[metric_key])


def compute_score_f1(solution_str, ground_truth, extra_info=None):
    return _compute_outcome_score(solution_str, ground_truth, extra_info, "answer_f1", "f1")


def compute_score_em(solution_str, ground_truth, extra_info=None):
    return _compute_outcome_score(solution_str, ground_truth, extra_info, "answer_em", "em")
