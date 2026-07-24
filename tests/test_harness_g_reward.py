import json

import numpy as np

from verl.utils.reward_score.harness_g_qa import analyze, compute_score, compute_score_f1


def _full_rollout():
    return """
<|im_start|>assistant
<think>start</think>
<query>{"query": "INIT"}</query><|im_end|>
<|im_start|>user
[HARNESS_G_OBS]
visible_sentences:
S0 | Ada Lovelace: Ada Lovelace worked with Charles Babbage.
available_actions:
A0 = SELECT S0
A1 = ANSWER_WITH S0 | evidence: Ada Lovelace worked with Charles Babbage.
[/HARNESS_G_OBS]<|im_end|>
<|im_start|>assistant
<think>select</think>
<query>{"query": "A0"}</query><|im_end|>
<|im_start|>user
[HARNESS_G_OBS]
event: SELECT
selected:
S0 | Ada Lovelace: Ada Lovelace worked with Charles Babbage.
available_actions:
A0 = LOOKUP E0 | entity: Ada Lovelace | via: from S0
A1 = ANSWER
[/HARNESS_G_OBS]<|im_end|>
<|im_start|>assistant
<think>lookup</think>
<query>{"query": "A0"}</query><|im_end|>
<|im_start|>user
[HARNESS_G_OBS]
event: LOOKUP
new_visible_sentences:
S1 | Ada Lovelace: Ada Lovelace was born in London.
available_actions:
A0 = SELECT S1
A1 = ANSWER
[/HARNESS_G_OBS]<|im_end|>
<|im_start|>assistant
<think>select answer evidence</think>
<query>{"query": "A0"}</query><|im_end|>
<|im_start|>user
[HARNESS_G_OBS]
event: SELECT
selected:
S0 | Ada Lovelace worked with Charles Babbage.
S1 | Ada Lovelace was born in London.
available_actions:
A0 = ANSWER
[/HARNESS_G_OBS]<|im_end|>
<|im_start|>assistant
<think>answer</think>
<query>{"query": "A0"}</query><|im_end|>
<|im_start|>user
[HARNESS_G_OBS]
event: ANSWER
selected_evidence:
S0 | Ada Lovelace worked with Charles Babbage.
S1 | Ada Lovelace was born in London.
[/HARNESS_G_OBS]<|im_end|>
<|im_start|>assistant
<think>final</think>
<answer>London</answer><|im_end|>
"""


def test_harness_g_reward_positive_and_penalties():
    metrics = analyze(_full_rollout(), ["London"])
    assert metrics["total_reward"] > 0
    assert metrics["answer_f1"] == 1.0
    assert metrics["answer_after_stop"] == 1
    assert metrics["valid_action_id_count"] == 5
    assert metrics["select_count"] == 2
    assert metrics["lookup_count"] == 1
    assert metrics["lookup_success_count"] == 1
    assert metrics["answer_count"] == 1

    natural = '<query>{"query": "Ada Lovelace birthplace"}</query><answer>London</answer>'
    assert compute_score(natural, ["London"]) < metrics["total_reward"]
    assert analyze("<answer>London</answer>", ["London"])["total_reward"] <= 0.05


def test_harness_g_outcome_f1_reward():
    assert compute_score_f1(_full_rollout(), ["London"]) == 1.0
    assert compute_score_f1(_full_rollout(), ["Paris"]) == 0.0
    # answer without a terminating ANSWER/ANSWER_WITH still scores plain F1
    metrics = analyze(_full_rollout(), ["London"])
    assert metrics["qa_reward_eligible"] == 1


def test_harness_g_answer_with_counts_as_select_and_stop():
    rollout = """
<|im_start|>assistant
<query>{"query": "INIT"}</query><|im_end|>
<|im_start|>user
[HARNESS_G_OBS]
visible_sentences:
S0 | Ada Lovelace: Ada Lovelace was born in London.
available_actions:
A0 = SELECT S0
A1 = ANSWER_WITH S0 | evidence: Ada Lovelace was born in London.
[/HARNESS_G_OBS]<|im_end|>
<|im_start|>assistant
<query>{"query": "A1"}</query><|im_end|>
<|im_start|>user
[HARNESS_G_OBS]
event: ANSWER_WITH
selected_evidence:
S0 | Ada Lovelace was born in London.
[/HARNESS_G_OBS]<|im_end|>
<|im_start|>assistant
<answer>London</answer><|im_end|>
"""
    metrics = analyze(rollout, ["London"])
    assert metrics["answer_with_count"] == 1
    assert metrics["select_count"] == 1
    assert metrics["stop_count"] == 1
    assert metrics["answer_after_stop"] == 1
    assert metrics["qa_reward_eligible"] == 1
    assert metrics["total_reward"] > 0


def test_harness_g_reward_uses_supporting_evidence_targets():
    rollout = _full_rollout()
    base = analyze(rollout, ["London"])
    enriched = analyze(
        rollout,
        ["London"],
        extra_info={"supporting_evidence": ["Ada Lovelace was born in London."]},
    )
    assert enriched["supporting_evidence_count"] == 1
    assert enriched["gold_fact_count"] == 1
    assert enriched["gold_fact_selected_count"] == 1
    assert enriched["selected_coverage"] == 1.0
    assert base["supporting_evidence_count"] == 0


def test_harness_g_reward_parses_json_wrapped_observation():
    rollout = _full_rollout().replace("\n", "\\n")
    metrics = analyze(rollout, ["London"])
    assert metrics["selected_sentence_count"] > 0


def test_harness_g_select_answer_beats_direct_answer():
    full = compute_score(_full_rollout(), ["London"])
    direct = compute_score("<answer>London</answer>", ["London"])
    assert full > direct


def test_harness_g_reward_metrics_serializes_nested_extra_info(tmp_path, monkeypatch):
    metrics_path = tmp_path / "reward_metrics.jsonl"
    monkeypatch.setenv("HARNESS_G_REWARD_METRICS_PATH", str(metrics_path))
    score = compute_score(
        "<answer>London</answer>",
        ["London"],
        extra_info={"question": np.array(["Where?"]), "supporting_evidence": [np.array(["Ada was born in London."])]},
    )
    assert score <= 0.05
    rows = [json.loads(line) for line in metrics_path.read_text(encoding="utf-8").splitlines()]
    assert rows
    assert rows[-1]["extra_info"]["question"] == ["Where?"]
