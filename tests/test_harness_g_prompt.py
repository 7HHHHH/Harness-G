"""Tests for the navigation prompt (action-space alignment).

The prompt must only describe actions that actually exist in the
environment: SELECT / LOOKUP / ANSWER_WITH / ANSWER. Legacy actions
(OPEN_CONTEXT / BRIDGE_ENTITY / EXPAND_ENTITY / REWRITE_QUERY / STOP) must
never appear.
"""
import script_process_harness_g as sp

_LEGACY_ACTIONS = ["OPEN_CONTEXT", "BRIDGE_ENTITY", "EXPAND_ENTITY", "REWRITE_QUERY", "STOP"]


def test_prompt_contains_actions():
    text = sp._navigation_v3_instruction()
    assert "SELECT" in text
    assert "LOOKUP" in text
    assert "ANSWER_WITH" in text
    assert "ANSWER" in text


def test_prompt_no_legacy_actions():
    text = sp._navigation_v3_instruction()
    for old in _LEGACY_ACTIONS:
        assert old not in text, f"legacy action {old} leaked into the prompt"


def test_instruction_template_matches_navigation_prompt():
    text = sp._instruction_template()
    assert text == sp._navigation_v3_instruction()
    assert text == sp.INSTRUCTION
