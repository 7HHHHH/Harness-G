import pytest

from harness_g.answer_scoring import (
    CONTEXT_MODE_ENV,
    ActorAnswerScorer,
    MockAnswerScorer,
    aggregate_alias_max_scores,
    build_answer_scoring_prompt,
    default_context_mode,
    evidence_context_sids,
    expand_answer_alias_pairs,
    normalize_answer_aliases,
)


class FakeGraphIndex:
    sentences = {
        "s1": {"text": "Sentence one supports the answer."},
        "s2": {"text": "Sentence two adds the missing fact."},
    }


def test_evidence_context_mode_b_preserves_order_and_dedupes():
    selected = ["s1", "s2", "s1"]
    surfaced = ["s2", "s3", "s4", "s3"]

    assert evidence_context_sids(selected, surfaced, mode="B") == [
        "s1",
        "s2",
        "s3",
        "s4",
    ]


def test_evidence_context_mode_a_returns_selected_only():
    selected = ["s1", "s2", "s1"]
    surfaced = ["s2", "s3"]

    assert evidence_context_sids(selected, surfaced, mode="A") == selected


def test_evidence_context_default():
    selected = ["s1"]
    surfaced = ["s2"]

    assert default_context_mode() == "B"
    assert evidence_context_sids(selected, surfaced) == ["s1", "s2"]


def test_build_answer_scoring_prompt_includes_question_evidence_and_answer_instruction():
    prompt = build_answer_scoring_prompt(
        "What is the final answer?",
        ["s1", "s2"],
        FakeGraphIndex(),
    )

    assert "What is the final answer?" in prompt
    assert "Sentence one supports the answer." in prompt
    assert "Sentence two adds the missing fact." in prompt
    assert "<answer>" in prompt
    assert "</answer>" in prompt
    assert "evidence observed so far" in prompt.lower()
    assert "using only the observed evidence" in prompt.lower()


def test_normalize_answer_aliases_expands_array_like_values_and_dedupes():
    class ArrayLike:
        def tolist(self):
            return ["Germany", " Deutschland ", "Germany", None]

    assert normalize_answer_aliases(ArrayLike()) == ["Germany", "Deutschland"]
    assert normalize_answer_aliases("no") == ["no"]
    assert normalize_answer_aliases(None) == []


def test_expand_and_aggregate_answer_alias_scores():
    prompts, aliases, owners = expand_answer_alias_pairs(
        ["prompt-0", "prompt-1"],
        [["first", "second"], "only"],
    )

    assert prompts == ["prompt-0", "prompt-0", "prompt-1"]
    assert aliases == ["first", "second", "only"]
    assert owners == [0, 0, 1]
    assert aggregate_alias_max_scores([0.2, 0.8, 0.4], owners, 2) == [0.8, 0.4]


def test_answer_alias_pair_helpers_fail_on_missing_aliases_or_scores():
    with pytest.raises(ValueError, match="no answer aliases"):
        expand_answer_alias_pairs(["prompt"], [[]])
    with pytest.raises(ValueError, match="at least one alias score"):
        aggregate_alias_max_scores([], [], 1)


def test_mock_answer_scorer_uses_provided_scores_and_length_normalizes():
    scorer = MockAnswerScorer({("prompt", "gold answer"): -4.0})

    assert scorer.score(["prompt"], ["gold answer"]) == [-4.0]
    assert scorer.score(["prompt"], ["gold answer"], length_normalize=True) == [-2.0]


def test_mock_answer_scorer_default_is_deterministic():
    scorer = MockAnswerScorer(prompt_char_weight=0.1, answer_char_weight=0.5)

    score = scorer.score(["abcd"], ["ef gh"])
    normalized = scorer.score(["abcd"], ["ef gh"], length_normalize=True)

    assert score[0] == pytest.approx(-(4 * 0.1 + 5 * 0.5))
    assert normalized[0] == pytest.approx(score[0] / 2)


def test_actor_answer_scorer_symbol_is_importable_without_model_runtime():
    assert ActorAnswerScorer.__name__ == "ActorAnswerScorer"
