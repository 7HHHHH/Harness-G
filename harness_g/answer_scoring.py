"""Answer-scoring helpers for SNC intrinsic rewards.

The score implemented here is g(C) = P_theta(a* | q, C), represented as the
teacher-forced log-probability of the gold answer under the frozen answerer.
"""

from __future__ import annotations

import os
from itertools import chain
from typing import Any, Iterable, Mapping, Protocol, Sequence


CONTEXT_MODE_ENV = "HARNESS_G_SNC_CONTEXT_MODE"
VALID_CONTEXT_MODES = frozenset({"A", "B"})


def normalize_answer_aliases(value: Any) -> list[str]:
    """Return ordered, unique answer aliases from parquet/runtime values.

    PyArrow/Pandas materializes the project's ``ground_truth`` lists as NumPy
    object arrays.  Calling ``str(array)`` produces a bracketed NumPy repr,
    which is not an answer alias.  Normalize at the SNC boundary instead so
    the intrinsic scorer and the outcome scorer use the same target semantics.
    """

    if value is None:
        return []
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, (list, tuple, set, frozenset)):
        raw_aliases = value
    else:
        raw_aliases = [value]

    aliases: list[str] = []
    seen: set[str] = set()
    for item in raw_aliases:
        if item is None:
            continue
        alias = str(item).strip()
        if not alias or alias in seen:
            continue
        seen.add(alias)
        aliases.append(alias)
    return aliases


def expand_answer_alias_pairs(
    prompts: Sequence[str], answer_values: Sequence[Any]
) -> tuple[list[str], list[str], list[int]]:
    """Expand one prompt per valid alias and retain its prompt owner index."""

    if len(prompts) != len(answer_values):
        raise ValueError(
            f"Expected one answer-alias value per prompt, got {len(prompts)} "
            f"and {len(answer_values)}"
        )

    expanded_prompts: list[str] = []
    expanded_aliases: list[str] = []
    owners: list[int] = []
    for prompt_index, (prompt, raw_aliases) in enumerate(zip(prompts, answer_values)):
        aliases = normalize_answer_aliases(raw_aliases)
        if not aliases:
            raise ValueError(
                f"SNC scorer received no answer aliases for prompt {prompt_index}"
            )
        for alias in aliases:
            expanded_prompts.append(str(prompt))
            expanded_aliases.append(alias)
            owners.append(prompt_index)
    return expanded_prompts, expanded_aliases, owners


def aggregate_alias_max_scores(
    pair_scores: Sequence[float], owners: Sequence[int], prompt_count: int
) -> list[float]:
    """Return the best alias score for each original prompt."""

    if len(pair_scores) != len(owners):
        raise ValueError(
            f"Expected one owner per alias score, got {len(pair_scores)} and {len(owners)}"
        )
    if prompt_count < 0:
        raise ValueError("prompt_count must be non-negative")

    prompt_scores = [float("-inf")] * prompt_count
    for owner, value in zip(owners, pair_scores):
        if owner < 0 or owner >= prompt_count:
            raise ValueError(f"Alias owner index {owner} is out of range")
        prompt_scores[owner] = max(prompt_scores[owner], float(value))
    if any(score == float("-inf") for score in prompt_scores):
        raise ValueError("Every prompt must have at least one alias score")
    return prompt_scores


def default_context_mode() -> str:
    """Return the runtime default for SNC answer-scoring context assembly."""

    return _normalize_context_mode("B")


def evidence_context_sids(
    selected_sids: Iterable[str],
    surfaced_sids: Iterable[str],
    mode: str | None = None,
) -> list[str]:
    """Assemble sentence ids for answer scoring under the SNC A/B decision.

    Mode B scores selected evidence plus this move's newly surfaced evidence.
    Mode A scores selected evidence only.
    """

    context_mode = default_context_mode() if mode is None else _normalize_context_mode(mode)
    if context_mode == "A":
        return list(selected_sids)

    ordered_sids: list[str] = []
    seen: set[str] = set()
    for sid in chain(selected_sids, surfaced_sids):
        if sid in seen:
            continue
        seen.add(sid)
        ordered_sids.append(sid)
    return ordered_sids


def build_answer_scoring_prompt(question: str, evidence_sids: Iterable[str], graph_index: Any) -> str:
    """Build the frozen-answerer prompt used to score a gold answer."""

    evidence_lines = []
    for idx, sid in enumerate(evidence_sids, start=1):
        sentence = _lookup_sentence(graph_index.sentences, sid)
        evidence_lines.append(f"[{idx}] {_sentence_text(sentence)}")

    evidence_block = "\n".join(evidence_lines) if evidence_lines else "(no evidence observed)"
    return (
        f"Question: {question}\n\n"
        f"Evidence observed so far:\n{evidence_block}\n\n"
        "Provide the final answer in <answer>...</answer> using only the observed evidence."
    )


class AnswerScorer(Protocol):
    """Interface for computing teacher-forced answer log-probabilities."""

    def score(
        self,
        prompts: list[str],
        gold_answers: list[str],
        length_normalize: bool = False,
    ) -> list[float]:
        """Return one g value for each prompt/gold-answer pair."""
        ...


class MockAnswerScorer:
    """Deterministic answer scorer for unit tests and dry plumbing checks."""

    def __init__(
        self,
        scores: Mapping[tuple[str, str], float] | None = None,
        prompt_char_weight: float = 0.001,
        answer_char_weight: float = 0.01,
    ) -> None:
        self.scores = dict(scores or {})
        self.prompt_char_weight = prompt_char_weight
        self.answer_char_weight = answer_char_weight

    def score(
        self,
        prompts: list[str],
        gold_answers: list[str],
        length_normalize: bool = False,
    ) -> list[float]:
        _validate_pairs(prompts, gold_answers)

        values = []
        for prompt, gold_answer in zip(prompts, gold_answers):
            score = self.scores.get((prompt, gold_answer))
            if score is None:
                score = -(
                    len(prompt) * self.prompt_char_weight
                    + len(gold_answer) * self.answer_char_weight
                )
            if length_normalize:
                score /= _response_token_count(gold_answer)
            values.append(float(score))
        return values


# REQUIRES real model + verl runtime; validated in M4/real-run, not unit-tested here.
class ActorAnswerScorer:
    """VERL actor adapter for model-backed answer scoring.

    The adapter builds DataProto batches containing prompt tokens plus
    teacher-forced response tokens, calls ``actor.compute_log_prob``, and sums
    response-token log-probabilities.
    """

    def __init__(
        self,
        actor: Any,
        tokenizer: Any,
        *,
        max_prompt_length: int | None = None,
        max_response_length: int | None = None,
        add_prompt_special_tokens: bool = True,
        add_response_special_tokens: bool = False,
        meta_info: Mapping[str, Any] | None = None,
    ) -> None:
        self.actor = actor
        self.tokenizer = tokenizer
        self.max_prompt_length = max_prompt_length
        self.max_response_length = max_response_length
        self.add_prompt_special_tokens = add_prompt_special_tokens
        self.add_response_special_tokens = add_response_special_tokens
        self.meta_info = {"temperature": 1.0}
        if meta_info is not None:
            self.meta_info.update(meta_info)

    def score(
        self,
        prompts: list[str],
        gold_answers: list[str],
        length_normalize: bool = False,
    ) -> list[float]:
        _validate_pairs(prompts, gold_answers)
        if not prompts:
            return []

        data, response_mask = self._build_dataproto(prompts, gold_answers)
        result = self.actor.compute_log_prob(data)
        log_probs = _extract_log_probs(result)
        response_mask = response_mask.to(log_probs.device)

        if log_probs.shape[-1] != response_mask.shape[-1]:
            log_probs = log_probs[:, -response_mask.shape[-1] :]

        scores = (log_probs * response_mask).sum(dim=-1)
        if length_normalize:
            denom = response_mask.sum(dim=-1).clamp_min(1)
            scores = scores / denom
        return [float(value) for value in scores.detach().cpu().tolist()]

    def _build_dataproto(
        self,
        prompts: Sequence[str],
        gold_answers: Sequence[str],
    ) -> tuple[Any, Any]:
        import torch
        from verl.protocol import DataProto

        prompt_rows = [
            _truncate_left(
                _encode(self.tokenizer, prompt, self.add_prompt_special_tokens),
                self.max_prompt_length,
            )
            for prompt in prompts
        ]
        response_rows = [
            _truncate_right(
                _encode(self.tokenizer, gold_answer, self.add_response_special_tokens),
                self.max_response_length,
            )
            for gold_answer in gold_answers
        ]

        pad_id = _pad_token_id(self.tokenizer)
        max_prompt_len = max(max(len(row) for row in prompt_rows), 1)
        max_response_len = max(max(len(row) for row in response_rows), 1)

        input_ids = []
        attention_mask = []
        responses = []
        response_mask = []
        for prompt_ids, response_ids in zip(prompt_rows, response_rows):
            prompt_pad_len = max_prompt_len - len(prompt_ids)
            response_pad_len = max_response_len - len(response_ids)

            padded_prompt = [pad_id] * prompt_pad_len + prompt_ids
            padded_response = response_ids + [pad_id] * response_pad_len

            input_ids.append(padded_prompt + padded_response)
            attention_mask.append(
                [0] * prompt_pad_len
                + [1] * len(prompt_ids)
                + [1] * len(response_ids)
                + [0] * response_pad_len
            )
            responses.append(padded_response)
            response_mask.append([1] * len(response_ids) + [0] * response_pad_len)

        input_ids_tensor = torch.tensor(input_ids, dtype=torch.long)
        attention_mask_tensor = torch.tensor(attention_mask, dtype=torch.long)
        response_mask_tensor = torch.tensor(response_mask, dtype=torch.float32)
        position_ids = attention_mask_tensor.cumsum(dim=-1) - 1
        position_ids = position_ids.clamp_min(0)

        data = DataProto.from_dict(
            tensors={
                "input_ids": input_ids_tensor,
                "attention_mask": attention_mask_tensor,
                "position_ids": position_ids,
                "responses": torch.tensor(responses, dtype=torch.long),
            },
            meta_info=dict(self.meta_info),
        )
        return data, response_mask_tensor.to(input_ids_tensor.device)


def _normalize_context_mode(mode: str) -> str:
    normalized = mode.strip().upper()
    if normalized not in VALID_CONTEXT_MODES:
        raise ValueError(
            f"Unknown SNC context mode {mode!r}; expected one of {sorted(VALID_CONTEXT_MODES)}"
        )
    return normalized


def _lookup_sentence(sentences: Mapping[Any, Any], sid: str) -> Any:
    try:
        return sentences[sid]
    except KeyError:
        str_sid = str(sid)
        if str_sid in sentences:
            return sentences[str_sid]
        if str_sid.isdigit():
            int_sid = int(str_sid)
            if int_sid in sentences:
                return sentences[int_sid]
        raise


def _sentence_text(sentence: Any) -> str:
    if isinstance(sentence, str):
        return sentence

    if isinstance(sentence, Mapping):
        for key in ("text", "sentence", "sent_text", "content", "raw_text"):
            value = sentence.get(key)
            if value is not None:
                return str(value)
        raise KeyError("Sentence record does not include a text field")

    for attr in ("text", "sentence", "sent_text", "content", "raw_text"):
        if hasattr(sentence, attr):
            value = getattr(sentence, attr)
            if value is not None:
                return str(value)

    raise TypeError(f"Unsupported sentence record type: {type(sentence).__name__}")


def _validate_pairs(prompts: Sequence[str], gold_answers: Sequence[str]) -> None:
    if len(prompts) != len(gold_answers):
        raise ValueError(
            f"Expected the same number of prompts and gold answers, got {len(prompts)} "
            f"and {len(gold_answers)}"
        )


def _response_token_count(response: str) -> int:
    return max(len(response.split()), 1)


def _encode(tokenizer: Any, text: str, add_special_tokens: bool) -> list[int]:
    if hasattr(tokenizer, "encode"):
        return list(tokenizer.encode(text, add_special_tokens=add_special_tokens))

    encoded = tokenizer(text, add_special_tokens=add_special_tokens)
    if isinstance(encoded, Mapping):
        return list(encoded["input_ids"])
    return list(encoded.input_ids)


def _truncate_left(token_ids: list[int], max_length: int | None) -> list[int]:
    if max_length is None or len(token_ids) <= max_length:
        return token_ids
    return token_ids[-max_length:]


def _truncate_right(token_ids: list[int], max_length: int | None) -> list[int]:
    if max_length is None or len(token_ids) <= max_length:
        return token_ids
    return token_ids[:max_length]


def _pad_token_id(tokenizer: Any) -> int:
    pad_token_id = getattr(tokenizer, "pad_token_id", None)
    if pad_token_id is not None:
        return int(pad_token_id)

    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    if eos_token_id is not None:
        return int(eos_token_id)

    return 0


def _extract_log_probs(result: Any) -> Any:
    batch = getattr(result, "batch", result)
    for key in ("old_log_probs", "log_probs", "response_log_probs"):
        try:
            return batch[key]
        except (KeyError, TypeError):
            continue
    raise KeyError("compute_log_prob result does not contain response log-probabilities")
