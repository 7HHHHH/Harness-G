#!/usr/bin/env python
"""Evaluate QA result files with an OpenAI-compatible judge.

The API key is read from an environment variable and is never written to disk.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import requests


ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL | re.IGNORECASE)


def extract_answer(prediction: str) -> str:
    matches = ANSWER_RE.findall(prediction or "")
    if matches:
        return re.sub(r"\s+", " ", matches[-1]).strip()
    return re.sub(r"\s+", " ", (prediction or "")).strip()[-500:]


def parse_judge_json(text: str) -> dict[str, Any]:
    text = (text or "").strip()
    candidates = [text]
    fenced = re.findall(r"```(?:json)?\s*(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    candidates.extend(chunk.strip() for chunk in fenced)
    obj_match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if obj_match:
        candidates.append(obj_match.group(0))
    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            continue
    lowered = text.lower()
    return {"correct": "true" in lowered and "false" not in lowered, "reason": text[:300]}


def judge_one(
    *,
    item: dict[str, Any],
    index: int,
    base_url: str,
    api_key: str,
    model: str,
    timeout: float,
    retries: int,
) -> dict[str, Any]:
    question = item.get("question", "")
    gold = item.get("golden_answers", [])
    prediction = extract_answer(item.get("prediction", ""))
    url = base_url.rstrip("/") + "/chat/completions"
    messages = [
        {
            "role": "system",
            "content": (
                "You are a strict QA evaluator. Decide whether the predicted answer "
                "correctly answers the question and is semantically equivalent to at "
                "least one gold answer. Accept aliases and minor wording differences. "
                "Return only compact JSON with keys correct and reason."
            ),
        },
        {
            "role": "user",
            "content": json.dumps(
                {
                    "question": question,
                    "gold_answers": gold,
                    "predicted_answer": prediction,
                },
                ensure_ascii=False,
            ),
        },
    ]
    payload = {
        "model": model,
        "messages": messages,
        "temperature": 0,
        "max_tokens": 128,
        "response_format": {"type": "json_object"},
    }
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    last_error = None
    for attempt in range(retries + 1):
        try:
            response = requests.post(url, headers=headers, json=payload, timeout=timeout)
            response.raise_for_status()
            data = response.json()
            content = data["choices"][0]["message"]["content"]
            judged = parse_judge_json(content)
            return {
                "index": index,
                "question": question,
                "golden_answers": gold,
                "predicted_answer": prediction,
                "correct": bool(judged.get("correct", False)),
                "reason": str(judged.get("reason", ""))[:500],
            }
        except Exception as exc:  # noqa: BLE001 - preserve retry detail in output.
            last_error = str(exc)
            if attempt < retries:
                time.sleep(min(2**attempt, 8))
    return {
        "index": index,
        "question": question,
        "golden_answers": gold,
        "predicted_answer": prediction,
        "correct": False,
        "error": last_error,
    }


def load_completed(path: Path) -> dict[int, dict[str, Any]]:
    completed: dict[int, dict[str, Any]] = {}
    if not path.exists():
        return completed
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            completed[int(row["index"])] = row
    return completed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_file", required=True)
    parser.add_argument("--output_file", required=True)
    parser.add_argument("--model", default=os.environ.get("OPENAI_MODEL", "gpt-4o-mini"))
    parser.add_argument("--base_url", default=os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"))
    parser.add_argument("--api_key_env", default="OPENAI_API_KEY")
    parser.add_argument("--limit", type=int, default=-1)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--no_resume", action="store_true")
    args = parser.parse_args()

    api_key = os.environ.get(args.api_key_env)
    if not api_key:
        raise SystemExit(f"missing API key env var: {args.api_key_env}")

    results_file = Path(args.results_file)
    output_file = Path(args.output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    details_file = output_file.with_suffix(".jsonl")
    data = json.loads(results_file.read_text(encoding="utf-8"))
    if args.limit > 0:
        data = data[: args.limit]

    completed = {} if args.no_resume else load_completed(details_file)
    pending = [(idx, item) for idx, item in enumerate(data) if idx not in completed]

    with details_file.open("a", encoding="utf-8") as out:
        with ThreadPoolExecutor(max_workers=max(args.workers, 1)) as pool:
            futures = [
                pool.submit(
                    judge_one,
                    item=item,
                    index=idx,
                    base_url=args.base_url,
                    api_key=api_key,
                    model=args.model,
                    timeout=args.timeout,
                    retries=args.retries,
                )
                for idx, item in pending
            ]
            for fut in as_completed(futures):
                row = fut.result()
                completed[int(row["index"])] = row
                out.write(json.dumps(row, ensure_ascii=False) + "\n")
                out.flush()

    ordered = [completed[idx] for idx in range(len(data)) if idx in completed]
    correct = sum(1 for row in ordered if row.get("correct"))
    failed = sum(1 for row in ordered if row.get("error"))
    summary = {
        "results_file": str(results_file),
        "model": args.model,
        "base_url": args.base_url,
        "count": len(ordered),
        "llm_correct": correct,
        "llm_accuracy": correct / len(ordered) if ordered else 0.0,
        "failed_calls": failed,
        "details_file": str(details_file),
    }
    output_file.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
