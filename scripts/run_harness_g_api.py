#!/usr/bin/env python3
import argparse
import json
import os
import sys
import threading
from pathlib import Path
from typing import Dict, List, Optional, Union

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fastapi import FastAPI
from pydantic import BaseModel

from harness_g.env import HarnessGEpisode
from harness_g.graph_index import HarnessGGraphIndex
from harness_g.metrics import aggregate_episode_metrics


app = FastAPI(title="Harness-G v2 API")

_GRAPH_INDEX: Optional[HarnessGGraphIndex] = None
_GRAPH_DIR: Optional[Path] = None
_DATA_SOURCE = "2WikiMultiHopQA"
_SESSIONS: Dict[str, HarnessGEpisode] = {}
_EPISODE_KWARGS = {
    "paragraph_topk": 20,
    "high_conf_chunk_k": 5,
    "visible_sentence_k": 6,
    "expanded_visible_sentence_k": 6,
    "max_turns": 6,
    "qh_max_words": 16,
}
_nav_events_lock = threading.Lock()


def _nav_events_path() -> str:
    return os.environ.get("HARNESS_G_NAV_EVENTS_PATH", "")


def _run_id() -> str:
    return os.environ.get("HARNESS_G_RUN_ID", "")


def _system_tag() -> str:
    return "MENU"


def _log_nav_event(session_id: str, episode: HarnessGEpisode) -> None:
    nav = getattr(episode, "_last_nav_event", None)
    if nav is None:
        return
    # The episode keeps only a one-shot pending event. Clear it before I/O so a
    # later non-navigation step cannot re-emit stale INIT/LOOKUP state.
    episode._last_nav_event = None
    path = _nav_events_path()
    if not path:
        return
    record = {
        "run_id": _run_id(),
        "system": _system_tag(),
        "session_id": session_id,
        "action_type": nav["action_type"],
        "query_text": nav.get("query_text"),
        "result_ids": nav.get("result_ids", []),
        "new_result_ids": nav.get("new_result_ids", []),
        "turn": nav["turn"],
    }
    line = json.dumps(record, ensure_ascii=False)
    with _nav_events_lock:
        with open(path, "a", encoding="utf-8") as f:
            f.write(line + "\n")


class StepRequest(BaseModel):
    session_id: str
    question: str = ""
    action: str = ""


class BatchStepRequest(BaseModel):
    requests: List[StepRequest]


def configure_api(
    data_source: str = "2WikiMultiHopQA",
    graph_dir: Optional[Union[str, Path]] = None,
    paragraph_topk: int = 20,
    high_conf_chunk_k: int = 5,
    visible_sentence_k: int = 6,
    expanded_visible_sentence_k: int = 6,
    max_turns: int = 6,
    qh_max_words: int = 16,
) -> HarnessGGraphIndex:
    global _GRAPH_INDEX, _GRAPH_DIR, _DATA_SOURCE, _EPISODE_KWARGS
    _DATA_SOURCE = data_source
    _GRAPH_DIR = Path(graph_dir) if graph_dir else Path("expr") / data_source / "harness_g_graph"
    _GRAPH_INDEX = HarnessGGraphIndex.load(_GRAPH_DIR)
    _EPISODE_KWARGS = {
        "paragraph_topk": paragraph_topk,
        "high_conf_chunk_k": high_conf_chunk_k,
        "visible_sentence_k": visible_sentence_k,
        "expanded_visible_sentence_k": expanded_visible_sentence_k,
        "max_turns": max_turns,
        "qh_max_words": qh_max_words,
    }
    return _GRAPH_INDEX


def _require_index() -> HarnessGGraphIndex:
    if _GRAPH_INDEX is None:
        configure_api(data_source=_DATA_SOURCE, graph_dir=_GRAPH_DIR)
    assert _GRAPH_INDEX is not None
    return _GRAPH_INDEX


def reset_sessions() -> None:
    _SESSIONS.clear()


@app.get("/health")
def health():
    index = _require_index()
    return {
        "status": "ok",
        "mode": "harness_g",
        "graph_dir": str(_GRAPH_DIR),
        "data_source": _DATA_SOURCE,
        "num_paragraphs": len(index.paragraphs),
        "num_sentences": len(index.sentences),
        "num_entities": len(index.entities),
        "sessions": len(_SESSIONS),
    }


@app.post("/harness_g_step")
def harness_g_step(request: BatchStepRequest):
    index = _require_index()
    responses: List[Optional[str]] = [None] * len(request.requests)
    pending_init = []

    try:
        from harness_g.snc_api import build_snc_step_payload, snc_enabled
        snc_on = snc_enabled()
    except Exception:
        snc_on = False
    snc_payloads: List[Optional[dict]] = [None] * len(request.requests)

    for idx, item in enumerate(request.requests):
        session_id = item.session_id or f"session_{len(_SESSIONS)}"
        episode = _SESSIONS.get(session_id)
        if episode is None:
            episode = HarnessGEpisode(
                session_id=session_id,
                question=item.question,
                graph_index=index,
                **_EPISODE_KWARGS,
            )
            _SESSIONS[session_id] = episode

        action = item.action or "INIT"
        if not episode.initialized:
            action = "INIT"
        if action.strip().upper() == "INIT" and not episode.initialized:
            pending_init.append((idx, episode))
            continue
        if snc_on:
            snc_payloads[idx] = build_snc_step_payload(episode, action)
        observation = episode.step(action)
        _log_nav_event(session_id, episode)
        responses[idx] = observation

    if pending_init:
        questions = [episode.question for _, episode in pending_init]
        ranked_by_question = index.hybrid_initial_retrieve_batch(
            questions,
            paragraph_topk=_EPISODE_KWARGS["paragraph_topk"],
            high_conf_chunk_k=_EPISODE_KWARGS["high_conf_chunk_k"],
            sentence_topk=max(_EPISODE_KWARGS["visible_sentence_k"], 8),
            entity_topk=8,
            topk=_EPISODE_KWARGS["visible_sentence_k"],
        )
        for idx, episode in pending_init:
            ranked = ranked_by_question.get(episode.question)
            if ranked is None:
                ranked = index.hybrid_initial_retrieve(
                    episode.question,
                    paragraph_topk=_EPISODE_KWARGS["paragraph_topk"],
                    high_conf_chunk_k=_EPISODE_KWARGS["high_conf_chunk_k"],
                    sentence_topk=max(_EPISODE_KWARGS["visible_sentence_k"], 8),
                    entity_topk=8,
                    topk=_EPISODE_KWARGS["visible_sentence_k"],
                )
            observation = episode.init_with_ranked(ranked)
            _log_nav_event(episode.session_id, episode)
            responses[idx] = observation

    if snc_on:
        return [
            {"observation": response or "", "snc_step": snc_payloads[idx]}
            for idx, response in enumerate(responses)
        ]
    return [response or "" for response in responses]


@app.post("/harness_g_metrics")
def harness_g_metrics():
    return aggregate_episode_metrics({sid: episode.metrics for sid, episode in _SESSIONS.items()})


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the stateful Harness-G v2 API.")
    parser.add_argument("--data_source", default="2WikiMultiHopQA")
    parser.add_argument("--graph_dir", default=None)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument("--paragraph_topk", type=int, default=20)
    parser.add_argument("--high_conf_chunk_k", type=int, default=5)
    parser.add_argument("--visible_sentence_k", type=int, default=6)
    parser.add_argument("--expanded_visible_sentence_k", type=int, default=6)
    parser.add_argument("--max_turns", type=int, default=6)
    parser.add_argument("--qh_max_words", type=int, default=16)
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    configure_api(
        data_source=args.data_source,
        graph_dir=args.graph_dir,
        paragraph_topk=args.paragraph_topk,
        high_conf_chunk_k=args.high_conf_chunk_k,
        visible_sentence_k=args.visible_sentence_k,
        expanded_visible_sentence_k=args.expanded_visible_sentence_k,
        max_turns=args.max_turns,
        qh_max_words=args.qh_max_words,
    )
    import uvicorn

    uvicorn.run(app, host=args.host, port=args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
