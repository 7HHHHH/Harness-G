import json
from typing import Iterable, List

from .graph_index import HarnessGGraphIndex


def retrieve_query(
    index: HarnessGGraphIndex,
    query: str,
    top_paragraphs: int = 5,
    top_sentences: int = 8,
) -> dict:
    paragraphs = index.search_paragraphs(query, topk=top_paragraphs)
    pids = [paragraph["pid"] for paragraph in paragraphs]
    candidate_sentences = index.get_sentences_for_paragraphs(pids)
    candidate_sids = [sentence["sid"] for sentence in candidate_sentences]
    ranked_sentences = index.rank_sentences(query, candidate_sids, topk=top_sentences)

    entities = []
    seen_entities = set()
    for sentence in ranked_sentences:
        for entity in index.get_entities_for_sentence(sentence["sid"]):
            if entity["eid"] not in seen_entities:
                seen_entities.add(entity["eid"])
                entities.append(entity)

    return {
        "query": query,
        "paragraphs": paragraphs,
        "sentences": ranked_sentences,
        "entities": entities,
    }


def format_harness_g_knowledge(
    index: HarnessGGraphIndex,
    query: str,
    top_paragraphs: int = 5,
    top_sentences: int = 8,
) -> str:
    result = retrieve_query(index, query, top_paragraphs, top_sentences)
    lines: List[str] = [
        "[HARNESS_G_KNOWLEDGE]",
        f"query: {query}",
        "top_paragraphs:",
    ]

    for i, paragraph in enumerate(result["paragraphs"]):
        title = paragraph.get("title", "")
        lines.append(f"P{i} | title: {title}")

    lines.append("")
    lines.append("evidence_sentences:")
    for i, sentence in enumerate(result["sentences"]):
        title = sentence.get("title", "")
        text = sentence.get("text", "")
        lines.append(f"S{i} | title: {title} | {text}")

    lines.append("")
    lines.append("entities:")
    for i, entity in enumerate(result["entities"]):
        lines.append(f"E{i} | {entity['canonical']} | {entity['label']}")

    lines.append("[/HARNESS_G_KNOWLEDGE]")
    return "\n".join(lines)


def queries_to_json_strings(
    index: HarnessGGraphIndex,
    queries: Iterable[str],
    top_paragraphs: int = 5,
    top_sentences: int = 8,
) -> List[str]:
    return [
        json.dumps(
            {
                "results": format_harness_g_knowledge(
                    index=index,
                    query=query,
                    top_paragraphs=top_paragraphs,
                    top_sentences=top_sentences,
                )
            },
            ensure_ascii=False,
        )
        for query in queries
    ]
