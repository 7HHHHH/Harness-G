from pathlib import Path
from typing import List

from .graph_index import HarnessGGraphIndex


def manifest_summary(index: HarnessGGraphIndex) -> str:
    manifest = index.manifest
    keys = [
        "graph_version",
        "graph_storage",
        "graph_type",
        "relation_extraction",
        "llm_graph_construction",
        "hyperedges",
        "graph_directed",
        "num_paragraphs",
        "num_passages",
        "num_sentences",
        "num_entities",
        "num_mentions",
        "num_ps_edges",
        "num_pe_edges",
        "num_se_edges",
        "num_sentence_sentence_edges",
        "num_entity_synonym_edges",
        "entity_synonym_method",
        "entity_extractor",
        "corpus_path",
    ]
    return "\n".join(f"{key}: {manifest.get(key)}" for key in keys)


def preview(index: HarnessGGraphIndex, limit: int = 10) -> str:
    lines = ["Manifest summary:", manifest_summary(index), "", "Paragraphs:"]
    for paragraph in list(index.paragraphs.values())[:limit]:
        lines.append(f"{paragraph['pid']} | title: {paragraph.get('title', '')} | {paragraph.get('text', '')[:180]}")

    lines.append("")
    lines.append("Sentences:")
    for sentence in list(index.sentences.values())[:limit]:
        lines.append(f"{sentence['sid']} | pid: {sentence['pid']} | {sentence.get('text', '')}")

    lines.append("")
    lines.append("Entities:")
    for entity in list(index.entities.values())[:limit]:
        forms = ", ".join(entity.get("surface_forms", []))
        lines.append(f"{entity['eid']} | {entity['canonical']} | {entity['label']} | {forms}")
    return "\n".join(lines)


def inspect_entity(index: HarnessGGraphIndex, entity_query: str, limit: int = 10) -> str:
    needle = entity_query.lower()
    matches = [
        entity
        for entity in index.entities.values()
        if needle in entity.get("canonical", "").lower()
    ][:limit]

    lines = [f"Entity matches for: {entity_query}"]
    if not matches:
        lines.append("No matching entities.")
        return "\n".join(lines)

    for entity in matches:
        lines.append("")
        lines.append(f"{entity['eid']} | {entity['canonical']} | {entity['label']}")
        lines.append(f"surface_forms: {', '.join(entity.get('surface_forms', []))}")
        for sentence in index.get_sentences_for_entity(entity["eid"])[:limit]:
            paragraph = index.paragraphs.get(sentence["pid"], {})
            lines.append(f"- {sentence['sid']} | paragraph: {paragraph.get('title', '')} | {sentence.get('text', '')}")
    return "\n".join(lines)


def inspect_sentence(index: HarnessGGraphIndex, sid: str) -> str:
    sentence = index.sentences.get(sid)
    if sentence is None:
        return f"Sentence not found: {sid}"

    paragraph = index.paragraphs.get(sentence["pid"], {})
    lines = [
        f"{sid}",
        f"paragraph: {sentence['pid']} | title: {paragraph.get('title', '')}",
        f"text: {sentence.get('text', '')}",
        "entities:",
    ]
    for entity in index.get_entities_for_sentence(sid):
        lines.append(f"- {entity['eid']} | {entity['canonical']} | {entity['label']}")
    return "\n".join(lines)


def inspect_paragraph(index: HarnessGGraphIndex, pid: str, limit: int = 50) -> str:
    paragraph = index.paragraphs.get(pid)
    if paragraph is None:
        return f"Paragraph not found: {pid}"

    lines = [
        f"{pid}",
        f"title: {paragraph.get('title', '')}",
        f"text: {paragraph.get('text', '')}",
        "sentences:",
    ]
    seen_entities = set()
    for sentence in index.get_sentences_for_paragraphs([pid])[:limit]:
        lines.append(f"- {sentence['sid']} | {sentence.get('text', '')}")
        for entity in index.get_entities_for_sentence(sentence["sid"]):
            seen_entities.add(entity["eid"])

    lines.append("entities:")
    for eid in sorted(seen_entities):
        entity = index.entities[eid]
        lines.append(f"- {entity['eid']} | {entity['canonical']} | {entity['label']}")
    return "\n".join(lines)
