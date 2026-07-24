import json
import re
from pathlib import Path
from typing import Iterable, Iterator, List, Optional, Tuple, Union

from .text_utils import normalize_text


DEFAULT_CORPUS_CANDIDATES = (
    "datasets/{data_source}/corpus.jsonl",
    "datasets/{data_source}/raw/corpus.jsonl",
    "datasets/{data_source}/raw/passages.jsonl",
    "datasets/{data_source}/raw/context.json",
    "datasets/{data_source}/raw/contexts.json",
    "datasets/{data_source}/raw/qa_train.json",
    "datasets/{data_source}/raw/qa_dev.json",
    "datasets/{data_source}/raw/qa_test.json",
)


TEXT_FIELDS = ("contents", "text", "paragraph", "passage", "context")
TITLE_FIELDS = ("title", "document_title")
ID_FIELDS = ("id", "doc_id", "_id")
QA_CONTEXT_FIELDS = ("contexts", "ctxs", "context", "paragraphs", "supporting_facts", "evidences")


def resolve_corpus_path(data_source: str, corpus_path: Optional[Union[str, Path]] = None) -> Tuple[Path, List[str]]:
    warnings: List[str] = []
    if corpus_path:
        path = Path(corpus_path)
        if not path.exists():
            raise FileNotFoundError(f"corpus_path does not exist: {path}")
        return path, warnings

    for template in DEFAULT_CORPUS_CANDIDATES:
        candidate = Path(template.format(data_source=data_source))
        if candidate.exists():
            return candidate, warnings

    raise FileNotFoundError(
        "No Harness-G corpus file found. Tried: "
        + ", ".join(template.format(data_source=data_source) for template in DEFAULT_CORPUS_CANDIDATES)
    )


def load_json_or_jsonl(path: Path) -> Iterable[dict]:
    if path.suffix.lower() == ".jsonl":
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    yield json.loads(line)
        return

    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        for row in data:
            if isinstance(row, dict):
                yield row
    elif isinstance(data, dict):
        rows = data.get("data")
        if isinstance(rows, list):
            for row in rows:
                if isinstance(row, dict):
                    yield row
        else:
            for key, value in data.items():
                if isinstance(value, dict):
                    row = dict(value)
                    row.setdefault("id", key)
                    yield row


def _first_field(row: dict, fields: Iterable[str], default: object = "") -> object:
    for field in fields:
        value = row.get(field)
        if value is not None and value != "":
            return value
    return default


def _record_from_plain_row(row: dict, auto_idx: int) -> Optional[dict]:
    text = _first_field(row, TEXT_FIELDS, "")
    if isinstance(text, list):
        text = " ".join(str(item) for item in text)
    text = normalize_text(text)
    if not text:
        return None

    doc_id = _first_field(row, ID_FIELDS, auto_idx)
    title = _first_field(row, TITLE_FIELDS, "")
    return {
        "doc_id": str(doc_id),
        "title": str(title or ""),
        "text": text,
        "source": "corpus",
    }


# A line that is entirely a quoted title, e.g. '"Etan Boritzer"'. Used to split
# packed corpus shards (one corpus.jsonl record holds ~100+ documents joined as
# '"Title"\nbody\n"Title2"\nbody2...').
_TITLE_LINE_RE = re.compile(r'^"([^"]{1,90})"\s*$')


def _split_packed_contents(raw_text: str) -> List[Tuple[str, str]]:
    """Split a packed 'contents' field into per-document ``(title, body)`` pairs.

    Three shapes are handled:
      * **Group A** — quoted-title blocks (2Wiki / PopQA / NQ / TriviaQA): each
        ``"Title"`` line starts a document; the title is extracted and the quote
        line dropped from the body (this removes the glued-title artifacts at the
        source).
      * **Group B** — newline-delimited bodies with no title (HotpotQA / Musique).
      * **Single document** — returned as one ``("", body)`` pair.

    Must run on the RAW text (before whitespace normalization) so newline
    document boundaries survive.
    """
    if not raw_text:
        return []
    lines = str(raw_text).split("\n")
    title_idxs = [i for i, ln in enumerate(lines) if _TITLE_LINE_RE.match(ln.strip())]
    if len(title_idxs) >= 2:
        docs: List[Tuple[str, str]] = []
        for k, start in enumerate(title_idxs):
            title = _TITLE_LINE_RE.match(lines[start].strip()).group(1).strip()
            end = title_idxs[k + 1] if k + 1 < len(title_idxs) else len(lines)
            body = " ".join(lines[start + 1:end])
            docs.append((title, body))
        return docs

    segments = [ln for ln in lines if ln.strip()]
    if len(segments) >= 2:
        return [("", seg) for seg in segments]
    return [("", raw_text)]


def _records_from_plain_row(row: dict, auto_idx: int) -> Iterator[dict]:
    """Yield one record per document, expanding packed multi-doc shards."""
    raw_text = _first_field(row, TEXT_FIELDS, "")
    if isinstance(raw_text, list):
        raw_text = "\n".join(str(item) for item in raw_text)
    raw_text = str(raw_text)
    if not raw_text.strip():
        return

    row_title = str(_first_field(row, TITLE_FIELDS, "") or "")
    doc_id = _first_field(row, ID_FIELDS, auto_idx)
    docs = _split_packed_contents(raw_text)
    multi = len(docs) > 1
    for k, (title, body) in enumerate(docs):
        body = normalize_text(body)
        if not body:
            continue
        sub_id = f"{doc_id}__{k}" if multi else str(doc_id)
        yield {
            "doc_id": sub_id,
            "title": str(title or row_title or ""),
            "text": body,
            "source": "corpus",
        }


def _context_to_records(value: object, example_idx: int) -> Iterator[dict]:
    if value is None:
        return
    if isinstance(value, str):
        text = normalize_text(value)
        if text:
            title = ""
            if text.startswith('"') and '"\n' in text:
                maybe_title, rest = text.split("\n", 1)
                title = maybe_title.strip('" ')
                text = normalize_text(rest)
            yield {
                "doc_id": f"qa_{example_idx}",
                "title": title,
                "text": text,
                "source": "qa_context",
            }
        return
    if isinstance(value, dict):
        record = _record_from_plain_row(value, example_idx)
        if record is not None:
            record["source"] = "qa_context"
            yield record
        return
    if isinstance(value, (list, tuple)):
        # Common shapes: [title, text], [[title, sents], ...], ["title\ntext", ...].
        if len(value) == 2 and isinstance(value[0], str) and isinstance(value[1], (str, list, tuple)):
            title = value[0]
            text_value = value[1]
            text = " ".join(str(item) for item in text_value) if isinstance(text_value, (list, tuple)) else text_value
            text = normalize_text(text)
            if text:
                yield {
                    "doc_id": f"qa_{example_idx}_{abs(hash(title)) % 1000000}",
                    "title": title,
                    "text": text,
                    "source": "qa_context",
                }
            return
        for item in value:
            yield from _context_to_records(item, example_idx)


def _records_from_qa_row(row: dict, example_idx: int) -> Iterator[dict]:
    found_context = False
    for field in QA_CONTEXT_FIELDS:
        if field not in row:
            continue
        found_context = True
        yield from _context_to_records(row.get(field), example_idx)
    if not found_context:
        return


def iter_corpus_records(
    corpus_path: Union[str, Path],
    max_docs: Optional[int] = None,
) -> Tuple[Iterator[dict], List[str], str]:
    path = Path(corpus_path)
    rows = list(load_json_or_jsonl(path))
    warnings: List[str] = []
    source_type = "jsonl_corpus" if path.suffix.lower() == ".jsonl" else "json_corpus"

    def iterator() -> Iterator[dict]:
        count = 0
        for auto_idx, row in enumerate(rows):
            if max_docs is not None and count >= max_docs:
                break

            emitted_plain = 0
            for record in _records_from_plain_row(row, auto_idx):
                if max_docs is not None and count >= max_docs:
                    break
                emitted_plain += 1
                yield record
                count += 1
            if emitted_plain:
                continue

            emitted = 0
            for record in _records_from_qa_row(row, auto_idx):
                if max_docs is not None and count >= max_docs:
                    break
                if record.get("text"):
                    emitted += 1
                    yield record
                    count += 1
            if emitted == 0 and ("question" in row or "answers" in row or "golden_answers" in row):
                warnings.append(
                    f"example {auto_idx} has question/answers but no usable corpus text; skipped"
                )

    if any("question" in row for row in rows):
        source_type = "qa_file_with_contexts"
    return iterator(), warnings, source_type
