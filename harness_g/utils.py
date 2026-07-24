import json
import re
from collections import OrderedDict
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Tuple


ENTITY_LABELS_FOR_SPACY = {
    "PERSON",
    "ORG",
    "GPE",
    "LOC",
    "WORK_OF_ART",
    "EVENT",
    "DATE",
    "NORP",
    "FAC",
}

MONTHS = (
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
    "Jan",
    "Feb",
    "Mar",
    "Apr",
    "Jun",
    "Jul",
    "Aug",
    "Sep",
    "Sept",
    "Oct",
    "Nov",
    "Dec",
)

STOP_ENTITY_SURFACES = {
    "The",
    "A",
    "An",
    "He",
    "She",
    "It",
    "They",
    "This",
    "That",
    "These",
    "Those",
    "In",
    "On",
    "At",
    "For",
    "By",
    "As",
    "From",
    "Question",
}


def parse_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    lowered = str(value).strip().lower()
    if lowered in {"true", "1", "yes", "y", "on"}:
        return True
    if lowered in {"false", "0", "no", "n", "off"}:
        return False
    raise ValueError(f"Cannot parse boolean value: {value!r}")


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def normalize_ws(text: str) -> str:
    return " ".join(str(text).replace("\u00a0", " ").split())


def canonicalize_entity(surface: str) -> str:
    return normalize_ws(surface).strip(" \t\n\r\"'`.,;:!?()[]{}").lower()


def tokenize_for_search(text: str) -> List[str]:
    return re.findall(r"[a-z0-9]+", str(text).lower())


_FIRST_NAME_INITIAL_RE = re.compile(r"^\S+\s+[A-Za-z]$")


def is_junk_entity_surface(surface: str) -> bool:
    """Surface forms that make useless graph nodes / navigation anchors.

    Filters pure numbers (years/counts), single characters, ``first-name +
    lone trailing initial`` fragments left by sentence-split errors
    (e.g. ``Leslie H``, ``Wenn V``), and surfaces carrying a dangling
    double-quote (corpus de-tokenization / glued-title artifact, e.g.
    ``Patiala" Career Academy School``). Empty surfaces are not "junk", just
    absent, so they return False.
    """
    s = (surface or "").strip()
    if not s:
        return False
    if s.isdigit():
        return True
    if len(s) <= 1:
        return True
    if '"' in s:
        return True
    if _FIRST_NAME_INITIAL_RE.match(s):
        return True
    return False


# Entity ``label`` values that name a literal/attribute value rather than a
# navigable entity: looking these up never reveals new chain evidence (the value
# is already on the page that mentions it). They are still legal *evidence* and
# may be SELECT-ed / ANSWER_WITH-ed; they are only barred as LOOKUP targets.
BAD_LOOKUP_TARGET_LABELS = {
    "DATE",
    "TIME",
    "CARDINAL",
    "ORDINAL",
    "QUANTITY",
    "PERCENT",
    "MONEY",
    "NORP",  # nationalities / religious / political groups (American, British, ...)
}

# Lowercase nationality / demonym adjectives. Rule-based extraction labels these
# ``ENTITY`` (not ``NORP``), so they need an explicit surface check.
_NATIONALITY_ADJECTIVES = {
    "american", "british", "english", "french", "german", "italian", "spanish",
    "russian", "chinese", "japanese", "korean", "indian", "canadian", "australian",
    "dutch", "swedish", "norwegian", "danish", "finnish", "polish", "greek",
    "turkish", "egyptian", "brazilian", "mexican", "argentine", "argentinian",
    "irish", "scottish", "welsh", "swiss", "austrian", "belgian", "portuguese",
    "iranian", "iraqi", "israeli", "saudi", "pakistani", "bangladeshi", "thai",
    "vietnamese", "indonesian", "filipino", "malaysian", "singaporean",
    "ukrainian", "romanian", "hungarian", "czech", "slovak", "bulgarian",
    "croatian", "serbian", "europe", "european", "asian", "african",
}


def is_bad_lookup_target(surface: str, label: str = "") -> bool:
    """True when an entity must not be offered as a V3 ``LOOKUP`` target.

    Combines :func:`is_junk_entity_surface` with label-based filtering of
    literal/attribute values (dates, cardinals, money, nationality groups, ...)
    and a lowercase nationality-adjective surface check for rule-based extracts
    that lack fine-grained labels. Filtered entities can still be selected as
    evidence; they are only barred from being navigation targets.
    """
    s = (surface or "").strip()
    if is_junk_entity_surface(s):
        return True
    if str(label or "").strip().upper() in BAD_LOOKUP_TARGET_LABELS:
        return True
    if s.lower() in _NATIONALITY_ADJECTIVES:
        return True
    return False


def read_jsonl(path: Path) -> Iterator[dict]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, data) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")


def iter_corpus_records(corpus_path: Path, max_docs: Optional[int] = None) -> Iterator[dict]:
    count = 0
    for auto_idx, raw in enumerate(read_jsonl(corpus_path)):
        if max_docs is not None and count >= max_docs:
            break

        doc_id = raw.get("id")
        if doc_id is None or doc_id == "":
            doc_id = raw.get("doc_id", auto_idx)
        title = raw.get("title", "") or ""
        text = ""
        for field in ("contents", "text", "paragraph"):
            candidate = raw.get(field, "")
            if candidate:
                text = candidate
                break
        text = normalize_ws(text)
        if not text:
            continue

        yield {
            "doc_id": str(doc_id),
            "title": str(title),
            "text": text,
        }
        count += 1


# Abbreviations whose trailing period must not end a sentence.
_SENTENCE_ABBREVIATIONS = (
    "dr", "mr", "mrs", "ms", "mx", "prof", "rev", "fr", "st", "sr", "jr",
    "gen", "col", "capt", "lt", "sgt", "maj", "sen", "rep", "gov", "hon",
    "pres", "vs", "etc", "al", "inc", "ltd", "co", "corp", "no", "vol",
    "mt", "ft", "messrs", "mme", "mlle",
)
_ABBREV_RE = re.compile(r"\b(?:" + "|".join(_SENTENCE_ABBREVIATIONS) + r")\.", re.IGNORECASE)
# Single capital initial that precedes a capitalized token, e.g. "H. Martinson".
_INITIAL_RE = re.compile(r"\b[A-Z]\.(?=\s+[A-Z])")
# Dotted acronyms such as "U.S.", "U.K.".
_DOTTED_ACRONYM_RE = re.compile(r"\b(?:[A-Za-z]\.){2,}")
_DOT_PLACEHOLDER = chr(0xE000)


def _protect_abbreviation_dots(text: str) -> str:
    def _mask(match: "re.Match[str]") -> str:
        return match.group(0).replace(".", _DOT_PLACEHOLDER)

    text = _DOTTED_ACRONYM_RE.sub(_mask, text)
    text = _INITIAL_RE.sub(_mask, text)
    text = _ABBREV_RE.sub(_mask, text)
    return text


def split_sentences(text: str) -> List[str]:
    text = normalize_ws(text)
    if not text:
        return []

    # Keep this regex path as the default to avoid external tokenizer data
    # downloads in fresh environments. Protect abbreviation / initial periods
    # first so "Leslie H. Martinson", "St. Mary", "Dr. Smith" and "U.S." are
    # not split mid-name, then restore them after splitting.
    protected = _protect_abbreviation_dots(text)
    pieces = re.split(r"(?<=[.!?])\s+(?=[\"'(\[]?[A-Z0-9])", protected)
    sentences = [normalize_ws(piece.replace(_DOT_PLACEHOLDER, ".")).strip() for piece in pieces]
    return [s for s in sentences if s]


def _dedupe_mentions(mentions: Iterable[Tuple[str, str]]) -> List[Tuple[str, str]]:
    seen = set()
    output = []
    for surface, label in mentions:
        surface = normalize_ws(surface).strip(" \t\n\r\"'`.,;:!?()[]{}")
        if not surface:
            continue
        key = (surface, label)
        canonical = canonicalize_entity(surface)
        if not canonical or canonical in {"the", "a", "an"}:
            continue
        if key not in seen:
            seen.add(key)
            output.append(key)
    return output


def extract_entities_rule_based(sentence: str) -> List[Tuple[str, str]]:
    mentions: List[Tuple[str, str]] = []
    month_alt = "|".join(MONTHS)

    date_patterns = [
        rf"\b(?:{month_alt})\s+\d{{1,2}},?\s+\d{{4}}\b",
        rf"\b\d{{1,2}}\s+(?:{month_alt})\s+\d{{4}}\b",
        rf"\b(?:{month_alt})\s+\d{{4}}\b",
        r"\b(?:1[5-9]\d{2}|20\d{2})\b",
    ]
    for pattern in date_patterns:
        for match in re.finditer(pattern, sentence):
            mentions.append((match.group(0), "DATE"))

    for match in re.finditer(r"\b(?:[A-Z]\.){2,}|\b[A-Z]{2,}(?:-[A-Z0-9]+)*\b", sentence):
        mentions.append((match.group(0), "ENTITY"))

    cap_word = r"[A-Z][A-Za-z]+(?:[-'][A-Za-z]+)*"
    connector = r"(?:of|the|and|for|in|on|at|de|da|du|van|von|la|le|el|&)"
    phrase_pattern = rf"\b{cap_word}(?:\s+(?:{connector}|{cap_word}))*\b"
    for match in re.finditer(phrase_pattern, sentence):
        surface = match.group(0).strip()
        if surface in STOP_ENTITY_SURFACES:
            continue
        cap_count = len(re.findall(cap_word, surface))
        if cap_count >= 2 or (cap_count == 1 and len(surface) >= 3 and match.start() > 0):
            mentions.append((surface, "ENTITY"))

    return _dedupe_mentions(mentions)


def load_spacy_model(model_name: str, use_gpu: bool = False):
    try:
        import spacy
    except Exception:
        return None
    if use_gpu:
        try:
            spacy.require_gpu()
        except Exception:
            try:
                spacy.prefer_gpu()
            except Exception:
                pass
    try:
        return spacy.load(model_name, disable=["tagger", "parser", "attribute_ruler", "lemmatizer"])
    except Exception:
        return None


def extract_entities_from_spacy_doc(doc) -> List[Tuple[str, str]]:
    mentions = []
    for ent in doc.ents:
        if ent.label_ in ENTITY_LABELS_FOR_SPACY:
            mentions.append((ent.text, ent.label_))
    return _dedupe_mentions(mentions)


def extract_entities_spacy(sentence: str, nlp) -> List[Tuple[str, str]]:
    doc = nlp(sentence)
    return extract_entities_from_spacy_doc(doc)


def append_unique(items: List[str], value: str) -> None:
    if value not in items:
        items.append(value)


def ordered_set(values: Iterable[str]) -> List[str]:
    return list(OrderedDict((value, None) for value in values).keys())
