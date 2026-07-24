import math
import re
import string
from collections import Counter
from typing import Iterable, List


STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "has",
    "he",
    "her",
    "his",
    "in",
    "is",
    "it",
    "its",
    "of",
    "on",
    "or",
    "she",
    "that",
    "the",
    "their",
    "this",
    "to",
    "was",
    "were",
    "what",
    "when",
    "where",
    "which",
    "who",
    "whom",
    "whose",
    "with",
}


def normalize_text(text: object) -> str:
    return " ".join(str(text).replace("\u00a0", " ").split())


def strip_punctuation(text: str) -> str:
    table = str.maketrans("", "", string.punctuation)
    return text.translate(table)


def normalize_for_match(text: object) -> str:
    return " ".join(strip_punctuation(str(text).lower()).split())


def tokenize(text: object, remove_stopwords: bool = True) -> List[str]:
    tokens = re.findall(r"[a-z0-9]+", str(text).lower())
    if remove_stopwords:
        return [tok for tok in tokens if tok not in STOPWORDS]
    return tokens


def lexical_vector(text: object) -> Counter:
    return Counter(tokenize(text))


def cosine_counter(lhs: Counter, rhs: Counter) -> float:
    if not lhs or not rhs:
        return 0.0
    dot = 0.0
    for token, value in lhs.items():
        if token in rhs:
            dot += value * rhs[token]
    if dot <= 0:
        return 0.0
    lhs_norm = math.sqrt(sum(value * value for value in lhs.values()))
    rhs_norm = math.sqrt(sum(value * value for value in rhs.values()))
    if lhs_norm == 0 or rhs_norm == 0:
        return 0.0
    return float(dot / (lhs_norm * rhs_norm))


def lexical_score(query: object, text: object) -> float:
    return cosine_counter(lexical_vector(query), lexical_vector(text))


def compact_keywords(text: object, max_words: int = 16) -> str:
    seen = set()
    words = []
    for token in tokenize(text):
        if token in seen:
            continue
        seen.add(token)
        words.append(token)
        if len(words) >= max_words:
            break
    return " ".join(words)


def contains_any_answer(text: object, answers: Iterable[object]) -> bool:
    normalized_text = normalize_for_match(text)
    for answer in answers:
        normalized_answer = normalize_for_match(answer)
        if normalized_answer and normalized_answer in normalized_text:
            return True
    return False
