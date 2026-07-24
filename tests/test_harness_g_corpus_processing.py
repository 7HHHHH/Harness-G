"""Tests for corpus/text processing: sentence splitting, packed-shard splitting,
entity-junk filtering, and display-line title dedup."""

from harness_g.corpus_loader import _records_from_plain_row, _split_packed_contents
from harness_g.formatting import selected_line, sentence_line
from harness_g.utils import is_junk_entity_surface, split_sentences


def test_split_sentences_protects_initials_and_abbreviations():
    assert split_sentences(
        "Leslie H. Martinson was an American director. He directed Batman."
    ) == ["Leslie H. Martinson was an American director.", "He directed Batman."]
    # saints / titles are not sentence boundaries
    assert split_sentences("St. Mary School was founded by Dr. Smith.") == [
        "St. Mary School was founded by Dr. Smith."
    ]
    assert split_sentences("Wenn V. Deramas made films.") == ["Wenn V. Deramas made films."]
    # normal multi-sentence text still splits
    assert split_sentences("A cat sat. The dog ran.") == ["A cat sat.", "The dog ran."]


def test_is_junk_entity_surface():
    for junk in ["1962", " 42 ", "S", "Leslie H", "Wenn V", 'Patiala" Career Academy School']:
        assert is_junk_entity_surface(junk), junk
    for keep in ["Black Gold", "Leslie H. Martinson", "Etan Boritzer", ""]:
        assert not is_junk_entity_surface(keep), keep


def test_split_packed_contents_group_a_quoted_titles():
    raw = '"Title One"\nBody one sentence.\n"Title Two"\nBody two sentence.'
    assert _split_packed_contents(raw) == [
        ("Title One", "Body one sentence."),
        ("Title Two", "Body two sentence."),
    ]


def test_split_packed_contents_group_b_newline_only():
    raw = "Doc one body.\nDoc two body.\nDoc three body."
    assert _split_packed_contents(raw) == [
        ("", "Doc one body."),
        ("", "Doc two body."),
        ("", "Doc three body."),
    ]


def test_split_packed_contents_single_doc():
    assert _split_packed_contents("Just one document body.") == [("", "Just one document body.")]


def test_records_from_plain_row_expands_packed_shard():
    row = {"id": "7", "contents": '"Alpha"\nAlpha body.\n"Beta"\nBeta body.'}
    recs = list(_records_from_plain_row(row, 0))
    assert [r["title"] for r in recs] == ["Alpha", "Beta"]
    assert [r["doc_id"] for r in recs] == ["7__0", "7__1"]
    assert recs[0]["text"] == "Alpha body."


def test_records_from_plain_row_single_doc_uses_title_field():
    row = {"id": "1", "title": "Ada Lovelace", "contents": "Ada Lovelace was born in London."}
    recs = list(_records_from_plain_row(row, 0))
    assert len(recs) == 1
    assert recs[0]["title"] == "Ada Lovelace"
    assert recs[0]["doc_id"] == "1"


def test_sentence_line_dedups_title_and_strips_quotes():
    assert (
        sentence_line("S1", {"title": "Etan Boritzer", "text": "Etan Boritzer was a writer."})
        == "S1 | Etan Boritzer was a writer."
    )
    assert (
        sentence_line("S2", {"title": "Theodred II", "text": "He was a bishop."})
        == "S2 | Theodred II: He was a bishop."
    )
    assert selected_line("S3", {"title": "", "text": '"a quoted opener'}) == "S3 | a quoted opener"
