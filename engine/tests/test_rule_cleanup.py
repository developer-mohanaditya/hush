"""Unit tests for deterministic rule-based transcript cleanup."""

from sottovox_engine.rule_cleanup import clean_transcript_rules


def test_empty_and_whitespace() -> None:
    assert clean_transcript_rules("   ") == ""


def test_filler_and_stutter_removal() -> None:
    raw = "um, the the umbrella is on the table, uhh you know."
    assert clean_transcript_rules(raw) == "The umbrella is on the table."


def test_spoken_formatting_and_sentence_case() -> None:
    raw = "hello world. new line second sentence! new paragraph third paragraph"
    expected = "Hello world.\nSecond sentence!\n\nThird paragraph"
    assert clean_transcript_rules(raw) == expected


def test_custom_terms_replacement() -> None:
    raw = "open sotto vox now"
    cleaned = clean_transcript_rules(raw, custom_terms={"sotto vox": "Hush"})
    assert cleaned == "Open Hush now"


def test_pronoun_i_and_sentence_case() -> None:
    raw = "yeah, i think i'm ready. are you?"
    assert clean_transcript_rules(raw) == "Yeah, I think I'm ready. Are you?"
