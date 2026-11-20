import sys
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
from praatio import textgrid

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.analyze import (compare_boundaries, inspect_grid, match_words, read_textgrid,
                             silence_durations, timestamp_issues, word_intervals)
from scripts.dictionary import coverage, missing_words, read_dictionary, validate_phones
from scripts.prepare_data import index_files, normalize, prepare


def test_pairing_and_duplicate_pcm(tmp_path):
    audio = tmp_path / "wav"
    texts = tmp_path / "transcripts"
    audio.mkdir()
    texts.mkdir()
    samples = np.arange(1600, dtype=np.int16)
    for name in ("First", "second"):
        sf.write(audio / f"{name}.wav", samples, 16000, subtype="PCM_16")
        (texts / f"{name.upper()}.TXT").write_text("I said bed.")
    (texts / "orphan.txt").write_text("unmatched")
    corpus = tmp_path / "mfa"
    rows, tokens, issues = prepare(audio, texts, corpus)
    assert issues == ["orphan"]
    assert tokens == {"First": ["i", "said", "bed"], "second": ["i", "said", "bed"]}
    assert [r["duplicate_of"] for r in rows if r["status"] == "paired"] == ["", "First"]
    assert sf.read(corpus / "First/First.wav", dtype="int16")[0].tolist() == samples.tolist()


def test_duplicate_identifier(tmp_path):
    (tmp_path / "a.wav").touch()
    (tmp_path / "A.WAV").touch()
    with pytest.raises(ValueError, match="Duplicate identifier"):
        index_files(tmp_path, ".wav")


def test_normalization():
    assert normalize("S.J.C.’s, in nineteen seventy- six; WBUR’s de-politicize.") == (
        "s j c's in nineteen seventy six w b u r's depoliticize"
    )
    assert normalize("I\r\ncan’t go.") == "i can't go"


def test_dictionary_and_coverage(tmp_path):
    dictionary = tmp_path / "example.dict"
    dictionary.write_text("bed\t0.99\t0.12\t0.42\t0.95\tb ɛ d\nbed\tb ɛ t\n's s\n", encoding="utf-8")
    entries = read_dictionary(dictionary)
    assert [phones for _, phones in entries["bed"]] == [["b", "ɛ", "d"], ["b", "ɛ", "t"]]
    words = ["bed's", "bad", "bed", "bad"]
    assert missing_words(words, entries) == ["bad", "bad"]
    result = coverage({"test": words}, entries)
    assert result["types"] == 3
    assert result["type_oov_rate"] == pytest.approx(1 / 3)
    assert result["token_oov_rate"] == .5
    validate_phones(entries, {"phone_mapping": dict.fromkeys("b ɛ d t s".split())})
    with pytest.raises(ValueError, match="Incompatible dictionary"):
        validate_phones(entries, {"phone_mapping": {"b": 1}})


def test_textgrid_parsing_and_silence(tmp_path):
    path = tmp_path / "test.TextGrid"
    grid = textgrid.Textgrid()
    grid.addTier(textgrid.IntervalTier("speaker - words", [(.2, .5, "bed"), (.5, .6, "'s")], 0, 1))
    grid.addTier(textgrid.IntervalTier("speaker - phonemes", [(.2, .28, "b"), (.3, .6, "spn")], 0, 1))
    grid.save(str(path), format="long_textgrid", includeBlankSpaces=True)
    duration, tiers = read_textgrid(path)
    assert duration == 1
    assert word_intervals(tiers["words"]) == [(.2, .6, "bed's")]
    assert silence_durations(tiers["phones"], duration) == pytest.approx((.2, .02, .4))
    metrics, *_ = inspect_grid(path, 1, ["bed's"], {"b"})
    assert metrics["valid"]
    assert metrics["transcript_agreement"] == 1
    assert metrics["unknown_phones"] == metrics["words_with_unknown_phones"] == 1


def test_invalid_timestamps():
    errors = timestamp_issues([(-.01, .4, "a"), (.3, .3, "b"), (.9, 1.1, "c")], 1)
    assert errors == ["out of bounds at 0", "nonpositive interval at 1",
                      "overlap or reversed order at 1", "out of bounds at 2"]


def test_word_matching_with_repeated_labels():
    assert match_words(["a", "x", "a", "b"], ["a", "a", "b"]) == [(0, 0), (2, 1), (3, 2)]


def test_boundary_displacement():
    reference = [(.1, .3, "bed"), (.4, .5, "now")]
    comparison = [(.12, .29, "bed"), (.4, .5, "now")]
    rows = list(compare_boundaries(reference, comparison))
    assert rows[0]["start_abs_ms"] == 20
    assert rows[0]["end_abs_ms"] == 10
    assert rows[0]["duration_delta_ms"] == -30
    assert rows[1]["end_abs_ms"] == 0
