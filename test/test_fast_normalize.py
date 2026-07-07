from pathlib import Path

import pytest

from silverspeak.homoglyphs.pipeline import normalize_fast

GRAPH = Path(__file__).resolve().parents[1] / "silverspeak" / "homoglyphs" / "hkb_data" / "graph.json.gz"
MIN_SCORE = 0.0
SCORE_MARGIN = 0.0


def _fast(text: str):
    return normalize_fast(
        text=text,
        graph_path=GRAPH,
        min_score=MIN_SCORE,
        score_margin=SCORE_MARGIN,
    )


def test_idempotency() -> None:
    attacked = "hеllо wоrld"
    once = _fast(text=attacked)
    twice = _fast(text=once.text)
    assert twice.text == once.text
    assert twice.chars_changed == []


def test_clean_cyrillic_unchanged() -> None:
    text = "Привет, мир"
    result = _fast(text=text)
    assert result.text == text
    assert result.chars_changed == []


def test_clean_chinese_unchanged() -> None:
    text = "你好世界"
    result = _fast(text=text)
    assert result.text == text


def test_clean_arabic_unchanged() -> None:
    text = "مرحبا بالعالم"
    result = _fast(text=text)
    assert result.text == text


def test_mixed_script_latin_attack_recovered() -> None:
    attacked = "hеllо wоrld"
    result = _fast(text=attacked)
    assert result.text == "hello world"
    assert len(result.chars_changed) >= 2


def test_ambiguous_not_injected() -> None:
    result = _fast(text="hеllо wоrld")
    assert "\ufffd" not in result.text


def test_strip_zero_width() -> None:
    text = "hello\u200bworld"
    result = _fast(text=text)
    assert result.text == "helloworld"
