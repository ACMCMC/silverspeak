from pathlib import Path

import pytest

from silverspeak.homoglyphs.attacks.greedy_attack import greedy_attack
from silverspeak.homoglyphs.attacks.random_attack import random_attack
from silverspeak.homoglyphs.attacks.targeted_attack import targeted_attack
from silverspeak.homoglyphs.benchmark import measure_clean_fpr, measure_round_trip, run_benchmark
from silverspeak.homoglyphs.pipeline import normalize_fast

GRAPH = Path(__file__).resolve().parents[1] / "silverspeak" / "homoglyphs" / "hkb_data" / "graph.json.gz"
MIN_SCORE = 0.0
SCORE_MARGIN = 0.0
SEED = 2242
RECOVERY_THRESHOLD = 0.90

CLEAN_SAMPLES = [
    "hello world",
    "Привет, мир",
    "你好世界",
    "مرحبا بالعالم",
    "Γειά σου Κόσμε",
    "The quick brown fox jumps over the lazy dog.",
    "",
]

ROUND_TRIP_SAMPLES = [
    "Hello world",
    "This is a test sentence for homoglyph attacks.",
    "The quick brown fox jumps over the lazy dog.",
]

CRAFTED_ATTACKS = [
    ("hello world", "hеllо wоrld"),
]


def _norm(text: str) -> str:
    return normalize_fast(
        text=text,
        graph_path=GRAPH,
        min_score=MIN_SCORE,
        score_margin=SCORE_MARGIN,
    ).text


def _random_attack(text: str) -> str:
    return random_attack(text=text, percentage=0.15, random_seed=SEED)


def _greedy_attack(text: str) -> str:
    return greedy_attack(text=text, percentage=0.15, random_seed=SEED)


def _targeted_attack(text: str) -> str:
    return targeted_attack(text=text, percentage=0.15, random_seed=SEED)


def test_clean_text_fpr_is_zero() -> None:
    fpr, changed, total = measure_clean_fpr(samples=CLEAN_SAMPLES, normalize_fn=_norm)
    assert changed == 0
    assert fpr == 0.0
    assert total == len(CLEAN_SAMPLES)


@pytest.mark.parametrize("original,attacked", CRAFTED_ATTACKS)
def test_crafted_cross_script_round_trip_exact(original: str, attacked: str) -> None:
    result = measure_round_trip(
        original=original,
        attack_fn=lambda text: attacked,
        normalize_fn=_norm,
    )
    assert result.exact_match


@pytest.mark.parametrize("attack_fn", [_random_attack, _greedy_attack, _targeted_attack])
def test_round_trip_recovery_rate(attack_fn) -> None:
    accuracies = []
    for sample in ROUND_TRIP_SAMPLES:
        result = measure_round_trip(original=sample, attack_fn=attack_fn, normalize_fn=_norm)
        accuracies.append(result.char_accuracy)
    mean_accuracy = sum(accuracies) / len(accuracies)
    assert mean_accuracy >= RECOVERY_THRESHOLD


def test_benchmark_report_fast_random() -> None:
    report = run_benchmark(
        clean_samples=CLEAN_SAMPLES,
        round_trip_samples=ROUND_TRIP_SAMPLES,
        attack_fn=_random_attack,
        normalize_fn=_norm,
    )
    assert report.clean_fpr == 0.0
    assert len(report.round_trips) == len(ROUND_TRIP_SAMPLES)
