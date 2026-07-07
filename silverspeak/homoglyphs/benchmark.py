from dataclasses import dataclass
from typing import List, Protocol


class TextTransform(Protocol):
    def __call__(self, *, text: str) -> str: ...


@dataclass
class RoundTripResult:
    original: str
    attacked: str
    recovered: str
    exact_match: bool
    char_accuracy: float


@dataclass
class BenchmarkReport:
    clean_fpr: float
    clean_changed: int
    clean_total: int
    round_trips: List[RoundTripResult]


def _char_accuracy(a: str, b: str) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    matches = sum(1 for x, y in zip(a, b) if x == y)
    denom = max(len(a), len(b))
    return matches / denom


def measure_clean_fpr(
    samples: List[str],
    normalize_fn: TextTransform,
) -> tuple:
    changed = 0
    for sample in samples:
        if normalize_fn(text=sample) != sample:
            changed += 1
    total = len(samples)
    fpr = changed / total if total else 0.0
    return fpr, changed, total


def measure_round_trip(
    original: str,
    attack_fn: TextTransform,
    normalize_fn: TextTransform,
) -> RoundTripResult:
    attacked = attack_fn(text=original)
    recovered = normalize_fn(text=attacked)
    return RoundTripResult(
        original=original,
        attacked=attacked,
        recovered=recovered,
        exact_match=recovered == original,
        char_accuracy=_char_accuracy(a=recovered, b=original),
    )


def run_benchmark(
    clean_samples: List[str],
    round_trip_samples: List[str],
    attack_fn: TextTransform,
    normalize_fn: TextTransform,
) -> BenchmarkReport:
    fpr, changed, total = measure_clean_fpr(samples=clean_samples, normalize_fn=normalize_fn)
    trips = [measure_round_trip(original=s, attack_fn=attack_fn, normalize_fn=normalize_fn) for s in round_trip_samples]
    return BenchmarkReport(
        clean_fpr=fpr,
        clean_changed=changed,
        clean_total=total,
        round_trips=trips,
    )
