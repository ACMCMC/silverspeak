from pathlib import Path

from silverspeak.homoglyphs.fast_normalize import fast_normalize
from silverspeak.homoglyphs.hkb.kb import DEFAULT_HKB_PATH, HomoglyphKB
from silverspeak.homoglyphs.normalize_result import NormalizeResult


def normalize(
    text: str,
    pipeline: str,
    kb: HomoglyphKB,
    min_score: float,
    score_margin: float,
) -> NormalizeResult:
    if pipeline == "fast":
        return fast_normalize(
            text=text,
            kb=kb,
            min_score=min_score,
            score_margin=score_margin,
        )
    raise ValueError(f"Unknown pipeline: {pipeline}")


def normalize_fast(
    text: str,
    graph_path: Path,
    min_score: float,
    score_margin: float,
) -> NormalizeResult:
    kb = HomoglyphKB(graph_path=graph_path)
    return normalize(
        text=text,
        pipeline="fast",
        kb=kb,
        min_score=min_score,
        score_margin=score_margin,
    )


def load_default_kb() -> HomoglyphKB:
    return HomoglyphKB(graph_path=DEFAULT_HKB_PATH)
