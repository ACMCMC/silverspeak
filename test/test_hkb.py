import gzip
import json
from pathlib import Path

import pytest

from silverspeak.homoglyphs.hkb.build import build_hkb
from silverspeak.homoglyphs.hkb.kb import HomoglyphKB

MAPS_DIR = Path(__file__).resolve().parents[1] / "silverspeak" / "homoglyphs"
SOURCES = ["identical", "confusables", "ocr_refined"]


@pytest.fixture
def kb_path(tmp_path: Path) -> Path:
    out = tmp_path / "graph.json.gz"
    build_hkb(
        maps_dir=MAPS_DIR,
        output_path=out,
        sources=SOURCES,
        bidirectional=True,
        visual_paths=None,
    )
    return out


@pytest.fixture
def kb(kb_path: Path) -> HomoglyphKB:
    return HomoglyphKB(graph_path=kb_path)


def test_build_is_deterministic(tmp_path: Path) -> None:
    out_a = tmp_path / "a.json.gz"
    out_b = tmp_path / "b.json.gz"
    build_hkb(maps_dir=MAPS_DIR, output_path=out_a, sources=SOURCES, bidirectional=True, visual_paths=None)
    build_hkb(maps_dir=MAPS_DIR, output_path=out_b, sources=SOURCES, bidirectional=True, visual_paths=None)
    with gzip.open(out_a, "rt", encoding="utf-8") as f:
        a = json.load(f)
    with gzip.open(out_b, "rt", encoding="utf-8") as f:
        b = json.load(f)
    assert a["neighbors"] == b["neighbors"]


def test_reverse_candidates_are_ranked_not_collapsed(kb: HomoglyphKB) -> None:
    candidates = kb.canonical_candidates(char="l", script="Latin", min_score=0.0)
    dsts = [c["dst"] for c in candidates]
    assert len(dsts) > 1
    scores = [c["score"] for c in candidates]
    assert scores == sorted(scores, reverse=True)


def test_latin_phishing_neighbor_exists(kb: HomoglyphKB) -> None:
    items = kb.homoglyphs_of(char="a", sources=["identical"], min_score=0.0)
    dsts = {item["dst"] for item in items}
    assert "а" in dsts


def test_coverage_report_counts_scripts(kb: HomoglyphKB) -> None:
    report = kb.coverage_report(text="hello Привет")
    assert "Latin" in report
    assert report["Latin"]["chars"] == 5


def test_full_build_includes_visual_edges() -> None:
    visual_paths = [
        MAPS_DIR / "hkb_data" / "candidate-discoveries.json",
        MAPS_DIR / "hkb_data" / "cross-script-discoveries.json",
    ]
    if not all(p.is_file() for p in visual_paths):
        return
    out = MAPS_DIR / "hkb_data" / "graph_test_visual.json.gz"
    artifact = build_hkb(
        maps_dir=MAPS_DIR,
        output_path=out,
        sources=SOURCES,
        bidirectional=True,
        visual_paths=visual_paths,
    )
    assert artifact["stats"]["visual_pair_count"] > 0
    kb = HomoglyphKB(graph_path=out)
    visual = kb.homoglyphs_of(char="a", sources=["visual"], min_score=0.7)
    assert len(visual) > 0
    out.unlink()
