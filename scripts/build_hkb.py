#!/usr/bin/env python3
import sys
from pathlib import Path

from silverspeak.homoglyphs.hkb.build import build_hkb

MAPS_DIR = Path(__file__).resolve().parents[1] / "silverspeak" / "homoglyphs"
HKB_DATA = MAPS_DIR / "hkb_data"
OUTPUT = HKB_DATA / "graph.json.gz"
SOURCES = ["identical", "confusables", "ocr_refined"]
VISUAL_FILES = [
    HKB_DATA / "candidate-discoveries.json",
    HKB_DATA / "cross-script-discoveries.json",
]


def main() -> None:
    visual_paths = [p for p in VISUAL_FILES if p.is_file()]
    if not visual_paths:
        print("no visual data found; run scripts/fetch_visual_data.py first")
    artifact = build_hkb(
        maps_dir=MAPS_DIR,
        output_path=OUTPUT,
        sources=SOURCES,
        bidirectional=True,
        visual_paths=visual_paths if visual_paths else None,
    )
    stats = artifact["stats"]
    print(f"wrote {OUTPUT}")
    print(
        f"edges={stats['edge_count']} chars={stats['char_count']} "
        f"ambiguous={stats['ambiguous_src_count']} visual_pairs={stats['visual_pair_count']}"
    )


if __name__ == "__main__":
    main()
    sys.exit(0)
