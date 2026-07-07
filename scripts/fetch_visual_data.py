#!/usr/bin/env python3
import sys
import urllib.request
from pathlib import Path

from silverspeak.homoglyphs.hkb.visual import CANDIDATE_DISCOVERIES_URL, CROSS_SCRIPT_URL

HKB_DATA = Path(__file__).resolve().parents[1] / "silverspeak" / "homoglyphs" / "hkb_data"
FILES = {
    "candidate-discoveries.json": CANDIDATE_DISCOVERIES_URL,
    "cross-script-discoveries.json": CROSS_SCRIPT_URL,
}


def fetch_visual_data(data_dir: Path) -> None:
    data_dir.mkdir(parents=True, exist_ok=True)
    for name, url in FILES.items():
        dest = data_dir / name
        print(f"fetching {url}")
        urllib.request.urlretrieve(url, dest)
        print(f"wrote {dest} ({dest.stat().st_size} bytes)")


def main() -> None:
    fetch_visual_data(data_dir=HKB_DATA)


if __name__ == "__main__":
    main()
    sys.exit(0)
