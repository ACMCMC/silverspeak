import json
from pathlib import Path
from typing import Dict, List, Tuple

VISUAL_SOURCE = "visual"
CANDIDATE_DISCOVERIES_URL = (
    "https://raw.githubusercontent.com/paultendo/confusable-vision/main/data/output/candidate-discoveries.json"
)
CROSS_SCRIPT_URL = (
    "https://raw.githubusercontent.com/paultendo/confusable-vision/main/data/output/cross-script-discoveries.json"
)


def _pair_score(pair: Dict) -> float:
    summary = pair.get("summary", {})
    return float(summary.get("meanSsim", summary.get("meanDistance", 0.0)))


def load_visual_pairs(path: Path) -> List[Tuple[str, str, float]]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    pairs = data.get("pairs", data) if isinstance(data, dict) else data
    out: List[Tuple[str, str, float]] = []
    for pair in pairs:
        if "source" in pair:
            src = pair["source"]
            dst = pair["target"]
        elif "charA" in pair:
            src = pair["charA"]
            dst = pair["charB"]
        else:
            continue
        if len(src) != 1 or len(dst) != 1:
            continue
        score = _pair_score(pair=pair)
        if score <= 0:
            continue
        out.append((src, dst, score))
    return out


def merge_visual_edges(
    edges: Dict[Tuple[str, str, str], float],
    visual_pairs: List[Tuple[str, str, float]],
    bidirectional: bool,
) -> None:
    for src, dst, score in visual_pairs:
        key = (src, dst, VISUAL_SOURCE)
        if key not in edges or score > edges[key]:
            edges[key] = score
        if bidirectional:
            key_rev = (dst, src, VISUAL_SOURCE)
            if key_rev not in edges or score > edges[key_rev]:
                edges[key_rev] = score
