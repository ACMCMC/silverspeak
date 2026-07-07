import gzip
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from silverspeak.homoglyphs.hkb.visual import (
    CANDIDATE_DISCOVERIES_URL,
    CROSS_SCRIPT_URL,
    VISUAL_SOURCE,
    load_visual_pairs,
    merge_visual_edges,
)

SOURCE_SCORES = {
    "identical": 1.0,
    "confusables": 0.8,
    "ocr_refined": 0.7,
    "ocr": 0.6,
    "visual": 0.75,
}

MAP_FILES = {
    "identical": "identical_map.json",
    "confusables": "unicode_confusables_map.json",
    "ocr": "ocr_chars_map.json",
    "ocr_refined": "ocr_chars_refined_map.json",
}


def _load_map(path: Path) -> Dict[str, List[str]]:
    with open(path, encoding="utf-8") as f:
        data: Dict[str, List[str]] = json.load(f)
        return data


def _add_edge(
    edges: Dict[Tuple[str, str, str], float],
    src: str,
    dst: str,
    source: str,
    score: Optional[float],
) -> None:
    if len(src) != 1 or len(dst) != 1 or src == dst:
        return
    key = (src, dst, source)
    edge_score = SOURCE_SCORES[source] if score is None else score
    if key not in edges or edge_score > edges[key]:
        edges[key] = edge_score


def _merge_maps(
    maps_dir: Path,
    sources: List[str],
    bidirectional: bool,
) -> Dict[Tuple[str, str, str], float]:
    edges: Dict[Tuple[str, str, str], float] = {}
    for source in sources:
        path = maps_dir / MAP_FILES[source]
        data = _load_map(path=path)
        for src, dsts in data.items():
            for dst in dsts:
                _add_edge(edges=edges, src=src, dst=dst, source=source, score=None)
                if bidirectional:
                    _add_edge(edges=edges, src=dst, dst=src, source=source, score=None)
    return edges


def _rank_neighbors(
    edges: Dict[Tuple[str, str, str], float],
) -> Dict[str, List[Dict]]:
    by_src: Dict[str, List[Dict]] = {}
    for (src, dst, source), score in edges.items():
        by_src.setdefault(src, []).append({"dst": dst, "score": score, "source": source})
    for src in by_src:
        by_src[src].sort(key=lambda e: (-e["score"], e["dst"], e["source"]))
    return by_src


def build_hkb(
    maps_dir: Path,
    output_path: Path,
    sources: List[str],
    bidirectional: bool,
    visual_paths: Optional[List[Path]],
) -> Dict:
    edge_scores = _merge_maps(maps_dir=maps_dir, sources=sources, bidirectional=bidirectional)
    visual_pair_count = 0
    if visual_paths:
        all_visual: List[Tuple[str, str, float]] = []
        for vpath in visual_paths:
            if vpath.is_file():
                all_visual.extend(load_visual_pairs(path=vpath))
        visual_pair_count = len(all_visual)
        merge_visual_edges(edges=edge_scores, visual_pairs=all_visual, bidirectional=bidirectional)
    neighbors = _rank_neighbors(edges=edge_scores)
    ambiguous = sum(1 for items in neighbors.values() if len({e["dst"] for e in items}) > 1)
    artifact = {
        "version": 2,
        "maps_dir": str(maps_dir),
        "sources": sources,
        "bidirectional": bidirectional,
        "visual_paths": [str(p) for p in visual_paths] if visual_paths else [],
        "stats": {
            "edge_count": len(edge_scores),
            "char_count": len(neighbors),
            "ambiguous_src_count": ambiguous,
            "visual_pair_count": visual_pair_count,
        },
        "neighbors": neighbors,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    raw = json.dumps(artifact, ensure_ascii=False, separators=(",", ":"))
    with gzip.open(output_path, "wt", encoding="utf-8") as f:
        f.write(raw)
    return artifact
