import gzip
import json
from pathlib import Path
from typing import Dict, List, Optional

from silverspeak.homoglyphs.script_block_category_utils import char_script

DEFAULT_HKB_PATH = Path(__file__).parent.parent / "hkb_data" / "graph.json.gz"


class HomoglyphKB:
    def __init__(self, graph_path: Path):
        self.graph_path = graph_path
        with gzip.open(graph_path, "rt", encoding="utf-8") as f:
            data = json.load(f)
        self.version = data["version"]
        self.stats = data["stats"]
        self.neighbors: Dict[str, List[Dict]] = data["neighbors"]

    def homoglyphs_of(
        self,
        char: str,
        sources: Optional[List[str]],
        min_score: float,
    ) -> List[Dict]:
        items = self.neighbors.get(char, [])
        out = []
        for item in items:
            if item["score"] < min_score:
                continue
            if sources is not None and item["source"] not in sources:
                continue
            out.append(item)
        return out

    def canonical_candidates(
        self,
        char: str,
        script: str,
        min_score: float,
    ) -> List[Dict]:
        items = self.neighbors.get(char, [])
        out = []
        for item in items:
            if item["score"] < min_score:
                continue
            if char_script(char=item["dst"]) != script:
                continue
            out.append(item)
        out.sort(key=lambda e: (-e["score"], e["dst"], e["source"]))
        return out

    def is_ambiguous(self, char: str) -> bool:
        items = self.neighbors.get(char, [])
        dsts = {item["dst"] for item in items}
        return len(dsts) > 1

    def coverage_report(self, text: str) -> Dict:
        scripts: Dict[str, Dict] = {}
        for char in text:
            if len(char) != 1:
                continue
            script = char_script(char=char)
            if script not in scripts:
                scripts[script] = {
                    "chars": 0,
                    "with_homoglyphs": 0,
                    "ambiguous": 0,
                    "sources": {},
                }
            scripts[script]["chars"] += 1
            items = self.neighbors.get(char, [])
            if items:
                scripts[script]["with_homoglyphs"] += 1
            if self.is_ambiguous(char=char):
                scripts[script]["ambiguous"] += 1
            for item in items:
                src = item["source"]
                scripts[script]["sources"][src] = scripts[script]["sources"].get(src, 0) + 1
        return scripts


def load_default_kb() -> HomoglyphKB:
    return HomoglyphKB(graph_path=DEFAULT_HKB_PATH)
