import unicodedata
from pathlib import Path
from typing import Dict, List, Optional

from silverspeak.homoglyphs.hkb.kb import DEFAULT_HKB_PATH, HomoglyphKB, load_default_kb
from silverspeak.homoglyphs.normalize_result import AmbiguousSpan, CharChange, NormalizeResult
from silverspeak.homoglyphs.script_block_category_utils import char_script, detect_dominant_script
from silverspeak.homoglyphs.strip import strip_format_chars


def _dedupe_by_dst(candidates: List[Dict]) -> List[Dict]:
    best: Dict[str, Dict] = {}
    for item in candidates:
        dst = item["dst"]
        if dst not in best or item["score"] > best[dst]["score"]:
            best[dst] = item
    ranked = list(best.values())
    ranked.sort(key=lambda e: (-e["score"], e["dst"], e["source"]))
    return ranked


def _pick_canonical(
    char: str,
    candidates: List[Dict],
    score_margin: float,
) -> Optional[Dict]:
    ranked = _dedupe_by_dst(candidates=candidates)
    if not ranked:
        return None
    if ranked[0]["dst"] == char:
        return None
    if len(ranked) == 1:
        return ranked[0]
    if ranked[0]["score"] - ranked[1]["score"] > score_margin:
        return ranked[0]
    return None


def fast_normalize(
    text: str,
    kb: HomoglyphKB,
    min_score: float,
    score_margin: float,
) -> NormalizeResult:
    if not text:
        return NormalizeResult(text="", ambiguous=[], chars_changed=[])
    stripped = strip_format_chars(text=text)
    nfkc = unicodedata.normalize("NFKC", stripped)
    if not nfkc:
        return NormalizeResult(text="", ambiguous=[], chars_changed=[])

    dominant_script = detect_dominant_script(text=nfkc)
    out_chars: List[str] = []
    ambiguous: List[AmbiguousSpan] = []
    chars_changed: List[CharChange] = []

    for pos, char in enumerate(nfkc):
        if char_script(char=char) == dominant_script:
            out_chars.append(char)
            continue

        candidates = kb.canonical_candidates(
            char=char,
            script=dominant_script,
            min_score=min_score,
        )
        picked = _pick_canonical(char=char, candidates=candidates, score_margin=score_margin)

        if picked is not None:
            out_chars.append(picked["dst"])
            chars_changed.append(CharChange(pos=pos, src=char, dst=picked["dst"], score=picked["score"]))
            continue

        ranked = _dedupe_by_dst(candidates=candidates)
        if len(ranked) > 1:
            ambiguous.append(AmbiguousSpan(pos=pos, char=char, candidates=ranked))
        out_chars.append(char)

    return NormalizeResult(
        text="".join(out_chars),
        ambiguous=ambiguous,
        chars_changed=chars_changed,
    )


def normalize_fast(
    text: str,
    graph_path: Path,
    min_score: float,
    score_margin: float,
) -> NormalizeResult:
    kb = HomoglyphKB(graph_path=graph_path)
    return fast_normalize(
        text=text,
        kb=kb,
        min_score=min_score,
        score_margin=score_margin,
    )
