from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class AmbiguousSpan:
    pos: int
    char: str
    candidates: List[Dict]


@dataclass
class CharChange:
    pos: int
    src: str
    dst: str
    score: float


@dataclass
class NormalizeResult:
    text: str
    ambiguous: List[AmbiguousSpan] = field(default_factory=list)
    chars_changed: List[CharChange] = field(default_factory=list)
