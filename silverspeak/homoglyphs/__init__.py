from .attacks import greedy_attack, random_attack, targeted_attack
from .fast_normalize import normalize_fast
from .hkb.kb import load_default_kb
from .homoglyph_replacer import HomoglyphReplacer
from .utils import TypesOfHomoglyphs

__all__ = [
    "normalize_fast",
    "load_default_kb",
    "HomoglyphReplacer",
    "TypesOfHomoglyphs",
    "greedy_attack",
    "random_attack",
    "targeted_attack",
]
