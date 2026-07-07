import random
from typing import Dict, List, Optional, Set

from silverspeak.homoglyphs.homoglyph_replacer import HomoglyphReplacer
from silverspeak.homoglyphs.script_block_category_utils import (
    char_block,
    char_script,
    detect_dominant_block,
    detect_dominant_script,
)
from silverspeak.homoglyphs.utils import (
    _DEFAULT_HOMOGLYPHS_TO_USE,
    _DEFAULT_UNICODE_CATEGORIES_TO_REPLACE,
    TypesOfHomoglyphs,
)


def validate_attack_input(text: str, percentage: float) -> None:
    if text is None:
        raise ValueError("Input text cannot be None")
    if percentage < 0.0 or percentage > 1.0:
        raise ValueError("Percentage must be between 0.0 and 1.0")


def make_replacer(
    unicode_categories_to_replace: Set[str],
    types_of_homoglyphs_to_use: List[TypesOfHomoglyphs],
    replace_with_priority: bool,
    random_seed: int,
) -> HomoglyphReplacer:
    return HomoglyphReplacer(
        unicode_categories_to_replace=unicode_categories_to_replace,
        types_of_homoglyphs_to_use=types_of_homoglyphs_to_use,
        replace_with_priority=replace_with_priority,
        random_seed=random_seed,
    )


def filter_chars_map(
    chars_map: Dict[str, List[str]],
    text: str,
    same_script: bool,
    same_block: bool,
) -> Dict[str, List[str]]:
    filtered = chars_map
    if same_script:
        dominant_script = detect_dominant_script(text=text)
        filtered = {
            char: [h for h in homoglyphs if char_script(char=h) == dominant_script]
            for char, homoglyphs in filtered.items()
        }
    if same_block:
        dominant_block = detect_dominant_block(text=text)
        filtered = {
            char: [h for h in homoglyphs if char_block(char=h) == dominant_block]
            for char, homoglyphs in filtered.items()
        }
    return {char: homoglyphs for char, homoglyphs in filtered.items() if homoglyphs}


def attack_random_state(random_seed: Optional[int]) -> random.Random:
    return random.Random(x=42 if random_seed is None else random_seed)


def replacer_seed(random_seed: Optional[int]) -> int:
    return 42 if random_seed is None else random_seed
