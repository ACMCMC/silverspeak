import logging
from typing import List, Set

from silverspeak.homoglyphs.attacks.shared import (
    attack_random_state,
    filter_chars_map,
    make_replacer,
    validate_attack_input,
)
from silverspeak.homoglyphs.utils import (
    _DEFAULT_HOMOGLYPHS_TO_USE,
    _DEFAULT_UNICODE_CATEGORIES_TO_REPLACE,
    TypesOfHomoglyphs,
)

logger = logging.getLogger(__name__)


def greedy_attack(
    text: str,
    percentage: float = 0.1,
    random_seed: int = 42,
    unicode_categories_to_replace: Set[str] = _DEFAULT_UNICODE_CATEGORIES_TO_REPLACE,
    types_of_homoglyphs_to_use: List[TypesOfHomoglyphs] = _DEFAULT_HOMOGLYPHS_TO_USE,
    replace_with_priority: bool = False,
    same_script: bool = False,
    same_block: bool = False,
) -> str:
    validate_attack_input(text=text, percentage=percentage)
    if not text:
        return ""

    random_state = attack_random_state(random_seed=random_seed)
    replacer = make_replacer(
        unicode_categories_to_replace=unicode_categories_to_replace,
        types_of_homoglyphs_to_use=types_of_homoglyphs_to_use,
        replace_with_priority=replace_with_priority,
        random_seed=random_seed,
    )
    chars_map = filter_chars_map(
        chars_map=replacer.chars_map,
        text=text,
        same_script=same_script,
        same_block=same_block,
    )

    num_to_replace = int(len(text) * percentage)
    if num_to_replace == 0 and percentage > 0:
        num_to_replace = 1

    result = []
    replacements_made = 0
    for char in text:
        if char in chars_map and replacements_made < num_to_replace:
            replacement = random_state.choice(chars_map[char])
            result.append(replacement)
            replacements_made += 1
            continue
        result.append(char)

    return "".join(result)
