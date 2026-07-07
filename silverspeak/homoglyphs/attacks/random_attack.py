import logging
import random
from typing import Dict, List, Optional, Set

from silverspeak.homoglyphs.attacks.shared import (
    attack_random_state,
    filter_chars_map,
    make_replacer,
    replacer_seed,
    validate_attack_input,
)
from silverspeak.homoglyphs.utils import (
    _DEFAULT_HOMOGLYPHS_TO_USE,
    _DEFAULT_UNICODE_CATEGORIES_TO_REPLACE,
    TypesOfHomoglyphs,
)

logger = logging.getLogger(__name__)


def random_attack(
    text: str,
    percentage: float = 0.1,
    random_seed: Optional[int] = None,
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
        random_seed=replacer_seed(random_seed=random_seed),
    )

    chars_map = filter_chars_map(
        chars_map=replacer.chars_map,
        text=text,
        same_script=same_script,
        same_block=same_block,
    )
    if not chars_map:
        logger.warning("No valid replacements found with current filters (script/block)")
        return text

    num_to_replace = int(len(text) * percentage)
    text_chars = list(text)
    replaceable_chars = [(i, char) for i, char in enumerate(text_chars) if char in chars_map]

    if num_to_replace > len(replaceable_chars):
        num_to_replace = len(replaceable_chars)
    if num_to_replace == 0:
        return text

    replacements_applied = 0
    original_positions: Dict[int, int] = {}

    while replacements_applied < num_to_replace and replaceable_chars:
        idx = random_state.randint(0, len(replaceable_chars) - 1)
        position, char = replaceable_chars.pop(idx)
        adjusted_position = position
        for orig_pos, shift in sorted(original_positions.items()):
            if position > orig_pos:
                adjusted_position += shift

        replacement = random_state.choice(chars_map[char])
        position_shift = len(replacement) - 1

        if position_shift > 0:
            original_positions[position] = position_shift
            text_chars = text_chars[:adjusted_position] + list(replacement) + text_chars[adjusted_position + 1 :]
        else:
            text_chars[adjusted_position] = replacement

        replacements_applied += 1

    return "".join(text_chars)
