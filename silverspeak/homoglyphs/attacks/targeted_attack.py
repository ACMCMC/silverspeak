import random
from typing import Dict, List, Optional, Set

from silverspeak.homoglyphs.attacks.shared import make_replacer, replacer_seed, validate_attack_input
from silverspeak.homoglyphs.unicode_scoring import TARGETED_PROPERTIES, score_homoglyphs_for_character
from silverspeak.homoglyphs.utils import (
    _DEFAULT_HOMOGLYPHS_TO_USE,
    _DEFAULT_UNICODE_CATEGORIES_TO_REPLACE,
    TypesOfHomoglyphs,
)


def targeted_attack(
    text: str,
    percentage: float = 0.1,
    random_seed: Optional[int] = None,
    unicode_categories_to_replace: Set[str] = _DEFAULT_UNICODE_CATEGORIES_TO_REPLACE,
    types_of_homoglyphs_to_use: List[TypesOfHomoglyphs] = _DEFAULT_HOMOGLYPHS_TO_USE,
    replace_with_priority: bool = False,
) -> str:
    validate_attack_input(text=text, percentage=percentage)
    if not text:
        return ""

    random_state = random.Random(x=replacer_seed(random_seed=random_seed))
    replacer = make_replacer(
        unicode_categories_to_replace=unicode_categories_to_replace,
        types_of_homoglyphs_to_use=types_of_homoglyphs_to_use,
        replace_with_priority=replace_with_priority,
        random_seed=replacer_seed(random_seed=random_seed),
    )

    chars_map = {char: homoglyphs for char, homoglyphs in replacer.chars_map.items() if homoglyphs}
    if not chars_map:
        return text

    num_to_replace = int(len(text) * percentage)
    if num_to_replace == 0 and percentage > 0:
        num_to_replace = 1

    replacement_options = []
    for i, char in enumerate(text):
        if char not in chars_map:
            continue
        possible_replacements = []
        for homoglyph in chars_map[char]:
            if homoglyph == char:
                continue
            score = score_homoglyphs_for_character(
                homoglyph=homoglyph,
                char=char,
                PROPERTIES=TARGETED_PROPERTIES,
            )
            possible_replacements.append((homoglyph, score))
        possible_replacements.sort(key=lambda x: x[1], reverse=True)
        if possible_replacements:
            replacement_options.append((i, char, possible_replacements[0][0], possible_replacements[0][1]))

    replacement_options.sort(key=lambda x: x[3], reverse=True)
    if len(replacement_options) < num_to_replace:
        num_to_replace = len(replacement_options)
    if num_to_replace == 0:
        return text

    score_groups: Dict[float, list] = {}
    for option in replacement_options:
        score_groups.setdefault(option[3], []).append(option)

    replacements_to_apply = []
    remaining_to_select = num_to_replace
    for score in sorted(score_groups.keys(), reverse=True):
        if remaining_to_select <= 0:
            break
        candidates = score_groups[score]
        random_state.shuffle(candidates)
        to_take = min(remaining_to_select, len(candidates))
        replacements_to_apply.extend(candidates[:to_take])
        remaining_to_select -= to_take

    chars = list(text)
    for idx, orig_char, replacement, _score in sorted(replacements_to_apply, key=lambda x: x[0], reverse=True):
        if replacement and replacement != orig_char:
            if len(replacement) > 1:
                chars.pop(idx)
                for j, rep_char in enumerate(replacement):
                    chars.insert(idx + j, rep_char)
            else:
                chars[idx] = replacement

    return "".join(chars)
