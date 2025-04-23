# %%
import logging
import random
from typing import List, Literal

from silverspeak.homoglyphs.homoglyph_replacer import HomoglyphReplacer
from silverspeak.homoglyphs.homoglyph_replacer import HomoglyphReplacer
from silverspeak.homoglyphs.utils import (
    _DEFAULT_HOMOGLYPHS_TO_USE,
    _DEFAULT_UNICODE_CATEGORIES_TO_REPLACE,
    TypesOfHomoglyphs,
)
from silverspeak.homoglyphs.script_block_category_utils import (
    detect_dominant_block,
    detect_dominant_script,
)

import unicodedataplus

logger = logging.getLogger(__name__)


def random_attack(
    text: str,
    percentage=0.1,
    random_seed=42,
    unicode_categories_to_replace=_DEFAULT_UNICODE_CATEGORIES_TO_REPLACE,
    types_of_homoglyphs_to_use: List[TypesOfHomoglyphs] = _DEFAULT_HOMOGLYPHS_TO_USE,
    replace_with_priority: bool = False,
    same_script=False,
    same_block=False,
) -> str:
    """
    Replaces some characters in the text, randomly choosing which ones, leaving all others unchanged.
    """
    random_state = random.Random(x=random_seed)
    replacer = HomoglyphReplacer(
        unicode_categories_to_replace=unicode_categories_to_replace,
        random_seed=random_seed,
        types_of_homoglyphs_to_use=types_of_homoglyphs_to_use,
        replace_with_priority=replace_with_priority,
    )
    # Replace some characters in the text with their equivalent characters from the chars_map
    num_to_replace = int(len(text) * percentage)
    text = list(text)  # Convert to list to allow for in-place replacement
    chars_map = replacer.chars_map
    if same_script:
        dominant_script = detect_dominant_script(text)
        chars_map = {
            char: [
                replacement
                for replacement in replacements
                if unicodedataplus.script(replacement) == dominant_script
            ]
            for char, replacements in chars_map.items()
        }
    if same_block:
        dominant_block = detect_dominant_block(text)
        chars_map = {
            char: [
                replacement
                for replacement in replacements
                if unicodedataplus.block(replacement) == dominant_block
            ]
            for char, replacements in chars_map.items()
        }

    # Remove empty lists from chars_map
    chars_map = {
        char: replacements for char, replacements in chars_map.items() if replacements
    }

    replaceable_chars = [(i, char) for i, char in enumerate(text) if char in chars_map]
    replaceable_count = len(replaceable_chars)
    logger.debug(
        f"Found {replaceable_count} replaceable characters in the text. Will replace {num_to_replace} characters."
    )

    if num_to_replace > replaceable_count:
        logger.warning(
            f"There are not enough replaceable characters in the text. Will replace all replaceable characters ({replaceable_count} instead of {num_to_replace})."
        )

    while num_to_replace > 0 and replaceable_count > 0:
        position, char = random_state.choice(replaceable_chars)
        replacement = random_state.choice(
            chars_map[char]
        )  # This is a string of potentially many characters, so we need to make space for it
        num_to_replace -= 1
        replaceable_count -= 1
        replaceable_chars.remove((position, char))

        # If the replacement is longer than the original character, we need to adjust the position

        if len(replacement) > 1:
            text_before = text[:position]
            # 1 character is replaced by 1 or more characters
            text_after = text[position + 1 :]
            text = text_before + replacement + text_after
            replaceable_chars = [
                (i + len(replacement) - 1, char) if i >= position else (i, char)
                for i, char in replaceable_chars
            ]
        else:
            # 1 character is replaced by 1 character
            text[position] = replacement

        logger.debug(
            f"Replaced character {char} with {replacement}. {num_to_replace} characters left to replace."
        )

    return "".join(text)


# %%
