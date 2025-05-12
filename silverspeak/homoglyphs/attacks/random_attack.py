"""
Random homoglyph attack implementation.

This module implements a random character replacement attack using homoglyphs,
which can be used to create adversarial text samples that appear visually similar
to the original but use different Unicode characters.

Author: Aldan Creo (ACMC) <os@acmc.fyi>
License: See LICENSE file in the project root
"""

import logging
import random
from typing import Dict, List, Optional, Set, Union

import unicodedataplus

from silverspeak.homoglyphs.homoglyph_replacer import HomoglyphReplacer
from silverspeak.homoglyphs.script_block_category_utils import detect_dominant_block, detect_dominant_script
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
    """
    Replace random characters in text with visually similar homoglyphs.

    This function replaces a specified percentage of characters in the input text
    with homoglyphs (visually similar characters with different Unicode code points).
    The replacements are chosen randomly, and the function can be configured to maintain
    script and block consistency.

    Args:
        text (str): The input text to transform.
        percentage (float, optional): The percentage of characters to replace (0.0-1.0).
            Defaults to 0.1 (10%).
        random_seed (Optional[int], optional): Seed for the random number generator
            to ensure reproducible results. Defaults to None (non-reproducible).
        unicode_categories_to_replace (Set[str]): Unicode categories
            to consider for replacement. Defaults to predefined categories.
        types_of_homoglyphs_to_use (List[TypesOfHomoglyphs], optional): Types of homoglyphs
            to use for replacements. Defaults to predefined types.
        replace_with_priority (bool, optional): Whether to replace characters based on priority.
            Defaults to False.
        same_script (bool, optional): Whether to only use homoglyphs from the same Unicode script
            as the dominant script in the text. Defaults to False.
        same_block (bool, optional): Whether to only use homoglyphs from the same Unicode block
            as the dominant block in the text. Defaults to False.

    Returns:
        str: The transformed text with homoglyph replacements.

    Raises:
        ValueError: If the text is None or empty, or if percentage is out of range.

    Example:
        ```python
        # Replace 5% of characters with homoglyphs
        modified_text = random_attack("Hello world", percentage=0.05, random_seed=42)

        # Replace 10% of characters with homoglyphs from the same script
        modified_text = random_attack("Hello world", percentage=0.1, same_script=True)
        ```
    """
    # Input validation
    if text is None:
        raise ValueError("Input text cannot be None")

    if not text:
        return ""

    if percentage < 0.0 or percentage > 1.0:
        raise ValueError("Percentage must be between 0.0 and 1.0")

    # Initialize random number generator
    random_state = random.Random(x=random_seed)

    # Create homoglyph replacer
    try:
        replacer = HomoglyphReplacer(
            unicode_categories_to_replace=unicode_categories_to_replace,
            types_of_homoglyphs_to_use=types_of_homoglyphs_to_use,
            replace_with_priority=replace_with_priority,
            random_seed=42 if random_seed is None else random_seed,
        )
    except Exception as e:
        logger.error(f"Failed to initialize HomoglyphReplacer: {e}")
        raise

    # Get character mapping
    chars_map = replacer.chars_map

    # Filter homoglyphs by script if requested
    if same_script:
        dominant_script = detect_dominant_script(text)
        logger.debug(f"Dominant script detected: {dominant_script}")
        chars_map = {
            char: [
                replacement for replacement in replacements if unicodedataplus.script(replacement) == dominant_script
            ]
            for char, replacements in chars_map.items()
        }

    # Filter homoglyphs by block if requested
    if same_block:
        dominant_block = detect_dominant_block(text)
        logger.debug(f"Dominant block detected: {dominant_block}")
        chars_map = {
            char: [replacement for replacement in replacements if unicodedataplus.block(replacement) == dominant_block]
            for char, replacements in chars_map.items()
        }

    # Remove entries with empty replacement lists
    chars_map = {char: replacements for char, replacements in chars_map.items() if replacements}

    if not chars_map:
        logger.warning("No valid replacements found with current filters (script/block)")
        return text

    # Calculate number of characters to replace
    num_to_replace = int(len(text) * percentage)

    # Convert text to list for in-place character replacement
    text_chars = list(text)

    # Find replaceable characters
    replaceable_chars = [(i, char) for i, char in enumerate(text_chars) if char in chars_map]
    replaceable_count = len(replaceable_chars)

    logger.debug(
        f"Found {replaceable_count} replaceable characters in the text. Will replace {num_to_replace} characters."
    )

    # Check if we have enough replaceable characters
    if num_to_replace > replaceable_count:
        logger.warning(
            f"Not enough replaceable characters in the text. Will replace {replaceable_count} characters instead of {num_to_replace}."
        )
        num_to_replace = replaceable_count

    if num_to_replace == 0:
        logger.info("No characters will be replaced (percentage too low or no replaceable characters)")
        return text

    # Apply replacements
    replacements_applied = 0
    original_positions: Dict[int, int] = {}  # Track original positions to handle multi-character replacements

    while replacements_applied < num_to_replace and replaceable_chars:
        # Choose a random replaceable character
        idx = random_state.randint(0, len(replaceable_chars) - 1)
        position, char = replaceable_chars.pop(idx)

        # Adjust position if we've already made replacements
        adjusted_position = position
        for orig_pos, shift in sorted(original_positions.items()):
            if position > orig_pos:
                adjusted_position += shift

        # Choose a random replacement
        replacement = random_state.choice(chars_map[char])

        # Calculate position shift (if replacement is longer than original)
        position_shift = len(replacement) - 1  # -1 because we're replacing 1 character

        if position_shift > 0:
            # For multi-character replacements, track the position shift
            original_positions[position] = position_shift

            # Replace by splitting and rejoining the text
            text_before = text_chars[:adjusted_position]
            text_after = text_chars[adjusted_position + 1 :]
            text_chars = text_before + list(replacement) + text_after
        else:
            # Simple single-character replacement
            text_chars[adjusted_position] = replacement

        replacements_applied += 1

        logger.debug(
            f"Replaced character '{char}' at position {position} with '{replacement}'. "
            f"{num_to_replace - replacements_applied} replacements remaining."
        )

    result = "".join(text_chars)
    logger.info(f"Applied {replacements_applied} homoglyph replacements")

    return result
