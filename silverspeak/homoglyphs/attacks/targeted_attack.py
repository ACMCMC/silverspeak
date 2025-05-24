"""
Targeted homoglyph attack implementation.

This module implements a targeted character replacement attack using homoglyphs,
replacing specific characters or patterns in the text to maximize the impact
while maintaining visual similarity to the original text. It uses context-aware
scoring to select optimal homoglyphs based on surrounding character properties.

Author: Aldan Creo (ACMC) <os@acmc.fyi>
License: See LICENSE file in the project root
"""

import logging
import random
import re
import unicodedata
from typing import Dict, List, Mapping, Optional, Pattern, Set, Tuple, Union

import unicodedataplus

from silverspeak.homoglyphs.homoglyph_replacer import HomoglyphReplacer
from silverspeak.homoglyphs.script_block_category_utils import detect_dominant_block, detect_dominant_script
from silverspeak.homoglyphs.unicode_scoring import score_homoglyphs_for_character
from silverspeak.homoglyphs.utils import (
    _DEFAULT_HOMOGLYPHS_TO_USE,
    _DEFAULT_UNICODE_CATEGORIES_TO_REPLACE,
    TypesOfHomoglyphs,
)

logger = logging.getLogger(__name__)


def targeted_attack(
    text: str,
    percentage: float = 0.1,
    random_seed: Optional[int] = None,
    unicode_categories_to_replace: Set[str] = _DEFAULT_UNICODE_CATEGORIES_TO_REPLACE,
    types_of_homoglyphs_to_use: List[TypesOfHomoglyphs] = _DEFAULT_HOMOGLYPHS_TO_USE,
    replace_with_priority: bool = False,
) -> str:
    """
    Replace a percentage of characters in text with property-matched homoglyphs.

    This function replaces a specified percentage of characters in the input text
    with homoglyphs (visually similar characters with different Unicode code points).
    The function selects homoglyphs based on matching the Unicode properties of the
    original character being replaced, without considering surrounding context.
    The implementation selects the highest scoring homoglyph replacements based on
    property matching.

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

    Returns:
        str: The transformed text with property-matched homoglyph replacements.

    Raises:
        ValueError: If the text is None or empty, or if percentage is out of range.

    Example:
        ```python
        # Replace 5% of characters with property-matched homoglyphs
        modified_text = targeted_attack(
            "Hello world",
            percentage=0.05,
            random_seed=42
        )

        # Replace 10% of characters with property-matched homoglyphs
        modified_text = targeted_attack(
            "Hello world",
            percentage=0.1
        )
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

    # Remove entries with empty homoglyph lists
    chars_map = {char: homoglyphs for char, homoglyphs in chars_map.items() if homoglyphs}

    if not chars_map:
        logger.warning("No valid replacements found")
        return text

    # Calculate number of characters to replace
    num_to_replace = int(len(text) * percentage)

    # Ensure we replace at least one character if percentage > 0
    if num_to_replace == 0 and percentage > 0 and len(text) > 0:
        num_to_replace = 1

    # Find all possible replacements and score them
    replacement_options = []

    for i, char in enumerate(text):
        if char in chars_map:
            # Get all possible homoglyphs with scores
            possible_replacements = []
            for homoglyph in chars_map[char]:
                if homoglyph == char:
                    continue  # Skip if it's the same character

                # Score the homoglyph based only on properties of the original character
                score = score_homoglyphs_for_character(
                    homoglyph=homoglyph,
                    char=char,
                    PROPERTIES=[
                        {
                            "script": {"fn": unicodedataplus.script, "weight": 3},
                            "block": {"fn": unicodedataplus.block, "weight": 5},
                            "category": {"fn": unicodedata.category, "weight": 10},
                            "bidirectional": {"fn": unicodedata.bidirectional, "weight": 2},
                            "east_asian_width": {"fn": unicodedata.east_asian_width, "weight": 1},
                        }
                    ],
                )

                possible_replacements.append((homoglyph, score))

            # Sort by score (highest first) and add to options
            possible_replacements.sort(key=lambda x: x[1], reverse=True)
            if possible_replacements:
                # Store index, original char, best replacement, and score
                replacement_options.append((i, char, possible_replacements[0][0], possible_replacements[0][1]))

    # Sort all replacement options by score
    replacement_options.sort(key=lambda x: x[3], reverse=True)

    # If we have fewer options than requested replacements, adjust
    if len(replacement_options) < num_to_replace:
        logger.warning(
            f"Not enough replaceable characters in the text. Will replace {len(replacement_options)} characters instead of {num_to_replace}."
        )
        num_to_replace = len(replacement_options)

    if num_to_replace == 0:
        logger.info("No characters will be replaced (percentage too low or no eligible characters)")
        return text

    # Select top scoring replacements to apply
    replacements_to_apply = replacement_options[:num_to_replace]

    # Apply replacements (starting from the end to avoid index shifting problems)
    replacements_to_apply.sort(key=lambda x: x[0], reverse=True)

    # Convert text to list for character replacement
    chars = list(text)

    # Track how many replacements were actually applied
    applied_replacements = 0

    for idx, orig_char, replacement, score in replacements_to_apply:
        if replacement and replacement != orig_char:
            # For multi-character replacements, insert and adjust indices
            if len(replacement) > 1:
                # Remove the original character and insert the replacement
                chars.pop(idx)
                for j, rep_char in enumerate(replacement):
                    chars.insert(idx + j, rep_char)
            else:
                # Simple single-character replacement
                chars[idx] = replacement

            applied_replacements += 1
            logger.debug(f"Replaced '{orig_char}' at position {idx} with '{replacement}' (score: {score:.2f})")

    result = "".join(chars)
    logger.info(f"Applied {applied_replacements} property-matched homoglyph replacements")

    return result
