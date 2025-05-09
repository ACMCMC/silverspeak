"""
Greedy homoglyph attack implementation.

This module implements a greedy character replacement attack using homoglyphs,
which replaces eligible characters in the text with visually similar homoglyphs.
Unlike random attack, this approach tries to replace as many characters as possible
within the percentage constraint.

Author: Aldan Creo (ACMC) <os@acmc.fyi>
Version: 1.0.0
License: See LICENSE file in the project root
"""

import logging
import random
from typing import List, Set

from silverspeak.homoglyphs.homoglyph_replacer import HomoglyphReplacer
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
    """
    Replace characters with homoglyphs using a greedy approach.

    This function replaces characters in the input text with homoglyphs (visually
    similar characters with different Unicode code points) using a greedy approach.
    The function will attempt to replace every eligible character up to the specified
    percentage limit.

    Args:
        text (str): The input text to transform.
        percentage (float, optional): The percentage of characters to replace (0.0-1.0).
            Defaults to 0.1 (10%).
        random_seed (int, optional): Seed for the random number generator to ensure
            reproducible results. Defaults to 42.
        unicode_categories_to_replace (Set[str], optional): Unicode categories of characters
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
        modified_text = greedy_attack("Hello world", percentage=0.05, random_seed=42)

        # Replace 10% of characters with homoglyphs, prioritizing replacements
        modified_text = greedy_attack("Hello world", percentage=0.1, replace_with_priority=True)
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
    random_state = random.Random(random_seed)

    # Create homoglyph replacer
    try:
        replacer = HomoglyphReplacer(
            unicode_categories_to_replace=unicode_categories_to_replace,
            random_seed=random_seed,
            types_of_homoglyphs_to_use=types_of_homoglyphs_to_use,
            replace_with_priority=replace_with_priority,
        )
    except Exception as e:
        logger.error(f"Failed to initialize HomoglyphReplacer: {e}")
        raise

    # Calculate number of characters to replace
    num_chars = len(text)
    num_to_replace = int(num_chars * percentage)
    if num_to_replace == 0 and percentage > 0 and num_chars > 0:
        num_to_replace = 1  # Replace at least one character if percentage is positive

    # Replace characters in the text
    result = []
    replacements_made = 0

    for char in text:
        if char in replacer.chars_map and replacements_made < num_to_replace:
            # Get replacements for this character
            replacements = replacer.chars_map[char]
            if replacements:
                # Choose a random replacement
                replacement = random_state.choice(replacements)
                result.append(replacement)
                replacements_made += 1
                logger.debug(f"Replaced '{char}' with '{replacement}'")
                continue

        # No replacement available or percentage limit reached
        result.append(char)

    logger.info(f"Applied {replacements_made} homoglyph replacements ({replacements_made/num_chars:.1%} of text)")
    return "".join(result)
