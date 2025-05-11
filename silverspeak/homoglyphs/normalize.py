"""
Text normalization functionality for homoglyph detection and replacement.

This module provides a simple interface to normalize text by replacing homoglyphs
with their standard equivalents using various normalization strategies.

Author: Aldan Creo (ACMC) <os@acmc.fyi>
License: See LICENSE file in the project root
"""

import logging
from typing import List, Optional, Set, Union

from silverspeak.homoglyphs.homoglyph_replacer import HomoglyphReplacer
from silverspeak.homoglyphs.utils import (
    _DEFAULT_HOMOGLYPHS_TO_USE,
    _DEFAULT_UNICODE_CATEGORIES_TO_REPLACE,
    NormalizationStrategies,
    TypesOfHomoglyphs,
)

logger = logging.getLogger(__name__)


def normalize_text(
    text: str,
    unicode_categories_to_replace: Set[str] = _DEFAULT_UNICODE_CATEGORIES_TO_REPLACE,
    types_of_homoglyphs_to_use: List[TypesOfHomoglyphs] = _DEFAULT_HOMOGLYPHS_TO_USE,
    replace_with_priority: bool = False,
    strategy: NormalizationStrategies = NormalizationStrategies.LOCAL_CONTEXT,
) -> str:
    """
    Normalize text by replacing homoglyphs with their standard equivalents.

    This function provides a convenient interface to the HomoglyphReplacer's normalize method,
    creating a temporary HomoglyphReplacer instance with the specified parameters.

    Args:
        text (str): The text to normalize.
        unicode_categories_to_replace (Set[str]): Unicode categories to replace.
            Defaults to predefined common categories.
        types_of_homoglyphs_to_use (List[TypesOfHomoglyphs]): Types of homoglyphs to consider.
            Defaults to predefined common homoglyph types.
        replace_with_priority (bool, optional): Whether to replace characters based on priority.
            When True, replacements are chosen by order in the homoglyph lists.
            When False, replacements are chosen based on context or other strategies.
            Defaults to False.
        strategy (NormalizationStrategies): The normalization strategy to use.
            Defaults to LOCAL_CONTEXT, which selects replacements based on surrounding characters.

    Returns:
        str: The normalized text with homoglyphs replaced.

    Raises:
        ValueError: If the text is None or invalid parameters are provided.
        NotImplementedError: If an unsupported normalization strategy is specified.

    Example:
        ```python
        # Normalize text using the default local context strategy
        normalized_text = normalize_text("Hеllo wоrld")  # Contains Cyrillic 'е' and 'о'

        # Normalize text using dominant script strategy
        from silverspeak.homoglyphs.utils import NormalizationStrategies
        normalized_text = normalize_text(
            "Hеllo wоrld",
            strategy=NormalizationStrategies.DOMINANT_SCRIPT
        )
        ```
    """
    if text is None:
        raise ValueError("Input text cannot be None")

    if not text:
        return ""

    try:
        replacer = HomoglyphReplacer(
            unicode_categories_to_replace=unicode_categories_to_replace,
            types_of_homoglyphs_to_use=types_of_homoglyphs_to_use,
            replace_with_priority=replace_with_priority,
        )
        return replacer.normalize(text, strategy=strategy)
    except Exception as e:
        logger.error(f"Error normalizing text: {e}")
        raise
