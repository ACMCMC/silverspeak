"""
Dominant script-based normalization strategies for homoglyph replacement.

This module provides functionality to normalize text based on the dominant
Unicode script or script-block combination detected in the input.

Author: Aldan Creo (ACMC) <os@acmc.fyi>
"""

import logging
from typing import Any, Dict

from ..script_block_category_utils import detect_dominant_block, detect_dominant_script

logger = logging.getLogger(__name__)


def apply_dominant_script_strategy(replacer, text: str, **kwargs) -> str:
    """
    Normalize text based on the dominant Unicode script detected in the input.

    This function first identifies the dominant script in the text and then applies
    a normalization strategy using character mappings appropriate for that script.

    Args:
        replacer: Instance of HomoglyphReplacer that provides normalization mappings.
        text (str): The input text to normalize.
        **kwargs: Additional keyword arguments to pass to the normalization map generator.
            Commonly used kwargs include:
            - category (str): Unicode category to filter by.
            - preserve_case (bool): Whether to preserve character case during normalization.

    Returns:
        str: The normalized text with homoglyphs replaced according to the dominant script.

    Raises:
        ValueError: If the text is empty or the replacer is not properly initialized.

    Note:
        This strategy is most effective for texts predominantly written in a single script.
    """
    if not text:
        logging.warning("Empty text provided for normalization")
        return ""

    if not replacer:
        raise ValueError("No replacer provided for normalization")

    dominant_script = detect_dominant_script(text)

    if dominant_script == "Unknown":
        logging.warning("Unable to determine dominant script, normalization may be suboptimal")

    normalization_map = replacer.get_normalization_map_for_script_block_and_category(script=dominant_script, **kwargs)

    if not normalization_map:
        logging.warning(f"No normalization map available for script '{dominant_script}'")
        return text

    return text.translate(str.maketrans(normalization_map))


def apply_dominant_script_and_block_strategy(replacer, text: str, **kwargs) -> str:
    """
    Normalize text based on both the dominant Unicode script and block detected in the input.

    This function identifies both the dominant script and Unicode block in the text and then applies
    a normalization strategy using character mappings appropriate for that specific script-block combination.
    This is more precise than using just the script or just the block alone.

    Args:
        replacer: Instance of HomoglyphReplacer that provides normalization mappings.
        text (str): The input text to normalize.
        **kwargs: Additional keyword arguments to pass to the normalization map generator.
            Commonly used kwargs include:
            - category (str): Unicode category to filter by.
            - preserve_case (bool): Whether to preserve character case during normalization.

    Returns:
        str: The normalized text with homoglyphs replaced according to the dominant script and block.

    Raises:
        ValueError: If the text is empty or the replacer is not properly initialized.

    Note:
        This strategy is more specific than just using script detection alone and may provide
        better normalization for mixed-script texts where specific blocks are important.
    """
    if not text:
        logging.warning("Empty text provided for normalization")
        return ""

    if not replacer:
        raise ValueError("No replacer provided for normalization")

    dominant_script = detect_dominant_script(text)
    dominant_block = detect_dominant_block(text)

    if dominant_script == "Unknown" or dominant_block == "Unknown":
        logging.warning("Unable to determine dominant script/block, normalization may be suboptimal")

    normalization_map = replacer.get_normalization_map_for_script_block_and_category(
        script=dominant_script, block=dominant_block, **kwargs
    )

    if not normalization_map:
        logging.warning(f"No normalization map available for script '{dominant_script}' and block '{dominant_block}'")
        return text

    return text.translate(str.maketrans(normalization_map))
