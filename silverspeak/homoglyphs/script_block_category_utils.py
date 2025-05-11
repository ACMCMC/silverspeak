import logging
from collections import Counter
from typing import Dict, List, Mapping, Optional, Set, Union

import unicodedataplus


def get_script_counts(text: str) -> Dict[str, int]:
    """
    Count the occurrences of each script in the text.

    Args:
        text (str): Text to analyze.

    Returns:
        Dict[str, int]: Counts of characters in each script.
    """
    script_counts = Counter(unicodedataplus.script(char) for char in text)
    return dict(script_counts)


def detect_dominant_script(text: str) -> str:
    """
    Detect the dominant script in the text.

    Args:
        text (str): Text to analyze.

    Returns:
        str: Dominant script in the text.
    """
    script_counts = get_script_counts(text=text)
    total_count = sum(script_counts.values())

    # Fix for mypy: explicit lambda to avoid type issue with key parameter
    dominant_script = max(script_counts.keys(), key=lambda k: script_counts[k])

    if script_counts[dominant_script] / total_count < 0.75:
        logging.warning(
            f"The dominant script '{dominant_script}' comprises less than 75% of the total character count. "
            f"This is unusual, as most texts predominantly consist of characters from a single script. "
            f"Proceed with caution, as this may affect the reliability of the analysis."
        )
    return dominant_script


def get_block_counts(text: str) -> Dict[str, int]:
    """
    Count the number of characters in each Unicode block in the text.

    Args:
        text (str): Text to analyze.

    Returns:
        Dict[str, int]: Counts of characters in each Unicode block.
    """
    block_counts = Counter(unicodedataplus.block(char) for char in text)
    return dict(block_counts)


def detect_dominant_block(text: str) -> str:
    """
    Detect the dominant Unicode block in the text.

    Args:
        text (str): Text to analyze.

    Returns:
        str: Dominant Unicode block in the text.
    """
    block_counts = get_block_counts(text=text)
    total_count = sum(block_counts.values())

    # Fix for mypy: explicit lambda to avoid type issue with key parameter
    dominant_block = max(block_counts.keys(), key=lambda k: block_counts[k])

    if block_counts[dominant_block] / total_count < 0.75:
        logging.warning(
            f"The dominant Unicode block '{dominant_block}' comprises less than 75% of the total character count. "
            f"This is unusual, as most texts predominantly consist of characters from a single block. "
            f"Proceed with caution, as this may affect the reliability of the analysis."
        )
    return dominant_block


def is_script_and_block(text: str, script: Optional[str], block: Optional[str]) -> bool:
    """
    Check if all characters in the text belong to the specified script and block.

    Args:
        text (str): Text to analyze.
        script (Optional[str]): Script to check against.
        block (Optional[str]): Unicode block to check against.

    Returns:
        bool: True if all characters belong to the specified script/block, False otherwise.
    """
    if not text:
        return True

    if not script:
        return True

    if not block:
        return all(unicodedataplus.script(char) == script for char in text)

    return all(unicodedataplus.script(char) == script and unicodedataplus.block(char) == block for char in text)


def is_category(
    text: str,
    category: Union[str, List[str], Set[str]],
) -> bool:
    """
    Check if all characters in the text belong to the specified Unicode category.

    Args:
        text (str): Text to analyze.
        category (Union[str, List[str], Set[str]]): Unicode category or collection of categories.

    Returns:
        bool: True if all characters belong to the specified category, False otherwise.
    """
    if isinstance(category, str):
        category = [category]

    # Convert category to a list if it's a set (for compatibility)
    categories = list(category)

    if not text:
        return True

    return all(unicodedataplus.category(char) in categories for char in text)
