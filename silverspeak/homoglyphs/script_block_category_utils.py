from collections import Counter
import unicodedataplus
import logging
from typing import List, Mapping, Optional, Union


def get_script_counts(text: str) -> Mapping[str, int]:
    """
    Count the occurrences of each script in the text.

    Args:
        text (str): Text to analyze.

    Returns:
        Mapping[str, int]: Counts of characters in each script.
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
    dominant_script = max(script_counts, key=script_counts.get)
    if script_counts[dominant_script] / total_count < 0.75:
        logging.warning(
            f"The dominant script '{dominant_script}' comprises less than 75% of the total character count. This is unusual, as most texts predominantly consist of characters from a single script. Proceed with caution, as this may affect the reliability of the analysis."
        )
    return dominant_script


def get_block_counts(text: str) -> Mapping[str, int]:
    """
    Count the number of characters in each Unicode block in the text.

    Args:
        text (str): Text to analyze.

    Returns:
        Mapping[str, int]: Counts of characters in each Unicode block.
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
    dominant_block = max(block_counts, key=block_counts.get)
    if block_counts[dominant_block] / total_count < 0.75:
        logging.warning(
            f"The dominant Unicode block '{dominant_block}' comprises less than 75% of the total character count. This is unusual, as most texts predominantly consist of characters from a single block. Proceed with caution, as this may affect the reliability of the analysis."
        )
    return dominant_block


def is_script_and_block(text: str, script: Optional[str], block: Optional[str]) -> bool:
    if not block:
        return all(unicodedataplus.script(char) == script for char in text)
    return all(
        unicodedataplus.script(char) == script and unicodedataplus.block(char) == block
        for char in text
    )


def is_category(
    text: str,
    category: Union[str, List[str]],
):
    """
    Check if all characters in the text belong to the specified Unicode category.

    Args:
        text (str): Text to analyze.
        category (Union[str, List[str]]): Unicode category or list of categories to check against.

    Returns:
        bool: True if all characters belong to the specified category, False otherwise.
    """
    if isinstance(category, str):
        category = [category]
    return all(unicodedataplus.category(char) in category for char in text)
