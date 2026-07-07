import logging
from collections import Counter
from typing import Dict

import unicodedataplus


def char_script(char: str) -> str:
    return str(unicodedataplus.script(char))


def char_block(char: str) -> str:
    return str(unicodedataplus.block(char))


def get_script_counts(text: str) -> Dict[str, int]:
    return dict(Counter(char_script(char=char) for char in text))


def detect_dominant_script(text: str) -> str:
    script_counts = get_script_counts(text=text)
    total_count = sum(script_counts.values())
    dominant_script = max(script_counts.keys(), key=lambda k: script_counts[k])
    if script_counts[dominant_script] / total_count < 0.75:
        logging.warning(
            f"The dominant script '{dominant_script}' comprises less than 75% of the total character count."
        )
    return dominant_script


def get_block_counts(text: str) -> Dict[str, int]:
    return dict(Counter(char_block(char=char) for char in text))


def detect_dominant_block(text: str) -> str:
    block_counts = get_block_counts(text=text)
    total_count = sum(block_counts.values())
    dominant_block = max(block_counts.keys(), key=lambda k: block_counts[k])
    if block_counts[dominant_block] / total_count < 0.75:
        logging.warning(
            f"The dominant Unicode block '{dominant_block}' comprises less than 75% of the total character count."
        )
    return dominant_block
