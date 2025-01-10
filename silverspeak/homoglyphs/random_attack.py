# %%
from silverspeak.homoglyphs.identical_map import chars_map
import random


import logging

logger = logging.getLogger(__name__)


def random_attack(text: str, percentage=0.1, random_seed=42) -> str:
    """
    Replaces some characters in the text, randomly choosing which ones, leaving all others unchanged.
    """
    random.seed(random_seed)
    # Replace some characters in the text with their equivalent characters from the chars_map
    num_to_replace = int(len(text) * percentage)
    text = list(text)  # Convert to list to allow for in-place replacement
    replaceable_chars = [(i, char) for i, char in enumerate(text) if char in chars_map]
    replaceable_count = len(replaceable_chars)
    logger.debug(
        f"Found {replaceable_count} replaceable characters in the text. Will replace {num_to_replace} characters."
    )

    while num_to_replace > 0 and replaceable_count > 0:
        position, char = random.choice(replaceable_chars)
        text[position] = chars_map[char]
        num_to_replace -= 1
        replaceable_count -= 1
        replaceable_chars.remove((position, char))
        logger.debug(
            f"Replaced character {char} with {chars_map[char]}. {num_to_replace} characters left to replace."
        )

    if num_to_replace > 0:
        logger.warning("Not enough replaceable characters found in the text.")
    return "".join(text)


# %%
