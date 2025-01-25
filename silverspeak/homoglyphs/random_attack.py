# %%
import logging
import random
from typing import List, Literal

from silverspeak.homoglyphs.utils import HomoglyphReplacer

logger = logging.getLogger(__name__)


def random_attack(
    text: str,
    percentage=0.1,
    random_seed=42,
    unicode_categories_to_replace=["Ll", "Lm", "Lo", "Lt", "Lu"],
    types_of_homoglyphs_to_use: List[Literal["identical", "confusables", "ocr"]] = [
        "identical",
        "confusables",
        "ocr",
    ],
    replace_with_priority: bool = False,
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
    replaceable_chars = [
        (i, char) for i, char in enumerate(text) if replacer.is_replaceable(char)
    ]
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
        replacement = replacer.get_homoglpyh(char)
        text[position] = replacement
        num_to_replace -= 1
        replaceable_count -= 1
        replaceable_chars.remove((position, char))
        logger.debug(
            f"Replaced character {char} with {replacement}. {num_to_replace} characters left to replace."
        )

    return "".join(text)


# %%
