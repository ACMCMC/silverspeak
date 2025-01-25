# %%
from typing import List, Literal

from silverspeak.homoglyphs.utils import HomoglyphReplacer


def greedy_attack(
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
    Fastest attack. It replaces all possible characters in the text.
    """
    replacer = HomoglyphReplacer(
        unicode_categories_to_replace=unicode_categories_to_replace,
        random_seed=random_seed,
        types_of_homoglyphs_to_use=types_of_homoglyphs_to_use,
        replace_with_priority=replace_with_priority,
    )

    # Replace all possible characters in the text with their equivalent characters from the chars_map
    result = []
    for char in text:
        if replacer.is_replaceable(char):
            # Replace with a random character from the set
            result.append(replacer.get_homoglpyh(char))
        else:
            result.append(char)
    return "".join(result)


# %%
