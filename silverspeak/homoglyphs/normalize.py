from typing import List
from silverspeak.homoglyphs.homoglyph_replacer import HomoglyphReplacer
from silverspeak.homoglyphs.homoglyph_replacer import HomoglyphReplacer
from silverspeak.homoglyphs.utils import (
    _DEFAULT_HOMOGLYPHS_TO_USE,
    _DEFAULT_UNICODE_CATEGORIES_TO_REPLACE,
    TYPES_OF_HOMOGLYPHS,
)


def normalize_text(
    text: str,
    unicode_categories_to_replace=_DEFAULT_UNICODE_CATEGORIES_TO_REPLACE,
    types_of_homoglyphs_to_use: List[TYPES_OF_HOMOGLYPHS] = _DEFAULT_HOMOGLYPHS_TO_USE,
    replace_with_priority=False,
) -> str:
    replacer = HomoglyphReplacer(
        unicode_categories_to_replace=unicode_categories_to_replace,
        types_of_homoglyphs_to_use=types_of_homoglyphs_to_use,
        replace_with_priority=replace_with_priority,
    )
    return replacer.normalize(text)
