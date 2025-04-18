# Type hints for the types of homoglyphs
from typing import Literal

TYPES_OF_HOMOGLYPHS = Literal[
    "identical", "confusables", "ocr", "ocr_refined"
]

_DEFAULT_UNICODE_CATEGORIES_TO_REPLACE = set(["Ll", "Lm", "Lo", "Lt", "Lu"])
_DEFAULT_HOMOGLYPHS_TO_USE = [
    "identical",
    "confusables",
    "ocr_refined",
]
