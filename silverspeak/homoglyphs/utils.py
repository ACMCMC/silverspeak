"""
Utility constants and enumerations for homoglyph operations.

Author: Aldan Creo (ACMC) <os@acmc.fyi>
License: See LICENSE file in the project root
"""

from enum import Enum
from typing import List, Set


class TypesOfHomoglyphs(Enum):
    IDENTICAL = "identical"
    CONFUSABLES = "confusables"
    OCR = "ocr"
    OCR_REFINED = "ocr_refined"


_DEFAULT_UNICODE_CATEGORIES_TO_REPLACE: Set[str] = set(["Ll", "Lm", "Lo", "Lt", "Lu"])

_DEFAULT_HOMOGLYPHS_TO_USE: List[TypesOfHomoglyphs] = [
    TypesOfHomoglyphs.IDENTICAL,
    TypesOfHomoglyphs.CONFUSABLES,
    TypesOfHomoglyphs.OCR_REFINED,
]
