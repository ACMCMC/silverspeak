# Type hints for the types of homoglyphs
from enum import Enum

class TypesOfHomoglyphs(Enum):
    IDENTICAL = "identical"
    CONFUSABLES = "confusables"
    OCR = "ocr"
    OCR_REFINED = "ocr_refined"

_DEFAULT_UNICODE_CATEGORIES_TO_REPLACE = set(["Ll", "Lm", "Lo", "Lt", "Lu"])
_DEFAULT_HOMOGLYPHS_TO_USE = [
    TypesOfHomoglyphs.IDENTICAL,
    TypesOfHomoglyphs.CONFUSABLES,
    TypesOfHomoglyphs.OCR_REFINED,
]

class NormalizationStrategies(Enum):
    DOMINANT_SCRIPT = "dominant_script"
    DOMINANT_SCRIPT_AND_BLOCK = "dominant_script_and_block"
    CONTEXT_AWARE = "context_aware"
    TOKENIZATION = "tokenization"
