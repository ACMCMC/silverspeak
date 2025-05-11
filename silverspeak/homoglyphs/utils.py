"""
Utility constants and enumerations for homoglyph operations.

This module provides enumerations and constants used throughout the SilverSpeak
homoglyph detection and replacement system.

Author: Aldan Creo (ACMC) <os@acmc.fyi>
License: See LICENSE file in the project root
"""

from enum import Enum
from typing import List, Set


class TypesOfHomoglyphs(Enum):
    """
    Enumeration of the different types of homoglyphs supported by SilverSpeak.

    Values:
        IDENTICAL: Characters that are visually identical in most fonts but have
                   different Unicode code points.
        CONFUSABLES: Characters identified as confusables by Unicode.
        OCR: Characters that might be confused in OCR systems.
        OCR_REFINED: A refined subset of OCR confusables with high visual similarity.
    """

    IDENTICAL = "identical"
    CONFUSABLES = "confusables"
    OCR = "ocr"
    OCR_REFINED = "ocr_refined"


# Unicode categories for letter characters
# Ll: Lowercase letter
# Lm: Modifier letter
# Lo: Other letter (includes many non-Latin scripts)
# Lt: Titlecase letter
# Lu: Uppercase letter
_DEFAULT_UNICODE_CATEGORIES_TO_REPLACE: Set[str] = set(["Ll", "Lm", "Lo", "Lt", "Lu"])

# Default homoglyph types to use in attacks and normalization
_DEFAULT_HOMOGLYPHS_TO_USE: List[TypesOfHomoglyphs] = [
    TypesOfHomoglyphs.IDENTICAL,
    TypesOfHomoglyphs.CONFUSABLES,
    TypesOfHomoglyphs.OCR_REFINED,
]


class NormalizationStrategies(Enum):
    """
    Enumeration of text normalization strategies for homoglyph replacement.

    Values:
        DOMINANT_SCRIPT: Normalize based on the dominant Unicode script in the text.
        DOMINANT_SCRIPT_AND_BLOCK: Normalize based on both dominant script and Unicode block.
        LOCAL_CONTEXT: Normalize based on surrounding character context.
        TOKENIZATION: Normalize based on tokenization of the text.
        LANGUAGE_MODEL: Normalize using a masked language model to determine the most likely characters.
        LLM_PROMPT: Normalize using a generative language model prompted to fix homoglyphs.
        SPELL_CHECK: Normalize using spelling correction algorithms with multilingual support.
        NGRAM: Normalize using character n-gram frequency analysis.
        OCR_CONFIDENCE: Normalize using OCR confidence scores or confusion matrices.
        GRAPH_BASED: Normalize using a graph-based character similarity network.
    """

    DOMINANT_SCRIPT = "dominant_script"
    DOMINANT_SCRIPT_AND_BLOCK = "dominant_script_and_block"
    LOCAL_CONTEXT = "local_context"
    TOKENIZATION = "tokenization"
    LANGUAGE_MODEL = "language_model"
    LLM_PROMPT = "llm_prompt"
    SPELL_CHECK = "spell_check"
    NGRAM = "ngram"
    OCR_CONFIDENCE = "ocr_confidence"
    GRAPH_BASED = "graph_based"
