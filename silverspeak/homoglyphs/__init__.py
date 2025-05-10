"""
SilverSpeak Homoglyphs Module

This module provides functionality for detecting, replacing, and normalizing homoglyphs
(characters that look visually similar but have different Unicode code points).

The module implements both attack mechanisms (to replace standard characters with homoglyphs)
and normalization strategies (to convert homoglyphs back to standard characters).

Main components:
- random_attack: Generate text with random homoglyph replacements
- greedy_attack: Generate text with strategically chosen homoglyph replacements
- normalize_text: Normalize text by replacing homoglyphs with standard characters
- HomoglyphReplacer: Core class for homoglyph replacement operations

Author: Aldan Creo (ACMC) <os@acmc.fyi>
License: See LICENSE file in the project root
"""

from .greedy_attack import greedy_attack
from .homoglyph_replacer import HomoglyphReplacer
from .normalize import normalize_text
from .random_attack import random_attack
from .utils import NormalizationStrategies, TypesOfHomoglyphs

__all__ = [
    "greedy_attack",
    "random_attack",
    "normalize_text",
    "HomoglyphReplacer",
    "TypesOfHomoglyphs",
    "NormalizationStrategies",
]
