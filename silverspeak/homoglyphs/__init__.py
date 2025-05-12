"""
SilverSpeak Homoglyphs Module

This module provides functionality for detecting, replacing, and normalizing homoglyphs
(characters that look visually similar but have different Unicode code points).

The module implements both attack mechanisms (to replace standard characters with homoglyphs)
and normalization strategies (to convert homoglyphs back to standard characters).

Main components:
- random_attack: Generate text with random homoglyph replacements
- greedy_attack: Generate text with strategically chosen homoglyph replacements
- targeted_attack: Generate text with targeted homoglyph replacements
- normalize_text: Normalize text by replacing homoglyphs with standard characters
- HomoglyphReplacer: Core class for homoglyph replacement operations

Author: Aldan Creo (ACMC) <os@acmc.fyi>
License: See LICENSE file in the project root
"""

from .homoglyph_replacer import HomoglyphReplacer
from .normalize import normalize_text
from .utils import NormalizationStrategies, TypesOfHomoglyphs
from .attacks import greedy_attack, random_attack, targeted_attack

__all__ = [
    "normalize_text",
    "HomoglyphReplacer",
    "TypesOfHomoglyphs",
    "NormalizationStrategies",
    "greedy_attack",
    "random_attack",
    "targeted_attack",
]
