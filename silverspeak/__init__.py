"""
SilverSpeak: A professional library for text normalization and homoglyph detection/replacement.

This library provides tools for detecting and normalizing homoglyphs (characters 
that look similar but have different Unicode code points), which can be used for
text normalization, security applications, and adversarial text generation.

Main components:
- random_attack: Generate text with random homoglyph replacements
- greedy_attack: Generate text with strategically chosen homoglyph replacements
- normalize_text: Normalize text by replacing homoglyphs with standard characters
- HomoglyphReplacer: Core class for homoglyph replacement operations

Author: Aldan Creo (ACMC) <os@acmc.fyi>
Version: 1.0.0
License: See LICENSE file in the project root
"""

__version__ = "1.0.0"

from silverspeak.homoglyphs.random_attack import random_attack
from silverspeak.homoglyphs.greedy_attack import greedy_attack
from silverspeak.homoglyphs.normalize import normalize_text
from silverspeak.homoglyphs.homoglyph_replacer import HomoglyphReplacer
from silverspeak.homoglyphs.utils import TypesOfHomoglyphs, NormalizationStrategies

__all__ = [
    "random_attack",
    "greedy_attack",
    "normalize_text",
    "HomoglyphReplacer",
    "TypesOfHomoglyphs",
    "NormalizationStrategies",
]