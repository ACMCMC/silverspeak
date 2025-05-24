"""
Unicode property scoring utilities for homoglyph context analysis.

This module provides utilities for scoring homoglyphs based on their
Unicode properties and how well they match the surrounding context.

Author: Aldan Creo (ACMC) <os@acmc.fyi>
"""

import logging
import unicodedata
from typing import Dict, Mapping, Union

import unicodedataplus

logger = logging.getLogger(__name__)


def score_homoglyphs_for_character(
    homoglyph: str,
    char: str,
    PROPERTIES: Mapping[str, Dict[str, Union[callable, int]]] = {
        "script": {"fn": unicodedataplus.script, "weight": 3},
        "block": {"fn": unicodedataplus.block, "weight": 5},
        "category": {"fn": unicodedata.category, "weight": 10},
        "bidirectional": {"fn": unicodedata.bidirectional, "weight": 2},
        "east_asian_width": {"fn": unicodedata.east_asian_width, "weight": 1},
    },
) -> float:
    """
    Score a homoglyph based on how well it matches the properties of the original character.

    This method evaluates a potential homoglyph replacement by comparing its Unicode
    properties with those of the original character to be replaced. It ignores context
    and focuses solely on matching the properties of the original character.

    Args:
        homoglyph (str): The homoglyph to evaluate.
        char (str): The original character that would be replaced.
        context (str, optional): Ignored in this implementation, kept for API compatibility.
        context_window_size (int, optional): Ignored in this implementation, kept for API compatibility.

    Returns:
        float: A score indicating how well the homoglyph matches the original character.
            Higher scores indicate better matches.
    """
    # Skip if the homoglyph is the same as the character
    if homoglyph == char:
        return 0.0

    score = 0.0

    try:
        # Extract properties of the original character
        char_props = {prop: PROPERTIES[prop]["fn"](char) for prop in PROPERTIES}

        # Extract properties of the homoglyph
        homoglyph_props = {prop: PROPERTIES[prop]["fn"](homoglyph) for prop in PROPERTIES}

        # Score based on matching properties
        for prop, weight in PROPERTIES.items():
            try:
                if homoglyph_props[prop] == char_props[prop]:
                    score += weight["weight"]
            except Exception:
                continue

        # Add bonus for specific property combinations
        combination_scores = {
            ("block", "script"): {"weight": 1},  # Block + script combination bonus
            ("block", "category"): {"weight": 1},  # Block + category combination bonus
            ("script", "category"): {"weight": 1},  # Script + category combination bonus
        }

        for (prop1, prop2), bonus_weight in combination_scores.items():
            try:
                if homoglyph_props[prop1] == char_props[prop1] and homoglyph_props[prop2] == char_props[prop2]:
                    score += bonus_weight["weight"]
            except Exception:
                continue

    except Exception as e:
        logger.error(f"Error scoring homoglyph '{homoglyph}': {e}")
        return 0.0

    return score
