"""
Unicode property scoring utilities for homoglyph context analysis.

This module provides utilities for scoring homoglyphs based on their
Unicode properties and how well they match the surrounding context.

Author: Aldan Creo (ACMC) <os@acmc.fyi>
"""

import logging
import unicodedata
from typing import Dict, Mapping, Union, Optional

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


def score_homoglyphs_for_context_window(
    homoglyph: str,
    char: str,
    context: str,
    context_window_size: int = 10,
    PROPERTIES: Mapping[str, Dict[str, Union[callable, int]]] = {
        "script": {"fn": unicodedataplus.script, "weight": 3},
        "block": {"fn": unicodedataplus.block, "weight": 5},
        "category": {"fn": unicodedata.category, "weight": 10},
        "bidirectional": {"fn": unicodedata.bidirectional, "weight": 2},
        "east_asian_width": {"fn": unicodedata.east_asian_width, "weight": 1},
    },
) -> Dict[str, float]:
    """
    Score a homoglyph based on how well it matches the properties of the characters in a given context window.

    Args:
        homoglyph (str): The homoglyph to evaluate.
        char (str): The original character that would be replaced.
        context (str): The context window containing surrounding characters.
        context_window_size (int, optional): The size of the context window. Defaults to 10.
        PROPERTIES (Mapping): Dictionary of Unicode properties and their weights.

    Returns:
        Dict[str, float]: A dictionary with individual property scores and total score.
            Keys include each property name plus 'total_score' for the aggregated score.
    """
    # Initialize property scores dictionary
    property_scores = {prop: 0.0 for prop in PROPERTIES.keys()}
    context_chars_analyzed = 0

    try:
        # Extract properties of the homoglyph
        homoglyph_props = {prop: PROPERTIES[prop]["fn"](homoglyph) for prop in PROPERTIES}

        # Analyze each character in the context window (excluding the target character position)
        for ctx_char in context:
            if ctx_char == char:  # Skip the original character we're replacing
                continue

            try:
                # Extract properties of the context character
                ctx_props = {prop: PROPERTIES[prop]["fn"](ctx_char) for prop in PROPERTIES}
                context_chars_analyzed += 1

                # Score based on how well homoglyph properties match context character properties
                for prop, weight_info in PROPERTIES.items():
                    try:
                        if homoglyph_props[prop] == ctx_props[prop]:
                            # Weight context matches lower than direct character matches
                            property_scores[prop] += weight_info["weight"]
                    except Exception:
                        continue

            except Exception as e:
                logger.debug(f"Error analyzing context character '{ctx_char}': {e}")
                continue

        # Normalize scores by number of analyzed characters
        if context_chars_analyzed > 0:
            for prop in property_scores:
                property_scores[prop] = property_scores[prop] / context_chars_analyzed

        # Calculate total score
        total_score = sum(property_scores.values())
        property_scores["total_score"] = total_score

    except Exception as e:
        logger.error(f"Error scoring homoglyph '{homoglyph}' in context '{context}': {e}")
        return {prop: 0.0 for prop in PROPERTIES.keys()} | {"total_score": 0.0}
    
    return property_scores
