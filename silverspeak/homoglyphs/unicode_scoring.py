"""
Unicode property scoring utilities for homoglyph context analysis.

Author: Aldan Creo (ACMC) <os@acmc.fyi>
License: See LICENSE file in the project root
"""

import logging
import unicodedata
from typing import Any, Callable, Dict, Mapping, TypedDict

import unicodedataplus

logger = logging.getLogger(__name__)


class PropertySpec(TypedDict):
    fn: Callable[[str], Any]
    weight: int


DEFAULT_PROPERTIES: Mapping[str, PropertySpec] = {
    "script": {"fn": unicodedataplus.script, "weight": 3},
    "block": {"fn": unicodedataplus.block, "weight": 5},
    "category": {"fn": unicodedata.category, "weight": 10},
    "bidirectional": {"fn": unicodedata.bidirectional, "weight": 2},
    "east_asian_width": {"fn": unicodedata.east_asian_width, "weight": 1},
}

TARGETED_PROPERTIES: Mapping[str, PropertySpec] = {
    "script": {"fn": unicodedataplus.script, "weight": 2},
    "block": {"fn": unicodedataplus.block, "weight": 5},
    "plane": {"fn": lambda c: ord(c) >> 16, "weight": 3},
    "category": {"fn": unicodedata.category, "weight": 2},
    "bidirectional": {"fn": unicodedata.bidirectional, "weight": 2},
    "east_asian_width": {"fn": unicodedata.east_asian_width, "weight": 1},
}


def score_homoglyphs_for_character(
    homoglyph: str,
    char: str,
    PROPERTIES: Mapping[str, PropertySpec] = DEFAULT_PROPERTIES,
) -> float:
    if homoglyph == char:
        return 0.0

    score = 0.0

    try:
        char_props = {prop: PROPERTIES[prop]["fn"](char) for prop in PROPERTIES}
        homoglyph_props = {prop: PROPERTIES[prop]["fn"](homoglyph) for prop in PROPERTIES}

        for prop, weight in PROPERTIES.items():
            try:
                if homoglyph_props[prop] == char_props[prop]:
                    score += weight["weight"]
            except Exception:
                continue

        combination_scores = {
            ("block", "script"): {"weight": 1},
            ("block", "category"): {"weight": 1},
            ("script", "category"): {"weight": 1},
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
    PROPERTIES: Mapping[str, PropertySpec] = DEFAULT_PROPERTIES,
) -> Dict[str, float]:
    property_scores = {prop: 0.0 for prop in PROPERTIES.keys()}
    context_chars_analyzed = 0

    try:
        homoglyph_props = {prop: PROPERTIES[prop]["fn"](homoglyph) for prop in PROPERTIES}

        for ctx_char in context:
            if ctx_char == char:
                continue

            try:
                ctx_props = {prop: PROPERTIES[prop]["fn"](ctx_char) for prop in PROPERTIES}
                context_chars_analyzed += 1

                for prop, weight_info in PROPERTIES.items():
                    try:
                        if homoglyph_props[prop] == ctx_props[prop]:
                            property_scores[prop] += weight_info["weight"]
                    except Exception:
                        continue

            except Exception as e:
                logger.debug(f"Error analyzing context character '{ctx_char}': {e}")
                continue

        if context_chars_analyzed > 0:
            for prop in property_scores:
                property_scores[prop] = property_scores[prop] / context_chars_analyzed

        property_scores["total_score"] = sum(property_scores.values())

    except Exception as e:
        logger.error(f"Error scoring homoglyph '{homoglyph}' in context '{context}': {e}")
        return {prop: 0.0 for prop in PROPERTIES.keys()} | {"total_score": 0.0}

    return property_scores
