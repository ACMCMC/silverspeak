"""
Local context-based normalization strategy for homoglyph replacement.

This module provides functionality to normalize text by analyzing the local
character context when choosing optimal homoglyph replacements.

Author: Aldan Creo (ACMC) <os@acmc.fyi>
"""

import logging
import unicodedata
from typing import List, Mapping

import unicodedataplus

from ..unicode_scoring import score_homoglyphs_for_context_window

logger = logging.getLogger(__name__)


def apply_local_context_strategy(
    text: str,
    normalization_map: Mapping[str, List[str]],
    N: int = 10,
) -> str:
    """
    Normalize text using local character context to choose optimal homoglyph replacements.

    This advanced normalization strategy analyzes the surrounding characters (context window)
    of each target character and selects replacement homoglyphs that best match the Unicode
    properties of the surrounding text. This preserves visual and semantic coherence by
    ensuring that replacement characters have similar properties to their context.

    Args:
        text (str): The input text to normalize.
        normalization_map (Mapping[str, List[str]]): A mapping from original characters to
            their possible homoglyph replacements.
        N (int, optional): The size of the context window to analyze around each character.
            Defaults to 10 characters (5 before and 5 after the target character).

    Returns:
        str: The normalized text with homoglyphs replaced according to local context.

    Note:
        This strategy is computationally more intensive than simple mapping strategies but
        produces more natural-looking results that preserve the visual consistency of the text.
    """
    if not text:
        logging.warning("Empty text provided for normalization")
        return ""

    if not normalization_map:
        logging.warning("Empty normalization map provided")
        return text

    # We'll use the scoring functionality from unicode_scoring module
    # No need for property extraction functions here

    # Process text character by character, analyzing context windows
    replaced_text = ""
    for i, char in enumerate(text):
        # Check if the character is in the mapping
        if char in normalization_map:
            # Get all possible homoglyph replacements including the original
            possible_chars = [char] + normalization_map[char]

            # Define the context window around the current character
            start = max(0, i - N // 2)
            end = min(len(text), i + N // 2 + 1)
            context_window = text[start:end]

            # Ensure we have a sufficiently sized window
            if len(context_window) < min(N, len(text)):
                if start == 0:
                    context_window = text[: min(N, len(text))]
                elif end == len(text):
                    context_window = text[-min(N, len(text)) :]

            # Score each possible replacement based on how well it matches the context window
            scores = []
            detailed_scores = []  # Keep track of detailed scores for debugging
            for possible_char in possible_chars:
                try:
                    # Use the score_homoglyphs_for_context_window function from unicode_scoring
                    score_dict = score_homoglyphs_for_context_window(
                        homoglyph=possible_char,
                        char=char,
                        context=context_window,
                        context_window_size=N,
                        PROPERTIES={
                            "script": {"fn": unicodedataplus.script, "weight": 3},
                            "block": {"fn": unicodedataplus.block, "weight": 5},
                            "category": {"fn": unicodedata.category, "weight": 10},
                            "bidirectional": {"fn": unicodedata.bidirectional, "weight": 2},
                            "east_asian_width": {"fn": unicodedata.east_asian_width, "weight": 1},
                        },
                    )
                    # Extract the total score for comparison
                    score = score_dict.get("total_score", 0.0)
                    scores.append((possible_char, score))
                    detailed_scores.append((possible_char, score_dict))
                except Exception as e:
                    logging.error(f"Error calculating score for '{possible_char}': {e}")
                    scores.append((possible_char, 0))  # Assign lowest score on error
                    detailed_scores.append((possible_char, {"total_score": 0.0, "error": str(e)}))

            if not scores:
                replaced_text += char
                continue

            # Select the best-scoring replacement
            best_char, best_score = max(scores, key=lambda x: x[1])

            # Log detailed scoring information for debugging
            # Find the detailed score for the best character
            best_detailed_score = next(
                (score_dict for char_name, score_dict in detailed_scores if char_name == best_char), {}
            )
            logging.debug(
                f"Character '{char}' at index {i}: chosen '{best_char}' with total score {best_score}. "
                f"Detailed scores: {best_detailed_score}. Context: '{context_window}'"
            )

            # Log a warning if multiple characters tie for the best score
            ties = [s[0] for s in scores if s[1] == best_score]
            if len(ties) > 1 and len(set(ties)) > 1:  # More than one unique character with best score
                # Get detailed scores for all tied characters
                tied_detailed_scores = [
                    (char_name, score_dict) for char_name, score_dict in detailed_scores if char_name in ties
                ]
                logging.debug(
                    f"Found a tie for the best character for '{char}' at index {i}. "
                    f"Options: {ties}. Using '{best_char}'. "
                    f"Tied detailed scores: {tied_detailed_scores}"
                )

            replaced_text += best_char
        else:
            # If the character is not in the mapping, keep it unchanged
            replaced_text += char

    return replaced_text
