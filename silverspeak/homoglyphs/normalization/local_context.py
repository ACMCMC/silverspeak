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

    # Dictionary of Unicode property extraction functions
    PROPERTY_FNS = {
        "script": unicodedataplus.script,
        "block": unicodedataplus.block,
        "category": unicodedataplus.category,
        "vertical_orientation": unicodedataplus.vertical_orientation,
        "bidirectional": unicodedata.bidirectional,
        "east_asian_width": unicodedata.east_asian_width,
        # These properties are not used in the current implementation because they are not useful - just keep them here for reference but actually it doesn't make sense to use them because the surrounding characters are not expected to share these properties
        # "combining": unicodedata.combining,
        # "mirrored": unicodedata.mirrored,
    }

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

            # Extract Unicode properties from the context window
            try:
                properties = {prop: [PROPERTY_FNS[prop](c) for c in context_window] for prop in PROPERTY_FNS}
            except Exception as e:
                logging.error(f"Error extracting Unicode properties: {e}")
                replaced_text += char
                continue

            # Define property weights for hierarchical scoring
            PROPERTY_WEIGHTS = {
                "block": 10,       # Highest priority
                "category": 5,     # Third highest priority
                "script": 3,       # Second highest priority
                "bidirectional": 2,
                "east_asian_width": 1,
                "vertical_orientation": 1,
            }
            
            # Calculate weighted property matching scores for each possible replacement
            scores = []
            for possible_char in possible_chars:
                try:
                    # Initialize score
                    score = 0
                    
                    # Extract properties of the possible character once
                    char_props = {prop: PROPERTY_FNS[prop](possible_char) for prop in PROPERTY_FNS}
                    
                    # Count weighted matches for each property
                    for prop, weight in PROPERTY_WEIGHTS.items():
                        context_values = properties[prop]
                        # Add weight for each match in context
                        matches = sum(char_props[prop] == value for value in context_values)
                        score += matches * weight
                    
                    # Add bonus points for specific property combinations
                    for ctx_index, _ in enumerate(context_window):
                        combo_bonus = 0
                        # Check for block + script combination
                        if (ctx_index < len(context_window) and 
                            char_props["block"] == properties["block"][ctx_index] and 
                            char_props["script"] == properties["script"][ctx_index]):
                            combo_bonus += 3
                        
                        # Check for block + category combination
                        if (ctx_index < len(context_window) and 
                            char_props["block"] == properties["block"][ctx_index] and 
                            char_props["category"] == properties["category"][ctx_index]):
                            combo_bonus += 2
                            
                        # Check for script + category combination
                        if (ctx_index < len(context_window) and 
                            char_props["script"] == properties["script"][ctx_index] and 
                            char_props["category"] == properties["category"][ctx_index]):
                            combo_bonus += 2
                            
                        score += combo_bonus
                    
                    scores.append((possible_char, score))
                except Exception as e:
                    logging.error(f"Error calculating score for '{possible_char}': {e}")
                    scores.append((possible_char, 0))  # Assign lowest score on error

            if not scores:
                replaced_text += char
                continue

            # Select the best-scoring replacement
            best_char, best_score = max(scores, key=lambda x: x[1])

            # Log a warning if multiple characters tie for the best score
            ties = [s[0] for s in scores if s[1] == best_score]
            if len(ties) > 1 and len(set(ties)) > 1:  # More than one unique character with best score
                if logging.getLogger().isEnabledFor(logging.DEBUG):
                    logging.debug(
                        f"Found a tie for the best character for '{char}' at index {i}. "
                        f"Options: {ties}. Using '{best_char}'."
                    )

            replaced_text += best_char
        else:
            # If the character is not in the mapping, keep it unchanged
            replaced_text += char

    return replaced_text
