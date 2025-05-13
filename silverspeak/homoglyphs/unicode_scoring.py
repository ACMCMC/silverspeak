"""
Unicode property scoring utilities for homoglyph context analysis.

This module provides utilities for scoring homoglyphs based on their
Unicode properties and how well they match the surrounding context.

Author: Aldan Creo (ACMC) <os@acmc.fyi>
"""

import logging
import unicodedata
from typing import Dict, Mapping

import unicodedataplus

logger = logging.getLogger(__name__)


def score_homoglyph_for_context(
    homoglyph: str,
    char: str,
    context: str,
    context_window_size: int = 10,
) -> float:
    """
    Score a homoglyph based on how well it matches the surrounding context.
    
    This method evaluates a potential homoglyph replacement by comparing its Unicode
    properties with those of the surrounding characters, assigning a score that
    reflects how well the homoglyph fits within the given context.
    
    Args:
        homoglyph (str): The homoglyph to evaluate.
        char (str): The original character that would be replaced.
        context (str): The surrounding text context for property matching.
        context_window_size (int): Size of the context window. Defaults to 10.
            
    Returns:
        float: A score indicating how well the homoglyph matches the context.
            Higher scores indicate better matches.
    """
    # Dictionary of Unicode property extraction functions
    PROPERTY_FNS = {
        "script": unicodedataplus.script,
        "block": unicodedataplus.block,
        "category": unicodedata.category,
        "bidirectional": unicodedata.bidirectional,
        "east_asian_width": unicodedata.east_asian_width,
    }
    
    # Define property weights for scoring
    PROPERTY_WEIGHTS = {
        "block": 10,       # Highest priority
        "category": 5,     # Medium-high priority
        "script": 3,       # Medium priority
        "bidirectional": 2, # Medium-low priority
        "east_asian_width": 1, # Low priority
    }
    
    # Skip if the homoglyph is the same as the character
    if homoglyph == char:
        return 0.0
        
    score = 0.0
    
    # Extract properties of the homoglyph
    try:
        homoglyph_props = {prop: PROPERTY_FNS[prop](homoglyph) for prop in PROPERTY_FNS}
        
        # Calculate context property matches
        for ctx_char in context:
            for prop, weight in PROPERTY_WEIGHTS.items():
                if PROPERTY_FNS[prop](ctx_char) == homoglyph_props[prop]:
                    score += weight
        
        # Add bonus for specific property combinations in context
        for ctx_char in context:
            # Block + script combination bonus
            if (PROPERTY_FNS["block"](ctx_char) == homoglyph_props["block"] and 
                PROPERTY_FNS["script"](ctx_char) == homoglyph_props["script"]):
                score += 3
            
            # Block + category combination bonus
            if (PROPERTY_FNS["block"](ctx_char) == homoglyph_props["block"] and 
                PROPERTY_FNS["category"](ctx_char) == homoglyph_props["category"]):
                score += 2
                
            # Script + category combination bonus
            if (PROPERTY_FNS["script"](ctx_char) == homoglyph_props["script"] and 
                PROPERTY_FNS["category"](ctx_char) == homoglyph_props["category"]):
                score += 2
                
        # Normalize score by context length to make it comparable across different contexts
        if len(context) > 0:
            score = score / len(context)
            
    except Exception as e:
        logger.error(f"Error scoring homoglyph '{homoglyph}': {e}")
        return 0.0
        
    return score
