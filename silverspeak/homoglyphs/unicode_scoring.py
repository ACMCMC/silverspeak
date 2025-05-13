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
    reflects how well the homoglyph fits within the given context. The scoring now
    takes into account the prevalence of each property in the context window.
    
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
    
    # Base property weights - will be adjusted based on context prevalence
    BASE_PROPERTY_WEIGHTS = {
        "block": 10,       # Highest priority
        "category": 5,     # Medium-high priority
        "script": 3,       # Medium priority
        "bidirectional": 2, # Medium-low priority
        "east_asian_width": 1, # Low priority
    }
    
    # Skip if the homoglyph is the same as the character
    if homoglyph == char:
        return 0.0
    
    if not context or len(context) == 0:
        return 0.0
        
    score = 0.0
    
    try:
        # Extract properties of the homoglyph
        homoglyph_props = {prop: PROPERTY_FNS[prop](homoglyph) for prop in PROPERTY_FNS}
        
        # Calculate property prevalence in the context
        property_prevalence = {}
        for prop in PROPERTY_FNS:
            # Count occurrences of each property value in the context
            prop_values = {}
            for ctx_char in context:
                try:
                    prop_value = PROPERTY_FNS[prop](ctx_char)
                    prop_values[prop_value] = prop_values.get(prop_value, 0) + 1
                except Exception:
                    # Skip if we can't get property for this character
                    continue
            
            # Calculate prevalence for each property value
            for prop_value, count in prop_values.items():
                prevalence = count / len(context)
                property_prevalence[(prop, prop_value)] = prevalence
        
        # Calculate context-weighted scores for property matches
        for prop, base_weight in BASE_PROPERTY_WEIGHTS.items():
            try:
                homoglyph_prop_value = homoglyph_props[prop]
                # Use the prevalence as a weight factor for this property
                prevalence = property_prevalence.get((prop, homoglyph_prop_value), 0)
                # The more common this property is in the context, the more important it is
                context_weight = base_weight * prevalence
                
                # Calculate matches for this property
                matches = 0
                for ctx_char in context:
                    if PROPERTY_FNS[prop](ctx_char) == homoglyph_prop_value:
                        matches += 1
                
                # Add weighted score based on matches and context-adjusted weight
                if len(context) > 0:
                    match_score = (matches / len(context)) * context_weight
                    score += match_score
            except Exception as e:
                logger.debug(f"Error calculating score for property {prop}: {e}")
                continue
        
        # Add bonus for specific property combinations, also weighted by prevalence
        combination_scores = {
            ("block", "script"): 3,  # Block + script combination bonus
            ("block", "category"): 2,  # Block + category combination bonus
            ("script", "category"): 2,  # Script + category combination bonus
        }
        
        for (prop1, prop2), bonus_weight in combination_scores.items():
            try:
                # Get property values for the homoglyph
                prop1_value = homoglyph_props[prop1]
                prop2_value = homoglyph_props[prop2]
                
                # Count combined matches in context
                combo_matches = 0
                for ctx_char in context:
                    if (PROPERTY_FNS[prop1](ctx_char) == prop1_value and
                        PROPERTY_FNS[prop2](ctx_char) == prop2_value):
                        combo_matches += 1
                
                # Calculate combined prevalence
                combo_prevalence = combo_matches / len(context) if len(context) > 0 else 0
                
                # Add weighted score for property combination
                score += combo_prevalence * bonus_weight
            except Exception as e:
                logger.debug(f"Error calculating combo score for {prop1}+{prop2}: {e}")
                continue
        
    except Exception as e:
        logger.error(f"Error scoring homoglyph '{homoglyph}': {e}")
        return 0.0
        
    return score
