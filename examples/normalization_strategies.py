#!/usr/bin/env python3
"""
Example script demonstrating the various normalization strategies in SilverSpeak.

This script shows how to use:
1. Language Model normalization strategy
2. LLM Prompt normalization strategy
3. Spell Check normalization strategy

To run this example, first install the required dependencies:
    poetry install --with spell-check contextual-spell-check
"""

import logging
from silverspeak.homoglyphs.normalize import normalize_text
from silverspeak.homoglyphs.utils import NormalizationStrategies

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Sample text with homoglyphs
sample_text = "Tһis іs а tеst with ѕome һomoglурhs."  # Contains various homoglyphs


def demonstrate_language_model_strategy():
    """Demonstrate the language model normalization strategy."""
    logger.info("LANGUAGE MODEL NORMALIZATION STRATEGY")
    logger.info("-" * 50)
    logger.info(f"Original text: {sample_text}")

    # Word-level normalization (recommended)
    normalized_text = normalize_text(
        sample_text,
        strategy=NormalizationStrategies.LANGUAGE_MODEL,
        word_level=True,  # Use word-level masking (default)
        model_name="bert-base-uncased"  # Optional: specify a different model
    )
    
    logger.info(f"Normalized text: {normalized_text}")
    logger.info("")


def demonstrate_llm_prompt_strategy():
    """Demonstrate the LLM prompt normalization strategy."""
    logger.info("LLM PROMPT NORMALIZATION STRATEGY")
    logger.info("-" * 50)
    logger.info(f"Original text: {sample_text}")

    try:
        # LLM-based normalization
        normalized_text = normalize_text(
            sample_text,
            strategy=NormalizationStrategies.LLM_PROMPT,
            model_name="google/gemma-2-1b-it",  # Optional: specify a different model
            temperature=0.0  # Set to 0 for deterministic output
        )
        
        logger.info(f"Normalized text: {normalized_text}")
    except ImportError:
        logger.error("Required dependencies not installed. "
                     "Install them using: poetry install")
    logger.info("")


def demonstrate_spell_check_strategy():
    """Demonstrate the spell check normalization strategy."""
    logger.info("SPELL CHECK NORMALIZATION STRATEGY")
    logger.info("-" * 50)
    logger.info(f"Original text: {sample_text}")

    try:
        # Basic spell check (English)
        normalized_text = normalize_text(
            sample_text,
            strategy=NormalizationStrategies.SPELL_CHECK,
            language="en",  # Default is English
            distance=2,  # Maximum edit distance for corrections
            distance_threshold=0.7  # Similarity threshold
        )
        
        logger.info(f"Basic spell check (English): {normalized_text}")
        
        # With custom dictionary
        custom_text = "SіlvеrSреаk is а lіbrаrу for homoglурh dеtеctіon"
        logger.info(f"Custom dictionary example: {custom_text}")
        
        normalized_custom = normalize_text(
            custom_text,
            strategy=NormalizationStrategies.SPELL_CHECK,
            custom_words=["SilverSpeak", "homoglyph", "library", "detection"],
            language="en"
        )
        
        logger.info(f"With custom dictionary: {normalized_custom}")
        
        # Attempt contextual spell checking if available
        try:
            contextual_text = normalize_text(
                "Tһe cat is јumріng оn tһe fеnсe",
                strategy=NormalizationStrategies.SPELL_CHECK,
                use_contextual=True
            )
            logger.info(f"Contextual spell check: {contextual_text}")
        except ImportError:
            logger.warning("Contextual spell checking dependencies not installed. "
                         "Install them using: poetry install --with contextual-spell-check")
            
    except ImportError:
        logger.error("Spell checking dependencies not installed. "
                   "Install them using: poetry install --with spell-check")
    logger.info("")


if __name__ == "__main__":
    logger.info("SilverSpeak Normalization Strategies Example")
    logger.info("=" * 50)
    logger.info("")
    
    # Demonstrate each strategy
    demonstrate_language_model_strategy()
    demonstrate_llm_prompt_strategy()
    demonstrate_spell_check_strategy()
