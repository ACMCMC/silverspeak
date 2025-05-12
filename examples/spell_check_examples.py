#!/usr/bin/env python3
"""
Example script demonstrating the spell check normalization strategy in SilverSpeak.

This script shows advanced usage of the spell check strategy:
1. Basic spell checking in English
2. Multi-language support
3. Custom dictionary support
4. Contextual spell checking

To run this example, first install the required dependencies:
    poetry install --with spell-check contextual-spell-check
"""

import logging
from silverspeak.homoglyphs.normalize import normalize_text
from silverspeak.homoglyphs.utils import NormalizationStrategies

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def spell_check_basic():
    """Demonstrate basic spell checking functionality."""
    sample_text = "Tһis іs а tеst with ѕome һomoglурhs."  # Contains various homoglyphs
    
    logger.info("BASIC SPELL CHECKING")
    logger.info("-" * 50)
    logger.info(f"Original text: {sample_text}")
    
    try:
        normalized_text = normalize_text(
            sample_text,
            strategy=NormalizationStrategies.SPELL_CHECK
        )
        logger.info(f"Default settings: {normalized_text}")
        
        # Adjusting distance parameters
        normalized_strict = normalize_text(
            sample_text,
            strategy=NormalizationStrategies.SPELL_CHECK,
            distance=1  # More strict correction (default is 2)
        )
        logger.info(f"Stricter corrections (distance=1): {normalized_strict}")
        
        normalized_lenient = normalize_text(
            sample_text,
            strategy=NormalizationStrategies.SPELL_CHECK,
            distance_threshold=0.6  # More lenient threshold (default is 0.8)
        )
        logger.info(f"More lenient threshold (threshold=0.6): {normalized_lenient}")
        
    except ImportError:
        logger.error("Spell checking dependencies not installed. "
                   "Install them using: poetry install --with spell-check")
    logger.info("")


def spell_check_multilingual():
    """Demonstrate multi-language support in spell checking."""
    logger.info("MULTI-LANGUAGE SUPPORT")
    logger.info("-" * 50)
    
    try:
        # Spanish example
        spanish_text = "Вuеnоs díаs аmіgо"  # "Buenos días amigo" with homoglyphs
        logger.info(f"Spanish text with homoglyphs: {spanish_text}")
        
        normalized_spanish = normalize_text(
            spanish_text,
            strategy=NormalizationStrategies.SPELL_CHECK,
            language="es"
        )
        logger.info(f"Spanish normalization: {normalized_spanish}")
        
        # French example 
        french_text = "Вonjоur lе mоndе"  # "Bonjour le monde" with homoglyphs
        logger.info(f"French text with homoglyphs: {french_text}")
        
        normalized_french = normalize_text(
            french_text,
            strategy=NormalizationStrategies.SPELL_CHECK,
            language="fr"
        )
        logger.info(f"French normalization: {normalized_french}")
        
    except ImportError:
        logger.error("Spell checking dependencies not installed. "
                   "Install them using: poetry install --with spell-check")
    logger.info("")


def spell_check_custom_dictionary():
    """Demonstrate custom dictionary support in spell checking."""
    logger.info("CUSTOM DICTIONARY SUPPORT")
    logger.info("-" * 50)
    
    try:
        # Technical text with homoglyphs and domain-specific terms
        tech_text = "SіlvеrSреаk is а Рythоn lіbrаrу for homoglурh dеtеctіon аnd normаlizаtіon"
        logger.info(f"Technical text with homoglyphs: {tech_text}")
        
        # Without custom dictionary
        normalized_default = normalize_text(
            tech_text,
            strategy=NormalizationStrategies.SPELL_CHECK
        )
        logger.info(f"Without custom dictionary: {normalized_default}")
        
        # With custom dictionary for technical terms
        custom_dictionary = [
            "SilverSpeak", "Python", "library", "homoglyph", 
            "detection", "normalization"
        ]
        
        normalized_custom = normalize_text(
            tech_text,
            strategy=NormalizationStrategies.SPELL_CHECK,
            custom_words=custom_dictionary
        )
        logger.info(f"With custom dictionary: {normalized_custom}")
        
    except ImportError:
        logger.error("Spell checking dependencies not installed. "
                   "Install them using: poetry install --with spell-check")
    logger.info("")


def spell_check_contextual():
    """Demonstrate contextual spell checking."""
    logger.info("CONTEXTUAL SPELL CHECKING")
    logger.info("-" * 50)
    
    # Context-dependent examples
    context_text1 = "I like to еаt аn аррlе."  # "eat an apple" with homoglyphs
    context_text2 = "The рhоnе will rіng soon."  # "phone will ring" with homoglyphs
    
    logger.info(f"Context example 1: {context_text1}")
    logger.info(f"Context example 2: {context_text2}")
    
    try:
        # Regular spell checking
        normalized_regular1 = normalize_text(
            context_text1,
            strategy=NormalizationStrategies.SPELL_CHECK
        )
        logger.info(f"Regular spell check (1): {normalized_regular1}")
        
        normalized_regular2 = normalize_text(
            context_text2,
            strategy=NormalizationStrategies.SPELL_CHECK
        )
        logger.info(f"Regular spell check (2): {normalized_regular2}")
        
        # Try contextual spell checking if available
        try:
            normalized_contextual1 = normalize_text(
                context_text1,
                strategy=NormalizationStrategies.SPELL_CHECK,
                use_contextual=True
            )
            logger.info(f"Contextual spell check (1): {normalized_contextual1}")
            
            normalized_contextual2 = normalize_text(
                context_text2,
                strategy=NormalizationStrategies.SPELL_CHECK,
                use_contextual=True
            )
            logger.info(f"Contextual spell check (2): {normalized_contextual2}")
            
        except ImportError:
            logger.warning("Contextual spell checking dependencies not installed. "
                         "Install them using: poetry install --with contextual-spell-check")
            
    except ImportError:
        logger.error("Spell checking dependencies not installed. "
                   "Install them using: poetry install --with spell-check")
    logger.info("")


if __name__ == "__main__":
    logger.info("SilverSpeak Spell Check Normalization Examples")
    logger.info("=" * 50)
    logger.info("")
    
    # Demonstrate each aspect of spell checking
    spell_check_basic()
    spell_check_multilingual()
    spell_check_custom_dictionary()
    spell_check_contextual()
