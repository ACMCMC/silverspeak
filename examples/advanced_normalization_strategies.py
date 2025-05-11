#!/usr/bin/env python3
"""
Example script demonstrating the additional normalization strategies in SilverSpeak.

This script shows the use of:
1. N-gram frequency-based normalization strategy
2. OCR confidence-based normalization strategy
3. Graph-based character network normalization strategy

To run this example with full functionality, first install the required dependencies:
    poetry install
    poetry install --with ngram-analysis graph-analysis
    pip install pytesseract pillow  # For OCR confidence strategy

Each strategy can work with basic functionality even without the additional dependencies,
but will perform better with the appropriate libraries installed.
"""

import logging
from silverspeak.homoglyphs.normalize import normalize_text
from silverspeak.homoglyphs.utils import NormalizationStrategies

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def ngram_strategy_example():
    """Demonstrate n-gram frequency-based normalization."""
    sample_text = "Tһis іs а tеst with ѕome һomoglурhs."  # Contains various homoglyphs
    
    logger.info("N-GRAM FREQUENCY STRATEGY")
    logger.info("-" * 50)
    logger.info(f"Original text: {sample_text}")
    
    try:
        # Basic usage
        normalized_text = normalize_text(
            sample_text,
            strategy=NormalizationStrategies.NGRAM
        )
        logger.info(f"Default settings: {normalized_text}")
        
        # Adjusting threshold
        normalized_strict = normalize_text(
            sample_text,
            strategy=NormalizationStrategies.NGRAM,
            threshold=0.000001  # More strict threshold (default is 0.00001)
        )
        logger.info(f"Stricter threshold: {normalized_strict}")
        
    except Exception as e:
        logger.error(f"Error: {e}")
    logger.info("")


def ocr_confidence_strategy_example():
    """Demonstrate OCR confidence-based normalization."""
    sample_text = "Tһis іs а tеst with ѕome һomoglурhs."  # Contains various homoglyphs
    
    logger.info("OCR CONFIDENCE STRATEGY")
    logger.info("-" * 50)
    logger.info(f"Original text: {sample_text}")
    
    try:
        # Basic usage
        normalized_text = normalize_text(
            sample_text,
            strategy=NormalizationStrategies.OCR_CONFIDENCE
        )
        logger.info(f"Default settings: {normalized_text}")
        
        # Adjusting confidence threshold
        normalized_strict = normalize_text(
            sample_text,
            strategy=NormalizationStrategies.OCR_CONFIDENCE,
            confidence_threshold=0.8  # Higher confidence threshold (default is 0.7)
        )
        logger.info(f"Higher confidence threshold: {normalized_strict}")
        
        # Without Tesseract (using only confusion matrix)
        normalized_matrix = normalize_text(
            sample_text,
            strategy=NormalizationStrategies.OCR_CONFIDENCE,
            use_tesseract=False  # Use only confusion matrix
        )
        logger.info(f"Using only confusion matrix: {normalized_matrix}")
        
    except ImportError:
        logger.error("OCR dependencies not installed. Install them using: pip install pytesseract pillow")
    except Exception as e:
        logger.error(f"Error: {e}")
    logger.info("")


def graph_based_strategy_example():
    """Demonstrate graph-based character network normalization."""
    sample_text = "Tһis іs а tеst with ѕome һomoglурhs."  # Contains various homoglyphs
    
    logger.info("GRAPH-BASED STRATEGY")
    logger.info("-" * 50)
    logger.info(f"Original text: {sample_text}")
    
    try:
        # Basic usage
        normalized_text = normalize_text(
            sample_text,
            strategy=NormalizationStrategies.GRAPH_BASED
        )
        logger.info(f"Default settings: {normalized_text}")
        
    except Exception as e:
        logger.error(f"Error: {e}")
    logger.info("")


def compare_strategies():
    """Compare all normalization strategies on the same text."""
    sample_text = "Tһis іs а tеst with ѕome һomoglурhs."  # Contains various homoglyphs
    ambiguous_text = "Тhе number 0 and letter О look similar."  # Cyrillic 'Т' and 'О'
    
    logger.info("STRATEGY COMPARISON")
    logger.info("-" * 50)
    logger.info(f"Original text 1: {sample_text}")
    logger.info(f"Original text 2 (ambiguous): {ambiguous_text}")
    logger.info("")
    
    strategies = [
        NormalizationStrategies.DOMINANT_SCRIPT,
        NormalizationStrategies.LOCAL_CONTEXT,
        NormalizationStrategies.TOKENIZATION,
        NormalizationStrategies.SPELL_CHECK,
        NormalizationStrategies.NGRAM,
        NormalizationStrategies.OCR_CONFIDENCE,
        NormalizationStrategies.GRAPH_BASED
    ]
    
    for strategy in strategies:
        try:
            result1 = normalize_text(sample_text, strategy=strategy)
            result2 = normalize_text(ambiguous_text, strategy=strategy)
            
            logger.info(f"{strategy.value}:")
            logger.info(f"  Text 1: {result1}")
            logger.info(f"  Text 2: {result2}")
            logger.info("")
            
        except Exception as e:
            logger.error(f"Error with {strategy.value}: {e}")


if __name__ == "__main__":
    logger.info("SilverSpeak Additional Normalization Strategies Examples")
    logger.info("=" * 70)
    logger.info("")
    
    # Individual strategy examples
    ngram_strategy_example()
    ocr_confidence_strategy_example()
    graph_based_strategy_example()
    
    # Compare all strategies
    compare_strategies()
