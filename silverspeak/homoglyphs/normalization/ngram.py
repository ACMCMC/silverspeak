"""
N-gram frequency-based normalization strategy for homoglyphs.

This module provides functions to normalize text containing homoglyphs based on
character n-gram frequency analysis using NLTK. It identifies unlikely n-grams that might
indicate the presence of homoglyphs and normalizes them to standard characters.

Author: GitHub Copilot
"""

import re
import logging
import collections
from typing import Dict, List, Tuple, Set, Counter, Optional, Any, Union
from pathlib import Path
import os
import json
import string

logger = logging.getLogger(__name__)

# Create data directory for models if it doesn't exist
try:
    os.makedirs(os.path.join(os.path.dirname(__file__), "data"), exist_ok=True)
except Exception as e:
    logger.warning(f"Failed to create data directory: {e}")

# Try to import NLTK for n-gram models
try:
    import nltk
    from nltk.lm import NgramCounter, MLE
    from nltk.lm.preprocessing import padded_everygram_pipeline
    from nltk.util import everygrams
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False
    logger.warning(
        "NLTK not available, n-gram strategy will use a simplified approach. "
        "Install with: pip install nltk"
    )


class CharNgramAnalyzer:
    """Analyzer for character-level n-grams using NLTK or custom implementation."""
    
    def __init__(self, n_values: List[int] = None, language: str = "english"):
        """
        Initialize a character n-gram analyzer.
        
        Args:
            n_values: List of n-gram sizes to use (default: [2, 3, 4])
            language: Language to use for n-gram model (default: English)
        """
        self.n_values = n_values or [2, 3, 4]  # Use bigrams, trigrams, and 4-grams by default
        self.language = language
        self.models = {}
        self.use_nltk = NLTK_AVAILABLE
        self._ensure_nltk_data()
        
        # Initialize n-gram models
        self._initialize_models()
    
    def _ensure_nltk_data(self):
        """Ensure required NLTK data is available."""
        if not self.use_nltk:
            return
            
        try:
            from nltk.corpus import words
            words.words()  # This will raise LookupError if 'words' corpus isn't downloaded
        except LookupError:
            logger.info("Downloading required NLTK data...")
            try:
                nltk.download('words', quiet=True)
                nltk.download('punkt', quiet=True)
            except Exception as e:
                logger.error(f"Failed to download NLTK data: {e}")
                self.use_nltk = False
    
    def _initialize_models(self):
        """Initialize n-gram models for each n value."""
        if not self.use_nltk:
            # Use a simplified approach without NLTK
            self._initialize_simple_models()
            return
            
        logger.info("Initializing NLTK character n-gram models")
        
        from nltk.corpus import words, brown, gutenberg
        
        # Combine various corpora for a richer language model
        if self.language.lower() == "english":
            # Use a combination of words corpus and text corpora
            word_texts = [" ".join(words.words())]
            brown_texts = [" ".join(brown.words(fileid)) for fileid in brown.fileids()[:10]]  # Limit to 10 files
            gutenberg_texts = [" ".join(gutenberg.words(fileid)) for fileid in gutenberg.fileids()[:5]]  # Limit to 5 files
            
            all_texts = word_texts + brown_texts + gutenberg_texts
            training_text = " ".join(all_texts)
        else:
            # For non-English, use a simple default text
            training_text = (
                "The quick brown fox jumps over the lazy dog. "
                "A large fawn jumped quickly over white zinc boxes. "
                "All questions asked by five watched experts amaze the judge. "
                "The five boxing wizards jump quickly. "
                "How vexingly quick daft zebras jump! "
                "Sphinx of black quartz, judge my vow."
            )
        
        # Convert text to character sequences
        chars = list(training_text.lower())
        
        # Train models for each n value
        for n in self.n_values:
            # Create and train model
            train_data, vocab = padded_everygram_pipeline(n, [chars])
            model = MLE(n)
            model.fit(train_data, vocab)
            self.models[n] = model
            
            logger.debug(f"Trained {n}-gram model with NLTK")
    
    def _initialize_simple_models(self):
        """Initialize simplified n-gram models without NLTK."""
        logger.info("Initializing simplified character n-gram models (NLTK not available)")
        
        # Default training text for English
        training_text = (
            "The quick brown fox jumps over the lazy dog. "
            "A large fawn jumped quickly over white zinc boxes. "
            "All questions asked by five watched experts amaze the judge. "
            "The five boxing wizards jump quickly. "
            "How vexingly quick daft zebras jump! "
            "Sphinx of black quartz, judge my vow."
        )
        
        # Convert to lowercase
        chars = training_text.lower()
        
        # Create models for each n value
        for n in self.n_values:
            # Count n-grams
            ngrams = collections.Counter()
            padded_text = "<" * (n-1) + chars + ">" * (n-1)  # Pad with start/end markers
            
            for i in range(len(padded_text) - n + 1):
                ngram = padded_text[i:i+n]
                ngrams[ngram] += 1
                
            # Store the model as a counter with total count
            self.models[n] = {
                'counts': ngrams,
                'total': sum(ngrams.values())
            }
            
            logger.debug(f"Created simplified {n}-gram model with {len(ngrams)} unique n-grams")
    
    def score_text(self, text: str) -> List[float]:
        """
        Score each character in the text based on n-gram probability.
        
        Args:
            text: Text to score
            
        Returns:
            List of scores (0-1) for each character position
        """
        if not text:
            return []
        
        # Initialize scores for each character
        char_scores = [1.0] * len(text)
        
        if self.use_nltk:
            return self._score_with_nltk(text, char_scores)
        else:
            return self._score_with_simple_model(text, char_scores)
    
    def _score_with_nltk(self, text: str, char_scores: List[float]) -> List[float]:
        """Score text using NLTK models."""
        text = text.lower()
        
        # For each n-gram size
        for n in self.n_values:
            model = self.models[n]
            padded_text = "<" * (n-1) + text + ">" * (n-1)
            
            # Score each position
            for i in range(len(text)):
                # Extract context and target character
                context = []
                if i >= n-1:
                    context = list(padded_text[i-(n-1):i])
                else:
                    context = list(padded_text[:i])
                
                target = text[i]
                
                # Get probability from model
                try:
                    prob = model.score(target, context)
                    
                    # Convert log probability to a 0-1 scale and combine with current score
                    if prob > 0:  # This shouldn't happen but just in case
                        scaled_score = 0.01  # Very low score for unexpected high probability
                    else:
                        # Convert negative log probability to score between 0 and 1
                        # Closer to 1 means more likely
                        scaled_score = min(1.0, max(0.0, 1.0 + (prob / 10)))
                    
                    char_scores[i] *= scaled_score
                except:
                    # If there's any error, don't change the score
                    pass
        
        return char_scores
    
    def _score_with_simple_model(self, text: str, char_scores: List[float]) -> List[float]:
        """Score text using simplified models."""
        text = text.lower()
        
        # For each n-gram size
        for n in self.n_values:
            model = self.models[n]
            ngram_counts = model['counts']
            total_count = model['total']
            
            padded_text = "<" * (n-1) + text + ">" * (n-1)
            
            # Score each character using surrounding n-grams
            for i in range(len(text)):
                # Score based on n-grams containing this character
                relevant_scores = []
                
                # Check n-grams where this character appears at different positions
                for pos in range(n):
                    if i + pos < n-1:
                        continue
                    if i + pos - (n-1) >= len(text):
                        continue
                        
                    start = i + pos - (n-1)
                    ngram = padded_text[start:start+n]
                    
                    # Get normalized count as score
                    count = ngram_counts.get(ngram, 0)
                    score = count / (total_count or 1)  # Avoid division by zero
                    
                    # Apply Laplace smoothing for unseen n-grams
                    if count == 0:
                        score = 1 / (total_count + len(ngram_counts) + 1)
                    
                    relevant_scores.append(score)
                
                # Combine scores for this character
                if relevant_scores:
                    avg_score = sum(relevant_scores) / len(relevant_scores)
                    char_scores[i] *= avg_score
        
        return char_scores


def score_and_normalize_text(
    text: str,
    normalization_map: Dict[str, List[str]],
    analyzer: CharNgramAnalyzer,
    threshold: float = 0.01
) -> str:
    """
    Normalize text based on character n-gram scores.
    
    Args:
        text: Text to normalize
        normalization_map: Mapping from homoglyphs to standard characters
        analyzer: Character n-gram analyzer
        threshold: Probability threshold below which a character is considered suspect
        
    Returns:
        Normalized text with homoglyphs replaced
    """
    if not text:
        return text
    
    # Score each character
    char_scores = analyzer.score_text(text)
    
    # Create reverse mapping for fast lookups
    reverse_map = {}
    for homoglyph, std_chars in normalization_map.items():
        for std_char in std_chars:
            if std_char not in reverse_map:
                reverse_map[std_char] = set()
            reverse_map[std_char].add(homoglyph)
    
    # Normalize text based on scores
    result = list(text)
    
    for i, (char, score) in enumerate(zip(text, char_scores)):
        # Low score suggests this might be a homoglyph
        if score < threshold:
            # Check if this character could be a homoglyph of a standard character
            for std_char, homoglyphs in reverse_map.items():
                if char in homoglyphs:
                    logger.debug(f"Replacing '{char}' (score {score:.4f}) with '{std_char}' at position {i}")
                    result[i] = std_char
                    break
    
    return ''.join(result)


def apply_ngram_strategy(
    text: str,
    mapping: Dict[str, List[str]],
    language: str = "english",
    n_values: List[int] = None,
    threshold: float = 0.01,
    **kwargs,
) -> str:
    """
    Apply n-gram frequency-based normalization strategy to fix homoglyphs.
    
    Args:
        text: Text to normalize
        mapping: Homoglyph normalization map
        language: Language of the text (default: English)
        n_values: List of n-gram sizes to use (default: [2, 3, 4])
        threshold: Probability threshold below which a character is considered suspicious
        **kwargs: Additional arguments
        
    Returns:
        Normalized text with homoglyphs replaced
    """
    logger.info("Applying n-gram frequency normalization strategy")
    
    try:
        # Create n-gram analyzer
        analyzer = CharNgramAnalyzer(
            n_values=n_values,
            language=language
        )
        
        # Score and normalize text
        normalized_text = score_and_normalize_text(
            text=text,
            normalization_map=mapping,
            analyzer=analyzer,
            threshold=threshold
        )
        
        logger.info("N-gram normalization completed")
        return normalized_text
        
    except Exception as e:
        logger.error(f"Error in n-gram normalization: {e}")
        logger.warning("Returning original text due to error")
        return text
