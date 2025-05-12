"""
Spell checking-based normalization strategy for homoglyph replacement.

This module provides functionality to normalize text by using spell checkers
to identify and correct words containing potential homoglyphs. Unlike other
approaches, this directly leverages spelling correction algorithms which are
specifically designed to handle character-level errors.

The strategy supports multiple languages by utilizing language-specific
spell checkers when available.

Author: Aldan Creo (ACMC) <os@acmc.fyi>
"""

import logging
import re
from typing import Dict, List, Mapping, Optional, Set, Tuple, Union

logger = logging.getLogger(__name__)


def apply_spell_check_strategy(
    text: str,
    mapping: Mapping[str, List[str]],
    language: str = "en",
    distance_threshold: float = 0.8,
    min_word_length: int = 3,
    use_contextual: bool = True,
    custom_dictionary: Optional[List[str]] = None,
    **kwargs,
) -> str:
    """
    Normalize text using spell checking to identify and correct words containing homoglyphs.
    
    This strategy identifies words that might contain homoglyphs, then uses spell checking
    algorithms to suggest corrections. It supports multiple languages through appropriate
    spell checking libraries.

    Args:
        text (str): The input text to normalize.
        mapping (Mapping[str, List[str]]): A mapping from original characters to
            their possible homoglyph replacements.
        language (str): Language code (ISO 639-1) for language-specific spell checking.
            Defaults to 'en' for English. Supported languages depend on available dictionaries.
        distance_threshold (float): Threshold for accepting spelling corrections based on
            similarity to original word. Range 0.0-1.0, higher means stricter matching.
            Defaults to 0.8.
        min_word_length (int): Minimum word length to consider for spell checking.
            Shorter words are kept as is. Defaults to 3.
        use_contextual (bool): Whether to use contextual spell checking when available.
            This considers surrounding words when making corrections. Defaults to True.
        custom_dictionary (Optional[List[str]]): Custom word list to use for spell checking.
            If provided, takes precedence over language-specific dictionaries.
        **kwargs: Additional keyword arguments for spell checking configuration.

    Returns:
        str: The normalized text with homoglyphs replaced based on spell checking results.

    Raises:
        ImportError: If required dependencies are not installed.
        ValueError: If an unsupported language is specified.
        
    Note:
        This strategy works best for text with a consistent language and relatively
        few spelling errors apart from homoglyphs. Performance may vary based on
        the quality of available dictionaries for each language.
    """
    # Create reverse mapping to identify homoglyphs in text
    reverse_mapping = {}
    for char, homoglyphs in mapping.items():
        for homoglyph in homoglyphs:
            if homoglyph not in reverse_mapping:
                reverse_mapping[homoglyph] = []
            reverse_mapping[homoglyph].append(char)

    # Try to import spell checking libraries based on availability
    spell_checker = None
    contextual_spell_checker = None

    try:
        # First, try to import language-specific resources
        if language == "en":
            try:
                # Try symspellpy first (fast and effective)
                from symspellpy import SymSpell, Verbosity
                
                spell_checker = SymSpell(max_dictionary_edit_distance=2)
                dictionary_path = kwargs.get("dictionary_path", None)
                
                # Load dictionary
                if custom_dictionary:
                    # Create dictionary from custom word list
                    for word in custom_dictionary:
                        spell_checker.create_dictionary_entry(word, 1)
                elif dictionary_path:
                    spell_checker.load_dictionary(dictionary_path, term_index=0, count_index=1)
                else:
                    # Try to use default dictionary
                    import pkg_resources
                    dictionary_path = pkg_resources.resource_filename(
                        "symspellpy", "frequency_dictionary_en_82_765.txt"
                    )
                    spell_checker.load_dictionary(dictionary_path, term_index=0, count_index=1)
                
                logger.info("Using SymSpell for English spell checking")
                
            except ImportError:
                # Fall back to pyspellchecker
                try:
                    from spellchecker import SpellChecker
                    
                    spell_checker = SpellChecker(language=language, distance=2)
                    if custom_dictionary:
                        spell_checker.word_frequency.load_words(custom_dictionary)
                    
                    logger.info("Using pyspellchecker for spell checking")
                    
                except ImportError:
                    logger.warning(
                        "Neither symspellpy nor pyspellchecker found. "
                        "Install them using: poetry install --with spell-check"
                    )
        else:
            # For non-English languages, try pyspellchecker first
            try:
                from spellchecker import SpellChecker
                
                try:
                    spell_checker = SpellChecker(language=language, distance=2)
                    if custom_dictionary:
                        spell_checker.word_frequency.load_words(custom_dictionary)
                    logger.info(f"Using pyspellchecker for {language} spell checking")
                except ValueError:
                    logger.warning(f"Language '{language}' not supported by pyspellchecker")
                    
                    # Try to load custom dictionary if available
                    if custom_dictionary:
                        spell_checker = SpellChecker(language='en', distance=2)  # Use English as base
                        spell_checker.word_frequency.load_words(custom_dictionary)
                        logger.info(f"Using custom dictionary for {language} spell checking")
            except ImportError:
                logger.warning(
                    "pyspellchecker not found. Install it using: poetry install --with spell-check"
                )
        
        # Try to load contextual spell checker if requested
        if use_contextual:
            try:
                from contextlib import contextmanager
                import warnings
                
                # Suppress warnings during import
                @contextmanager
                def suppress_stdout_stderr():
                    import sys
                    try:
                        yield
                    finally:
                        pass
                
                with suppress_stdout_stderr():
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        # Try to import and initialize neuspell for contextual spell checking
                        from neuspell import BertChecker
                        
                        contextual_spell_checker = BertChecker()
                        contextual_spell_checker.from_pretrained()
                        logger.info("Using NeuSpell for contextual spell checking")
            except ImportError:
                logger.info(
                    "Contextual spell checker (neuspell) not available. "
                    "Install it using: poetry install --with contextual-spell-check"
                )
            except Exception as e:
                logger.warning(f"Error initializing contextual spell checker: {e}")
                contextual_spell_checker = None
                
    except Exception as e:
        logger.error(f"Error setting up spell checker: {e}")
        return text  # Return original text if setup fails

    # Function to check if a string might contain homoglyphs
    def contains_homoglyphs(word: str) -> bool:
        return any(char in mapping or char in reverse_mapping for char in word)

    # Function to normalize a single word using available spell checkers
    def normalize_word(word: str, context: Optional[str] = None) -> str:
        # Skip short words or words without potential homoglyphs
        if len(word) < min_word_length or not contains_homoglyphs(word):
            return word
            
        # Try contextual spell checking first if available and context is provided
        if contextual_spell_checker and context and use_contextual:
            try:
                corrected_context = contextual_spell_checker.correct_string(context)
                # Extract the corrected word from the context
                corrected_words = corrected_context.split()
                original_words = context.split()
                
                # Find the position of our word in the original context
                for i, orig_word in enumerate(original_words):
                    if orig_word == word and i < len(corrected_words):
                        return corrected_words[i]
            except Exception as e:
                logger.debug(f"Contextual spell checking failed: {e}")
        
        # Try regular spell checking
        if spell_checker:
            try:
                if isinstance(spell_checker, SymSpell):
                    # For SymSpell
                    suggestions = spell_checker.lookup(
                        word, 
                        Verbosity.CLOSEST, 
                        max_edit_distance=2,
                        include_unknown=True
                    )
                    
                    if suggestions:
                        suggestion = suggestions[0].term
                        # Check suggestion similarity
                        similarity = get_string_similarity(word, suggestion)
                        if similarity >= distance_threshold:
                            return suggestion
                else:
                    # For pyspellchecker
                    if word.lower() in spell_checker:
                        # Word exists in dictionary, but might have wrong case
                        return spell_checker.correction(word)
                    else:
                        # Get correction
                        correction = spell_checker.correction(word)
                        if correction:
                            # Check correction similarity
                            similarity = get_string_similarity(word, correction)
                            if similarity >= distance_threshold:
                                return correction
            except Exception as e:
                logger.debug(f"Spell checking failed for word '{word}': {e}")
        
        # If no good correction found or spell checking failed, return original word
        return word

    # Helper function to compute string similarity (simple Levenshtein ratio)
    def get_string_similarity(s1: str, s2: str) -> float:
        try:
            import Levenshtein
            return 1.0 - (Levenshtein.distance(s1, s2) / max(len(s1), len(s2)))
        except ImportError:
            # Fallback for simple similarity checking
            matches = sum(c1 == c2 for c1, c2 in zip(s1.lower(), s2.lower()))
            max_len = max(len(s1), len(s2))
            return matches / max_len if max_len > 0 else 1.0

    # Process the text
    if not spell_checker and not contextual_spell_checker:
        logger.warning("No spell checker available, returning original text")
        return text
        
    try:
        # Split text into words while preserving separators
        pattern = r'(\w+|\s+|[^\w\s]+)'
        tokens = re.findall(pattern, text)
        
        result = []
        i = 0
        while i < len(tokens):
            token = tokens[i]
            
            # Only process word tokens (skip whitespace and punctuation)
            if re.match(r'^\w+$', token):
                # Get some context for contextual spell checking
                context_size = 5  # Number of tokens before and after for context
                context_start = max(0, i - context_size)
                context_end = min(len(tokens), i + context_size + 1)
                context = ''.join(tokens[context_start:context_end])
                
                # Process the word
                normalized_token = normalize_word(token, context)
                result.append(normalized_token)
                
                # Log if correction was made
                if normalized_token != token:
                    logger.debug(f"Corrected '{token}' to '{normalized_token}'")
            else:
                # Keep non-word tokens as is
                result.append(token)
                
            i += 1
            
        return ''.join(result)
        
    except Exception as e:
        logger.error(f"Error during spell check normalization: {e}")
        return text  # Return original text if processing fails
