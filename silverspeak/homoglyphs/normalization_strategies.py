# filepath: /mnt/beegfs/home/aldan.creo/silverspeak/silverspeak/homoglyphs/normalization_strategies.py
"""
DEPRECATED: Legacy module for homoglyph normalization strategies.

This module is maintained for backward compatibility.
Use the functions in the silverspeak.homoglyphs.normalization package instead.

Author: Aldan Creo (ACMC) <os@acmc.fyi>
"""

import logging
import warnings
from typing import List, Mapping, Optional

# Import from the new location for backward compatibility
from .normalization import (
    apply_dominant_script_and_block_strategy as apply_dominant_script_and_block_strategy_impl,
    apply_dominant_script_strategy as apply_dominant_script_strategy_impl,
    apply_language_model_strategy as apply_language_model_strategy_impl,
    apply_local_context_strategy as apply_local_context_strategy_impl,
    apply_tokenizer_strategy as apply_tokenizer_strategy_impl,
    configure_logging as configure_logging_impl,
    VALID_LOG_LEVELS,
)

# We need these for backward compatibility with existing code
from .script_block_category_utils import detect_dominant_block, detect_dominant_script

# Configure logging with a standardized format for production use
logger = logging.getLogger(__name__)

# Show deprecation warning
warnings.warn(
    "The normalization_strategies module is deprecated. "
    "Use the functions in silverspeak.homoglyphs.normalization instead.",
    DeprecationWarning,
    stacklevel=2,
)


def configure_logging(level: str = "INFO", format_string: Optional[str] = None) -> None:
    """
    Configure the logging system for the library.
    
    This is a shim that forwards to the implementation in normalization.logging.

    Args:
        level (str): Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        format_string (Optional[str]): Custom format string for log messages.
            If None, a default format will be used.

    Returns:
        None
    """
    return configure_logging_impl(level=level, format_string=format_string)


def apply_local_context_strategy(
    text: str,
    normalization_map: Mapping[str, List[str]],
    N: int = 10,
) -> str:
    """
    Normalize text using local character context to choose optimal homoglyph replacements.
    
    This is a shim that forwards to the implementation in normalization.local_context.

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
    return apply_local_context_strategy_impl(text=text, normalization_map=normalization_map, N=N)


def apply_tokenizer_strategy(
    text: str,
    mapping: Mapping[str, List[str]],
    tokenizer_name: str = "bigscience/bloom",
    **kwargs,
) -> str:
    """
    Normalize text by choosing homoglyphs based on tokenizer vocabulary analysis.
    
    This is a shim that forwards to the implementation in normalization.tokenizer.

    Args:
        text (str): The input text to normalize.
        mapping (Mapping[str, List[str]]): A mapping from original characters to
            their possible homoglyph replacements.
        tokenizer_name (str, optional): The name/path of the HuggingFace tokenizer to use.
            Defaults to "bigscience/bloom".
        **kwargs: Additional keyword arguments including:
            - LONGEST_START_WEIGHT (float, optional): Weight for prioritizing tokens with
              longer starting sequences. Defaults to 0.4.
            - LONGEST_TOKEN_WEIGHT (float, optional): Weight for prioritizing longer tokens.
              Defaults to 0.3.
            - NUM_POSSIBLE_TOKENS_WEIGHT (float, optional): Weight for prioritizing characters
              that appear in more tokens. Defaults to 0.2.
            - NUM_TOKENS_CONTAINING_CHAR_WEIGHT (float, optional): Weight for prioritizing
              characters that appear in more possible token contexts. Defaults to 0.1.

    Returns:
        str: The normalized text optimized for the specified tokenizer.

    Raises:
        ImportError: If the transformers library is not installed.
        RuntimeError: If tokenizer loading fails.
    """
    return apply_tokenizer_strategy_impl(
        text=text, 
        mapping=mapping, 
        tokenizer_name=tokenizer_name, 
        **kwargs
    )


def apply_language_model_strategy(
    text: str,
    mapping: Mapping[str, List[str]],
    language_model=None,  # Optional[transformers.PreTrainedModel]
    tokenizer=None,       # Optional[transformers.PreTrainedTokenizer]
    model_name: str = "bert-base-multilingual-cased",
    batch_size: int = 8,
    max_length: int = 512,
    device: Optional[str] = None,
    **kwargs,
) -> str:
    """
    Normalize text using a language model to select the most contextually appropriate homoglyph replacements.
    
    This is a shim that forwards to the implementation in normalization.language_model.

    Args:
        text (str): The input text to normalize.
        mapping (Mapping[str, List[str]]): A mapping from original characters to
            their possible homoglyph replacements.
        language_model: A pre-loaded masked language model.
            If not provided, one will be loaded using the model_name parameter.
        tokenizer: A pre-loaded tokenizer matching the model.
            If not provided, one will be loaded using the model_name parameter.
        model_name (str): The HuggingFace model name to load if language_model and tokenizer
            are not provided. Defaults to "bert-base-multilingual-cased".
        batch_size (int): Number of text segments to process in each batch. Defaults to 8.
        max_length (int): Maximum length of text segments. Longer text will be split.
            Defaults to 512.
        device (Optional[str]): Device to run the model on ('cuda', 'cpu', etc.).
            Defaults to cuda if available, otherwise cpu.
        **kwargs: Additional keyword arguments passed to the model.

    Returns:
        str: The normalized text with homoglyphs selected based on language model predictions.

    Raises:
        ImportError: If required dependencies are not installed.
        RuntimeError: If model loading fails or other runtime errors occur.

    Note:
        This strategy is computationally intensive but produces high-quality results that
        maintain the semantic coherence of the original text.
    """
    return apply_language_model_strategy_impl(
        text=text,
        mapping=mapping,
        language_model=language_model,
        tokenizer=tokenizer,
        model_name=model_name,
        batch_size=batch_size,
        max_length=max_length,
        device=device,
        **kwargs
    )


def apply_dominant_script_strategy(replacer, text: str, **kwargs) -> str:
    """
    Normalize text based on the dominant Unicode script detected in the input.
    
    This is a shim that forwards to the implementation in normalization.dominant_script.

    Args:
        replacer: Instance of HomoglyphReplacer that provides normalization mappings.
        text (str): The input text to normalize.
        **kwargs: Additional keyword arguments to pass to the normalization map generator.
            Commonly used kwargs include:
            - category (str): Unicode category to filter by.
            - preserve_case (bool): Whether to preserve character case during normalization.

    Returns:
        str: The normalized text with homoglyphs replaced according to the dominant script.

    Raises:
        ValueError: If the text is empty or the replacer is not properly initialized.

    Note:
        This strategy is most effective for texts predominantly written in a single script.
    """
    return apply_dominant_script_strategy_impl(replacer=replacer, text=text, **kwargs)


def apply_dominant_script_and_block_strategy(replacer, text: str, **kwargs) -> str:
    """
    Normalize text based on both the dominant Unicode script and block detected in the input.
    
    This is a shim that forwards to the implementation in normalization.dominant_script.

    Args:
        replacer: Instance of HomoglyphReplacer that provides normalization mappings.
        text (str): The input text to normalize.
        **kwargs: Additional keyword arguments to pass to the normalization map generator.
            Commonly used kwargs include:
            - category (str): Unicode category to filter by.
            - preserve_case (bool): Whether to preserve character case during normalization.

    Returns:
        str: The normalized text with homoglyphs replaced according to the dominant script and block.

    Raises:
        ValueError: If the text is empty or the replacer is not properly initialized.

    Note:
        This strategy is more specific than just using script detection alone and may provide
        better normalization for mixed-script texts where specific blocks are important.
    """
    return apply_dominant_script_and_block_strategy_impl(replacer=replacer, text=text, **kwargs)
