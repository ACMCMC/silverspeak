"""
Homoglyph normalization strategies package.

This package provides various strategies for normalizing text containing homoglyphs
(characters that look visually similar but have different Unicode code points).

Available strategies:
- Local context (uses surrounding character context)
- Dominant script (based on the predominant Unicode script in text)
- Dominant script and block (uses both script and Unicode block)
- Tokenizer-based (uses tokenization patterns)
- Language model (uses ML models for context-aware replacement)

Authors: Aldan Creo (ACMC) <os@acmc.fyi>
"""

# Import all strategy functions for backward compatibility
from .dominant_script import apply_dominant_script_and_block_strategy, apply_dominant_script_strategy
from .language_model import apply_language_model_strategy
from .local_context import apply_local_context_strategy
from .logging import VALID_LOG_LEVELS, configure_logging
from .tokenizer import apply_tokenizer_strategy

# Expose all the functions at the package level
__all__ = [
    "apply_dominant_script_strategy",
    "apply_dominant_script_and_block_strategy",
    "apply_language_model_strategy",
    "apply_local_context_strategy",
    "apply_tokenizer_strategy",
    "configure_logging",
    "VALID_LOG_LEVELS",
]
