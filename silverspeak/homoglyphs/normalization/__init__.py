"""
Homoglyph normalization strategies package.

This package provides various strategies for normalizing text containing homoglyphs
(characters that look visually similar but have different Unicode code points).

Available strategies:
- Local context (uses surrounding character context)
- Dominant script (based on the predominant Unicode script in text)
- Dominant script and block (uses both script and Unicode block)
- Tokenizer-based (uses tokenization patterns)
- Language model (uses masked language models for context-aware replacement)
- LLM prompt (uses generative language models with prompts to fix homoglyphs)
- Spell check (uses spelling correction algorithms with multilingual support)
- N-gram (uses character n-gram frequency analysis)
- OCR confidence (uses OCR confidence scores or confusion matrices)
- Graph-based (uses graph-based character similarity networks)

Authors: Aldan Creo (ACMC) <os@acmc.fyi>
"""

# Import all strategy functions for backward compatibility
from .dominant_script import apply_dominant_script_and_block_strategy, apply_dominant_script_strategy
from .language_model import apply_language_model_strategy
from .llm_prompt import apply_llm_prompt_strategy
from .local_context import apply_local_context_strategy
from .logging import configure_logging, VALID_LOG_LEVELS
from .spell_check import apply_spell_check_strategy
from .tokenizer import apply_tokenizer_strategy
from .ngram import apply_ngram_strategy
from .ocr_confidence import apply_ocr_confidence_strategy
from .graph_based import apply_graph_strategy

# Expose all the functions at the package level
__all__ = [
    "apply_dominant_script_strategy",
    "apply_dominant_script_and_block_strategy",
    "apply_language_model_strategy",
    "apply_llm_prompt_strategy",
    "apply_local_context_strategy",
    "apply_spell_check_strategy",
    "apply_tokenizer_strategy",
    "apply_ngram_strategy",
    "apply_ocr_confidence_strategy",
    "apply_graph_strategy",
    "configure_logging",
    "VALID_LOG_LEVELS",
]
