"""
Language model-based normalization strategy for homoglyph replacement.

This module provides functionality to normalize text using a language model
to select the most contextually appropriate homoglyph replacements.

The strategy works by identifying words that contain potential homoglyphs,
masking these words, and then using a masked language model to predict the
normalized version of the text.

Author: Aldan Creo (ACMC) <os@acmc.fyi>
"""

import logging
import re
from collections import defaultdict
from typing import Dict, List, Mapping, Optional, Set, Tuple, Union

logger = logging.getLogger(__name__)

# Global flags for lazy loading
_TORCH_AVAILABLE = None
_TRANSFORMERS_AVAILABLE = None
_torch = None
_transformers = None

def _check_torch_availability():
    """Lazy check for PyTorch availability."""
    global _TORCH_AVAILABLE, _torch
    if _TORCH_AVAILABLE is None:
        try:
            import torch
            _torch = torch
            _TORCH_AVAILABLE = True
        except ImportError:
            _torch = None
            _TORCH_AVAILABLE = False
            logger.warning(
                "PyTorch not available, language model strategy will not work. "
                "Install with: pip install torch"
            )
    return _TORCH_AVAILABLE

def _check_transformers_availability():
    """Lazy check for Transformers availability."""
    global _TRANSFORMERS_AVAILABLE, _transformers
    if _TRANSFORMERS_AVAILABLE is None:
        try:
            import transformers
            _transformers = transformers
            _TRANSFORMERS_AVAILABLE = True
        except ImportError:
            _transformers = None
            _TRANSFORMERS_AVAILABLE = False
            logger.warning(
                "Transformers not available, language model strategy will not work. "
                "Install with: pip install transformers"
            )
    return _TRANSFORMERS_AVAILABLE

def _get_torch():
    """Get PyTorch module, loading it lazily if needed."""
    _check_torch_availability()
    return _torch

def _get_transformers():
    """Get Transformers module, loading it lazily if needed."""
    _check_transformers_availability()
    return _transformers


def apply_language_model_strategy(
    text: str,
    mapping: Mapping[str, List[str]],
    language_model=None,  # Optional[transformers.PreTrainedModel] 
    tokenizer=None,  # Optional[transformers.PreTrainedTokenizer]
    model_name: str = "bert-base-multilingual-cased",
    batch_size: int = 8,
    max_length: int = 512,
    device: Optional[str] = None,
    word_level: bool = True,
    min_confidence: float = 0.7,
    **kwargs,
) -> str:
    """
    Normalize text using a language model to select the most contextually appropriate homoglyph replacements.

    This advanced strategy uses a masked language model to predict the most likely words or characters
    where homoglyph replacements are possible. It produces text that is semantically coherent according
    to the language model's understanding of natural language.

    The strategy can operate at two levels:
    1. Word-level (default): Identifies words containing homoglyphs, masks them entirely, and predicts
       the most likely replacement words.
    2. Character-level: Masks individual homoglyph characters and predicts replacements one by one.

    Args:
        text (str): The input text to normalize.
        mapping (Mapping[str, List[str]]): A mapping from original characters to
            their possible homoglyph replacements.
        language_model (Optional[transformers.PreTrainedModel]): A pre-loaded masked language model.
            If not provided, one will be loaded using the model_name parameter.
        tokenizer (Optional[transformers.PreTrainedTokenizer]): A pre-loaded tokenizer matching the model.
            If not provided, one will be loaded using the model_name parameter.
        model_name (str): The HuggingFace model name to load if language_model and tokenizer
            are not provided. Defaults to "bert-base-multilingual-cased".
        batch_size (int): Number of text segments to process in each batch. Defaults to 8.
        max_length (int): Maximum length of text segments. Longer text will be split.
            Defaults to 512.
        device (Optional[str]): Device to run the model on ('cuda', 'cpu', etc.).
            Defaults to cuda if available, otherwise cpu.
        word_level (bool): If True, performs word-level masking and prediction rather than
            character-by-character. This better handles multi-character token issues. Defaults to True.
        min_confidence (float): Minimum confidence threshold for accepting model predictions.
            Predictions below this threshold will fall back to other heuristics. Range 0.0-1.0.
            Defaults to 0.7.
        **kwargs: Additional keyword arguments passed to the model.

    Returns:
        str: The normalized text with homoglyphs selected based on language model predictions.

    Raises:
        ImportError: If required dependencies are not installed.
        RuntimeError: If model loading fails or other runtime errors occur.

    Note:
        This strategy is computationally intensive but produces high-quality results that
        maintain the semantic coherence of the original text. The word-level approach is
        particularly effective for handling homoglyphs that affect tokenization.
    """
    # Check for required dependencies
    if not _check_torch_availability():
        logging.error(
            "This normalization strategy requires PyTorch. "
            "Install it with: pip install torch"
        )
        raise ImportError("Missing required dependency: torch")
    
    if not _check_transformers_availability():
        logging.error(
            "This normalization strategy requires the transformers library. "
            "Install it with: pip install transformers"
        )
        raise ImportError("Missing required dependency: transformers")
    
    # Get the modules
    torch = _get_torch()
    transformers = _get_transformers()
    
    # Apply torch inference mode decorator
    with torch.inference_mode():
        return _apply_language_model_strategy_impl(
            text, mapping, language_model, tokenizer, model_name,
            batch_size, max_length, device, word_level, min_confidence, **kwargs
        )

def _apply_language_model_strategy_impl(
    text: str,
    mapping: Mapping[str, List[str]],
    language_model,
    tokenizer,
    model_name: str,
    batch_size: int,
    max_length: int,
    device: Optional[str],
    word_level: bool,
    min_confidence: float,
    **kwargs,
) -> str:
    """Implementation of language model strategy."""
    torch = _get_torch()
    transformers = _get_transformers()
    
    if not text:
        logging.warning("Empty text provided for normalization")
        return ""

    if not mapping:
        logging.warning("Empty mapping provided for normalization")
        return text

    # Import transformers classes
    from transformers import AutoModelForMaskedLM, AutoTokenizer
    
    # Set default device if not specified
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    logging.info(f"Using device: {device}")

    # Load model and tokenizer if not provided
    if language_model is None or tokenizer is None:
        try:
            logging.info(f"Loading model and tokenizer: {model_name}")
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            language_model = AutoModelForMaskedLM.from_pretrained(model_name)

            # At this point language_model is guaranteed to be not None if we reach here
            assert language_model is not None, "Model loading failed but did not raise an exception"

            # Now we can safely use the methods - add type assertions for mypy
            assert hasattr(language_model, "to"), "Model does not have 'to' method"
            assert hasattr(language_model, "eval"), "Model does not have 'eval' method"
            language_model.to(device)
            language_model.eval()  # Set to evaluation mode
        except Exception as e:
            logging.error(f"Failed to load model or tokenizer '{model_name}': {e}")
            raise RuntimeError(f"Failed to load model or tokenizer: {e}")
    # Handle the case where language_model was provided but is None (should not happen given the condition above)
    elif language_model is None:
        raise RuntimeError("Language model is None and could not be loaded")
    # Handle the case where language_model was provided and is not None
    else:
        # Here language_model is guaranteed to be not None by the elif above
        assert hasattr(language_model, "to"), "Model does not have 'to' method"
        assert hasattr(language_model, "eval"), "Model does not have 'eval' method"
        language_model.to(device)
        language_model.eval()

    # Ensure tokenizer is not None after loading attempt
    if tokenizer is None:
        raise RuntimeError("Tokenizer is None after loading attempt")

    # Get mask token and ID
    mask_token = tokenizer.mask_token
    mask_token_id = tokenizer.mask_token_id

    if mask_token is None or mask_token_id is None:
        logging.error(f"Model {model_name} does not support masked language modeling")
        raise RuntimeError(f"Model {model_name} does not support masked language modeling")
        
    # Create a reverse mapping from homoglyphs to original characters
    # This helps us identify potential homoglyphs in the text
    reverse_mapping: Dict[str, List[str]] = defaultdict(list)
    for orig_char, homoglyphs in mapping.items():
        for homoglyph in homoglyphs:
            reverse_mapping[homoglyph].append(orig_char)
            
    # Also add original characters themselves to the reverse mapping
    for orig_char in mapping.keys():
        if orig_char not in reverse_mapping:
            reverse_mapping[orig_char] = []

    # Function to identify words that might contain homoglyphs
    def find_homoglyph_words(text_segment: str) -> List[Tuple[int, int, str]]:
        """Identify words in the text that contain potential homoglyphs.
        
        Returns a list of (start_pos, end_pos, word) tuples.
        """
        # Simple word boundary regex - can be improved for languages with different word separations
        words = list(re.finditer(r'\b\w+\b', text_segment))
        result = []
        
        for match in words:
            start, end = match.span()
            word = match.group()
            
            # Check if this word contains any homoglyphs
            contains_homoglyph = False
            for i, char in enumerate(word):
                if char in mapping or char in reverse_mapping:
                    contains_homoglyph = True
                    break
                    
            if contains_homoglyph:
                result.append((start, end, word))
                
        return result
    
    # Split long text into manageable segments
    normalized_text_segments = []

    # Process text in segments to handle long documents
    for i in range(0, len(text), max_length):
        segment = text[i : i + max_length]
        
        if word_level:
            # Word-level approach - mask entire words containing homoglyphs
            normalized_segment = segment
            homoglyph_words = find_homoglyph_words(segment)
            
            if not homoglyph_words:
                # No words with homoglyphs in this segment
                normalized_text_segments.append(segment)
                continue
            
            # Process words in batches
            for batch_start in range(0, len(homoglyph_words), batch_size):
                batch_words = homoglyph_words[batch_start : batch_start + batch_size]
                
                # Create masked versions for this batch, keeping track of original positions
                masked_segments = []
                for start_pos, end_pos, word in batch_words:
                    # Create a masked version of the segment
                    chars = list(normalized_segment)
                    # Replace the word with mask tokens (one per character to maintain length)
                    mask_length = end_pos - start_pos
                    chars[start_pos:end_pos] = [mask_token] * mask_length
                    masked_segments.append("".join(chars))
                
                # Tokenize and encode
                inputs = tokenizer(masked_segments, return_tensors="pt", padding=True, truncation=True)
                inputs = {k: v.to(device) for k, v in inputs.items()}
                
                # Run model inference
                with torch.no_grad():
                    from typing import cast, Callable
                    model_callable = cast(Callable, language_model)
                    outputs = model_callable(**inputs)
                
                # Process each masked word in the batch
                for batch_idx, (start_pos, end_pos, original_word) in enumerate(batch_words):
                    # Find mask token indices
                    token_ids = inputs.input_ids[batch_idx].tolist()
                    encoded_input = tokenizer.convert_ids_to_tokens(token_ids)
                    
                    # Find indices of all mask tokens in the tokenized sequence
                    mask_token_indices = [i for i, token_id in enumerate(token_ids) if token_id == mask_token_id]
                    
                    if not mask_token_indices:
                        logging.warning(f"Could not find mask token for word '{original_word}'")
                        continue
                    
                    # Predict tokens for all mask positions
                    top_token_ids = []
                    for mask_idx in mask_token_indices:
                        logits = outputs.logits[batch_idx, mask_idx]
                        probs = torch.softmax(logits, dim=0)
                        
                        # Get top 10 tokens
                        top_values, top_indices = torch.topk(probs, k=10)
                        top_token_ids.append([(idx.item(), val.item()) for idx, val in zip(top_indices, top_values)])
                    
                    # Try to reconstruct probable words from predicted tokens
                    candidate_words = []
                    confidence_scores = []
                    
                    # For simplicity, just use the first mask token's predictions
                    # This can be improved to consider all mask tokens for more robust reconstruction
                    for token_id, confidence in top_token_ids[0]:
                        token = tokenizer.convert_ids_to_tokens(token_id)
                        # Skip special tokens and very short tokens
                        if tokenizer.special_tokens_map and token in tokenizer.special_tokens_map.values():
                            continue
                        if len(token) < 2 and len(original_word) > 2:
                            continue
                            
                        # Clean up token (remove special prefixes)
                        if token.startswith("##"):
                            token = token[2:]
                        elif token.startswith("▁"):
                            token = token[1:]
                            
                        # Add as candidate if similar length to original word
                        if 0.5 <= len(token)/len(original_word) <= 2.0:
                            candidate_words.append(token)
                            confidence_scores.append(confidence)
                    
                    # If we have candidates, choose the highest confidence one that meets our threshold
                    if candidate_words and max(confidence_scores) >= min_confidence:
                        best_idx = confidence_scores.index(max(confidence_scores))
                        best_word = candidate_words[best_idx]
                        best_conf = confidence_scores[best_idx]
                        
                        # If predicted word is long enough, use it as replacement
                        if len(best_word) >= min(2, len(original_word)):
                            # Ensure replacement isn't longer than the original word
                            best_word = best_word[:len(original_word)]
                            # Pad if necessary
                            if len(best_word) < len(original_word):
                                best_word = best_word + original_word[len(best_word):]
                                
                            # Apply the replacement
                            chars = list(normalized_segment)
                            chars[start_pos:start_pos+len(best_word)] = list(best_word)
                            normalized_segment = "".join(chars)
                            
                            logging.info(f"Replaced '{original_word}' with '{best_word}' (confidence: {best_conf:.4f})")
            
            normalized_text_segments.append(normalized_segment)
        
        else:
            # Character-level approach (original logic)
            # Find positions of characters that have homoglyph alternatives
            positions_to_mask = [(pos, char) for pos, char in enumerate(segment) 
                                if char in mapping or char in reverse_mapping]

            if not positions_to_mask:
                # No characters to replace in this segment
                normalized_text_segments.append(segment)
                continue

            # Process in batches for character positions
            normalized_segment = list(segment)  # Convert to list for character-by-character replacement

            for batch_start in range(0, len(positions_to_mask), batch_size):
                batch_positions = positions_to_mask[batch_start : batch_start + batch_size]

                # Create masked versions for this batch
                masked_inputs = []
                for pos, char in batch_positions:
                    masked_segment = list(segment)  # Create a new copy for each position
                    masked_segment[pos] = mask_token
                    masked_inputs.append("".join(masked_segment))

                # Tokenize and encode
                inputs = tokenizer(masked_inputs, return_tensors="pt", padding=True, truncation=True)
                inputs = {k: v.to(device) for k, v in inputs.items()}

                # Run model inference
                with torch.no_grad():
                    from typing import cast, Callable
                    model_callable = cast(Callable, language_model)
                    outputs = model_callable(**inputs)

                # Process each position in the batch
                for batch_idx, (pos, original_char) in enumerate(batch_positions):
                    # Get the token index of the masked position
                    token_ids = inputs.input_ids[batch_idx].tolist()
                    try:
                        mask_token_index = token_ids.index(mask_token_id)
                    except ValueError:
                        logging.warning(f"Could not find mask token for position {pos}")
                        continue

                    # Get predictions for the masked position
                    logits = outputs.logits[batch_idx, mask_token_index]
                    
                    # Determine possible replacements
                    possible_chars = []
                    if original_char in mapping:
                        possible_chars = [original_char] + mapping[original_char]
                    elif original_char in reverse_mapping:
                        possible_chars = [original_char] + reverse_mapping[original_char]
                    else:
                        possible_chars = [original_char]
                        
                    possible_token_ids = []

                    # Map characters to token IDs
                    for char in possible_chars:
                        char_tokens = tokenizer.encode(char, add_special_tokens=False)
                        if len(char_tokens) == 1:  # Only consider single-token characters
                            possible_token_ids.append((char, char_tokens[0]))

                    if not possible_token_ids:
                        # Fallback to original if no single-token mappings found
                        normalized_segment[pos] = original_char
                        continue

                    # Get probability scores for possible replacements
                    scores = {}
                    for char, token_id in possible_token_ids:
                        scores[char] = logits[token_id].item()

                    # Select the highest scoring character
                    best_char = max(scores.keys(), key=lambda k: scores[k])
                    normalized_segment[pos] = best_char

            normalized_text_segments.append("".join(normalized_segment))

    # Join all processed segments
    return "".join(normalized_text_segments)
