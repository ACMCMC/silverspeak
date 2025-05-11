"""
Language model-based normalization strategy for homoglyph replacement.

This module provides functionality to normalize text using a language model
to select the most contextually appropriate homoglyph replacements.

Author: Aldan Creo (ACMC) <os@acmc.fyi>
"""

import logging
from typing import List, Mapping, Optional
import transformers

import torch

logger = logging.getLogger(__name__)


@torch.inference_mode()
def apply_language_model_strategy(
    text: str,
    mapping: Mapping[str, List[str]],
    language_model: Optional[transformers.PreTrainedModel] = None,
    tokenizer: Optional[transformers.PreTrainedTokenizer] = None,
    model_name: str = "bert-base-multilingual-cased",
    batch_size: int = 8,
    max_length: int = 512,
    device: Optional[str] = None,
    **kwargs,
) -> str:
    """
    Normalize text using a language model to select the most contextually appropriate homoglyph replacements.

    This advanced strategy uses a masked language model to predict the most likely character at each position
    where a homoglyph replacement is possible. It produces text that is semantically coherent according to the
    language model's understanding of natural language.

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
    try:
        import torch
        from transformers import AutoModelForMaskedLM, AutoTokenizer
    except ImportError:
        logging.error(
            "This normalization strategy requires the transformers and torch libraries. "
            "Install them with: pip install transformers torch"
        )
        raise ImportError("Missing required dependencies: transformers, torch")

    if not text:
        logging.warning("Empty text provided for normalization")
        return ""

    if not mapping:
        logging.warning("Empty mapping provided for normalization")
        return text

    # Set default device if not specified
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

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

    # Split long text into manageable segments
    normalized_text_segments = []

    # Process text in segments to handle long documents
    for i in range(0, len(text), max_length):
        segment = text[i : i + max_length]

        # Find positions of characters that have homoglyph alternatives
        positions_to_mask = [(pos, char) for pos, char in enumerate(segment) if char in mapping]

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
                # Add a runtime type check to satisfy mypy
                from typing import cast, Callable
                model_callable = cast(Callable, language_model)
                outputs = model_callable(**inputs)

            # Process each position in the batch
            for batch_idx, (pos, original_char) in enumerate(batch_positions):
                # Get the token index of the masked position
                # This is more complex than it seems due to tokenization
                token_ids = inputs.input_ids[batch_idx].tolist()
                mask_token_index = token_ids.index(mask_token_id)

                # Get predictions for the masked position
                logits = outputs.logits[batch_idx, mask_token_index]

                # Filter to only consider the original character and its homoglyphs
                possible_chars = [original_char] + mapping[original_char]
                possible_token_ids = []

                # Map characters to token IDs (may be multi-token)
                for char in possible_chars:
                    char_tokens = tokenizer.encode(char, add_special_tokens=False)
                    if len(char_tokens) == 1:  # Only consider single-token characters for simplicity
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
