import logging
import unicodedata
from collections import Counter
from typing import Any, Dict, List, Mapping, Optional, Tuple, Union

import torch
import tqdm
import transformers
import unicodedataplus

from .script_block_category_utils import detect_dominant_block, detect_dominant_script

# Configure logging with a standardized format for production use
logger = logging.getLogger(__name__)

# Default valid log levels
VALID_LOG_LEVELS = {
    "DEBUG": logging.DEBUG,
    "INFO": logging.INFO,
    "WARNING": logging.WARNING,
    "ERROR": logging.ERROR,
    "CRITICAL": logging.CRITICAL,
}


def configure_logging(level: str = "INFO", format_string: Optional[str] = None) -> None:
    """
    Configure the logging system for the library.

    Args:
        level (str): Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        format_string (Optional[str]): Custom format string for log messages.
            If None, a default format will be used.

    Returns:
        None
    """
    if format_string is None:
        format_string = "[%(asctime)s] %(levelname)s [%(name)s.%(funcName)s:%(lineno)d] %(message)s"

    log_level = VALID_LOG_LEVELS.get(level.upper(), logging.INFO)

    # Configure the root logger
    logging.basicConfig(level=log_level, format=format_string, datefmt="%Y-%m-%d %H:%M:%S")

    # Set the level for this module's logger
    logger.setLevel(log_level)

    # Prevent logging propagation if needed
    # logger.propagate = False


def apply_local_context_strategy(
    text: str,
    normalization_map: Mapping[str, List[str]],
    N: int = 10,
) -> str:
    """
    Normalize text using local character context to choose optimal homoglyph replacements.

    This advanced normalization strategy analyzes the surrounding characters (context window)
    of each target character and selects replacement homoglyphs that best match the Unicode
    properties of the surrounding text. This preserves visual and semantic coherence by
    ensuring that replacement characters have similar properties to their context.

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
    if not text:
        logging.warning("Empty text provided for normalization")
        return ""

    if not normalization_map:
        logging.warning("Empty normalization map provided")
        return text

    # Dictionary of Unicode property extraction functions
    PROPERTY_FNS = {
        "script": unicodedataplus.script,
        "block": unicodedataplus.block,
        "category": unicodedataplus.category,
        "vertical_orientation": unicodedataplus.vertical_orientation,
        "bidirectional": unicodedata.bidirectional,
        "combining": unicodedata.combining,
        "east_asian_width": unicodedata.east_asian_width,
        "mirrored": unicodedata.mirrored,
    }

    # Process text character by character, analyzing context windows
    replaced_text = ""
    for i, char in enumerate(text):
        # Check if the character is in the mapping
        if char in normalization_map:
            # Get all possible homoglyph replacements including the original
            possible_chars = [char] + normalization_map[char]

            # Define the context window around the current character
            start = max(0, i - N // 2)
            end = min(len(text), i + N // 2 + 1)
            context_window = text[start:end]

            # Ensure we have a sufficiently sized window
            if len(context_window) < min(N, len(text)):
                if start == 0:
                    context_window = text[: min(N, len(text))]
                elif end == len(text):
                    context_window = text[-min(N, len(text)) :]

            # Extract Unicode properties from the context window
            try:
                properties = {prop: [PROPERTY_FNS[prop](c) for c in context_window] for prop in PROPERTY_FNS}
            except Exception as e:
                logging.error(f"Error extracting Unicode properties: {e}")
                replaced_text += char
                continue

            # Calculate property matching scores for each possible replacement
            scores = []
            for possible_char in possible_chars:
                try:
                    # Count how many property values the replacement character matches in the context
                    score = sum(
                        PROPERTY_FNS[prop](possible_char) == value
                        for prop, values in properties.items()
                        for value in values
                    )
                    scores.append((possible_char, score))
                except Exception as e:
                    logging.error(f"Error calculating score for '{possible_char}': {e}")
                    scores.append((possible_char, 0))  # Assign lowest score on error

            if not scores:
                replaced_text += char
                continue

            # Select the best-scoring replacement
            best_char, best_score = max(scores, key=lambda x: x[1])

            # Log a warning if multiple characters tie for the best score
            ties = [s[0] for s in scores if s[1] == best_score]
            if len(ties) > 1 and len(set(ties)) > 1:  # More than one unique character with best score
                if logging.getLogger().isEnabledFor(logging.DEBUG):
                    logging.debug(
                        f"Found a tie for the best character for '{char}' at index {i}. "
                        f"Options: {ties}. Using '{best_char}'."
                    )

            replaced_text += best_char
        else:
            # If the character is not in the mapping, keep it unchanged
            replaced_text += char

    return replaced_text


def apply_tokenizer_strategy(
    text: str,
    mapping: Mapping[str, List[str]],
    tokenizer_name: str = "bigscience/bloom",
    **kwargs,
) -> str:
    """
    Normalize text by choosing homoglyphs based on tokenizer vocabulary analysis.

    This strategy uses tokenizer-based heuristics to select homoglyph replacements
    that would result in more natural tokenization patterns. It analyzes how each
    potential homoglyph fits within the tokenizer's vocabulary and scores replacements
    based on multiple weighted criteria.

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
    # Default weights if not provided
    LONGEST_START_WEIGHT = kwargs.get("LONGEST_START_WEIGHT", 0.4)
    LONGEST_TOKEN_WEIGHT = kwargs.get("LONGEST_TOKEN_WEIGHT", 0.3)
    NUM_POSSIBLE_TOKENS_WEIGHT = kwargs.get("NUM_POSSIBLE_TOKENS_WEIGHT", 0.2) 
    NUM_TOKENS_CONTAINING_CHAR_WEIGHT = kwargs.get("NUM_TOKENS_CONTAINING_CHAR_WEIGHT", 0.1)
    
    try:
        from transformers import AutoTokenizer, PreTrainedTokenizer
    except ImportError:
        logging.error(
            "This normalization strategy requires the transformers library. Install it with: pip install transformers"
        )
        raise ImportError("Missing required dependency: transformers")

    if not text:
        logging.warning("Empty text provided for normalization")
        return ""

    if not mapping:
        logging.warning("Empty mapping provided for normalization")
        return text

    # Load tokenizer with error handling
    try:
        logging.info(f"Loading tokenizer: {tokenizer_name}")
        tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    except Exception as e:
        logging.error(f"Failed to load tokenizer '{tokenizer_name}': {e}")
        raise RuntimeError(f"Failed to load tokenizer: {e}")

    # Get vocabulary and sort by length (longest tokens first)
    vocab = list(tokenizer.get_vocab().keys())
    vocab = sorted(vocab, key=len, reverse=True)

    # Remove special prefix (varies by tokenizer type)
    if any(token.startswith("▁") for token in vocab):
        # BLOOM, GPT-2, and other BPE-based tokenizers use ▁
        vocab = [token[1:] if token.startswith("▁") else token for token in vocab]
    elif any(token.startswith("##") for token in vocab):
        # BERT-style tokenizers use ##
        vocab = [token[2:] if token.startswith("##") else token for token in vocab]

    normalized_text = ""

    # Process each character with a progress bar
    for i, char in enumerate(text):
        # Check if the character is in mapping
        if char in mapping:
            # Get possible homoglyphs including original character
            possible_chars = [char] + mapping[char]

            # Find tokens in vocabulary that contain each possible character
            possible_token_starts = {}
            for possible_char in possible_chars:
                try:
                    # Find tokens containing this character and extract the prefix
                    tokens_with_char = [
                        (
                            token[: token.rindex(possible_char)],  # Prefix up to the character
                            len(token),  # Total token length
                            token,  # Original token (for debugging)
                        )
                        for token in vocab
                        if possible_char in token
                    ]
                    possible_token_starts[possible_char] = tokens_with_char
                except Exception as e:
                    logging.debug(f"Error processing character '{possible_char}': {e}")
                    possible_token_starts[possible_char] = []

            # Filter to tokens whose prefixes match the end of our normalized text so far
            for char_key in list(possible_token_starts.keys()):
                possible_token_starts[char_key] = [
                    token_tuple
                    for token_tuple in possible_token_starts[char_key]
                    if normalized_text.endswith(token_tuple[0])
                ]

            # Remove characters with no valid token matches
            possible_token_starts = {char_key: tokens for char_key, tokens in possible_token_starts.items() if tokens}

            if not possible_token_starts:
                # No valid matches found, keep original character
                if logging.getLogger().isEnabledFor(logging.DEBUG):
                    logging.debug(f"No valid token matches for '{char}' at index {i}, keeping original")
                normalized_text += char
                continue

            # Calculate scores for each criterion
            individual_scores = {}
            for char_key, tokens in possible_token_starts.items():
                # Calculate individual metrics
                avg_start_length = sum(len(token[0]) for token in tokens) / len(tokens)
                avg_token_length = sum(token[1] for token in tokens) / len(tokens)
                num_possible_tokens = len(tokens)
                num_tokens_containing_char = len(tokens)

                individual_scores[char_key] = {
                    "avg_start_length": avg_start_length,
                    "avg_token_length": avg_token_length,
                    "num_possible_tokens_score": num_possible_tokens,
                    "num_tokens_containing_char": num_tokens_containing_char,
                }

            # Find max values for normalization (with safety against division by zero)
            max_vals = {
                "avg_start_length": max((s["avg_start_length"] for s in individual_scores.values()), default=1) or 1,
                "avg_token_length": max((s["avg_token_length"] for s in individual_scores.values()), default=1) or 1,
                "num_possible_tokens_score": max(
                    (s["num_possible_tokens_score"] for s in individual_scores.values()), default=1
                )
                or 1,
                "num_tokens_containing_char": max(
                    (s["num_tokens_containing_char"] for s in individual_scores.values()), default=1
                )
                or 1,
            }

            # Calculate normalized and weighted scores
            final_scores = {}
            for char_key, scores_dict in individual_scores.items():
                # Normalize scores to 0-1 range
                norm_scores = {
                    "longest_start": scores_dict["avg_start_length"] / max_vals["avg_start_length"],
                    "longest_token": scores_dict["avg_token_length"] / max_vals["avg_token_length"],
                    "num_tokens": scores_dict["num_possible_tokens_score"] / max_vals["num_possible_tokens_score"],
                    "token_contexts": scores_dict["num_tokens_containing_char"]
                    / max_vals["num_tokens_containing_char"],
                }

                # Calculate weighted score
                final_scores[char_key] = (
                    LONGEST_START_WEIGHT * norm_scores["longest_start"]
                    + LONGEST_TOKEN_WEIGHT * norm_scores["longest_token"]
                    + NUM_POSSIBLE_TOKENS_WEIGHT * norm_scores["num_tokens"]
                    + NUM_TOKENS_CONTAINING_CHAR_WEIGHT * norm_scores["token_contexts"]
                )

            # Select best character and append to result
            best_char = max(final_scores.keys(), key=lambda k: final_scores[k])
            normalized_text += best_char
        else:
            # Character not in mapping, keep as is
            normalized_text += char

    return normalized_text


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


def apply_dominant_script_strategy(replacer, text: str, **kwargs) -> str:
    """
    Normalize text based on the dominant Unicode script detected in the input.

    This function first identifies the dominant script in the text and then applies
    a normalization strategy using character mappings appropriate for that script.

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
    if not text:
        logging.warning("Empty text provided for normalization")
        return ""

    if not replacer:
        raise ValueError("No replacer provided for normalization")

    dominant_script = detect_dominant_script(text)

    if dominant_script == "Unknown":
        logging.warning("Unable to determine dominant script, normalization may be suboptimal")

    normalization_map = replacer.get_normalization_map_for_script_block_and_category(script=dominant_script, **kwargs)

    if not normalization_map:
        logging.warning(f"No normalization map available for script '{dominant_script}'")
        return text

    return text.translate(str.maketrans(normalization_map))


def apply_dominant_script_and_block_strategy(replacer, text: str, **kwargs) -> str:
    """
    Normalize text based on both the dominant Unicode script and block detected in the input.

    This function identifies both the dominant script and Unicode block in the text and then applies
    a normalization strategy using character mappings appropriate for that specific script-block combination.
    This is more precise than using just the script or just the block alone.

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
    if not text:
        logging.warning("Empty text provided for normalization")
        return ""

    if not replacer:
        raise ValueError("No replacer provided for normalization")

    dominant_script = detect_dominant_script(text)
    dominant_block = detect_dominant_block(text)

    if dominant_script == "Unknown" or dominant_block == "Unknown":
        logging.warning("Unable to determine dominant script/block, normalization may be suboptimal")

    normalization_map = replacer.get_normalization_map_for_script_block_and_category(
        script=dominant_script, block=dominant_block, **kwargs
    )

    if not normalization_map:
        logging.warning(f"No normalization map available for script '{dominant_script}' and block '{dominant_block}'")
        return text

    return text.translate(str.maketrans(normalization_map))
