"""
Utility functions for SilverSpeak.

This module provides common utility functions used throughout the SilverSpeak library,
primarily for text processing, token alignment, and language model operations.
It includes functions for:
- Text encoding and decoding
- Token manipulation and conversion
- Sequence alignment algorithms
- Language model integrations
- Loglikelihood calculations

Author: Aldan Creo (ACMC) <os@acmc.fyi>
Version: 1.0.0
License: See LICENSE file in the project root
"""

import logging
import math
import random
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
from torch import Tensor
from torch.nn import CrossEntropyLoss
from transformers import AutoModelForCausalLM, AutoTokenizer

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

# Default language model configuration
DEFAULT_MODEL = "bigscience/bloom-560m"

# Initialize tokenizer and model lazily to prevent unnecessary loading
_tokenizer = None
_model = None


def get_tokenizer(model_name: str = DEFAULT_MODEL):
    """
    Get or initialize the tokenizer.

    Args:
        model_name (str): Name of the model to use for tokenization.
            Defaults to the predefined DEFAULT_MODEL.

    Returns:
        The tokenizer instance.
    """
    global _tokenizer
    if _tokenizer is None:
        logger.info(f"Initializing tokenizer: {model_name}")
        try:
            _tokenizer = AutoTokenizer.from_pretrained(model_name)
        except Exception as e:
            logger.error(f"Failed to initialize tokenizer: {e}")
            raise RuntimeError(f"Failed to initialize tokenizer: {e}")
    return _tokenizer


def get_model(model_name: str = DEFAULT_MODEL):
    """
    Get or initialize the language model.

    Args:
        model_name (str): Name of the model to use.
            Defaults to the predefined DEFAULT_MODEL.

    Returns:
        The model instance.
    """
    global _model
    if _model is None:
        logger.info(f"Initializing model: {model_name}")
        try:
            _model = AutoModelForCausalLM.from_pretrained(model_name)
            if torch.cuda.is_available():
                _model.cuda()
                logger.info("Using CUDA for model acceleration")
            else:
                logger.info("CUDA not available, using CPU for model")
        except Exception as e:
            logger.error(f"Failed to initialize model: {e}")
            raise RuntimeError(f"Failed to initialize model: {e}")
    return _model


def encode_text(text: str, model_name: str = DEFAULT_MODEL) -> torch.Tensor:
    """
    Encode text using the chosen tokenizer.

    Args:
        text (str): The text to encode.
        model_name (str, optional): The model name to use for tokenization.

    Returns:
        torch.Tensor: The encoded text as token IDs.

    Raises:
        ValueError: If the input text is empty.
    """
    if not text:
        raise ValueError("Cannot encode empty text")

    tokenizer = get_tokenizer(model_name)
    try:
        input_ids = tokenizer.encode(text, return_tensors="pt")[0]
        return input_ids
    except Exception as e:
        logger.error(f"Error encoding text: {e}")
        raise


def decode_tokens(tokens: torch.Tensor, model_name: str = DEFAULT_MODEL) -> str:
    """
    Decode tokens using the chosen tokenizer.

    Args:
        tokens (torch.Tensor): The tokens to decode.
        model_name (str, optional): The model name to use for tokenization.

    Returns:
        str: The decoded text.

    Raises:
        ValueError: If the input tokens are empty.
    """
    if tokens is None or (isinstance(tokens, torch.Tensor) and tokens.numel() == 0):
        raise ValueError("Cannot decode empty tokens")

    tokenizer = get_tokenizer(model_name)
    try:
        return tokenizer.decode(tokens)
    except Exception as e:
        logger.error(f"Error decoding tokens: {e}")
        raise


def convert_ids_to_tokens(
    tokens: Union[torch.Tensor, List[int]], model_name: str = DEFAULT_MODEL, **kwargs
) -> List[str]:
    """
    Convert token IDs to token strings.

    Args:
        tokens (Union[torch.Tensor, List[int]]): The token IDs to convert.
        model_name (str, optional): The model name to use for tokenization.
        **kwargs: Additional arguments to pass to the tokenizer.

    Returns:
        List[str]: The token strings.

    Raises:
        ValueError: If the input tokens are empty.
    """
    if not tokens:
        raise ValueError("Cannot convert empty token IDs")

    tokenizer = get_tokenizer(model_name)
    try:
        return tokenizer.convert_ids_to_tokens(tokens, **kwargs)
    except Exception as e:
        logger.error(f"Error converting IDs to tokens: {e}")
        raise


def convert_tokens_to_ids(tokens: List[str], model_name: str = DEFAULT_MODEL, **kwargs) -> List[int]:
    """
    Convert token strings to token IDs.

    Args:
        tokens (List[str]): The token strings to convert.
        model_name (str, optional): The model name to use for tokenization.
        **kwargs: Additional arguments to pass to the tokenizer.

    Returns:
        List[int]: The token IDs.

    Raises:
        ValueError: If the input tokens are empty.
    """
    if not tokens:
        raise ValueError("Cannot convert empty tokens")

    tokenizer = get_tokenizer(model_name)
    try:
        return tokenizer.convert_tokens_to_ids(tokens, **kwargs)
    except Exception as e:
        logger.error(f"Error converting tokens to IDs: {e}")
        raise


def convert_tokens_to_string(tokens: List[str], model_name: str = DEFAULT_MODEL) -> str:
    """
    Convert token strings to a single string, removing special tokens.

    Args:
        tokens (List[str]): The token strings to convert.
        model_name (str, optional): The model name to use for tokenization.

    Returns:
        str: The combined string.

    Raises:
        ValueError: If the input tokens are empty.
    """
    if not tokens:
        raise ValueError("Cannot convert empty tokens")

    tokenizer = get_tokenizer(model_name)
    try:
        return tokenizer.convert_tokens_to_string(tokens)
    except Exception as e:
        logger.error(f"Error converting tokens to string: {e}")
        raise


def get_loglikelihoods_of_tokens(
    input_ids: torch.Tensor, model_name: str = DEFAULT_MODEL
) -> Tuple[List[Tuple[int, float]], Any]:
    """
    Calculate the loglikelihood of each token in a text using the chosen model.

    This function computes how likely each token in the sequence is according to
    the language model given the preceding context.

    Args:
        input_ids (torch.Tensor): The token IDs to calculate loglikelihoods for.
        model_name (str, optional): The model name to use.

    Returns:
        Tuple[List[Tuple[int, float]], Any]:
            - A list of tuples (token_id, loglikelihood)
            - The model outputs (containing logits, attention, etc.)

    Raises:
        ValueError: If the input tokens are empty.
        RuntimeError: If model inference fails.
    """
    if input_ids is None or (isinstance(input_ids, torch.Tensor) and input_ids.numel() == 0):
        raise ValueError("Cannot calculate loglikelihoods for empty input")

    model = get_model(model_name)
    loss_fct = CrossEntropyLoss(reduction="none")

    # Move the input to the model's device
    input_ids = input_ids.to(model.device)

    try:
        with torch.no_grad():
            outputs = model(input_ids.unsqueeze(0), output_attentions=True, output_hidden_states=True)

        # Shift so that tokens < n predict n
        # For example, if input is "This is a text", model predicts "is a text [EOS]" from "This is a text"
        shift_logits = outputs["logits"][..., :-1, :].contiguous()
        squeezed_logits = shift_logits.view(-1, shift_logits.size(-1))  # Remove batch dimension
        shift_labels = input_ids[..., 1:].contiguous()  # Flatten the tokens
        loss = loss_fct(squeezed_logits, shift_labels.view(-1))

        # Generate a list of tuples (token_id, loglikelihood) for each token in the text
        loglikelihoods = [(input_ids[0].item(), 0)]  # Add the first token with loglikelihood 0
        for i, token in enumerate(shift_labels):
            loglikelihoods.append((token.item(), -loss[i].item()))

        return loglikelihoods, outputs

    except Exception as e:
        logger.error(f"Error calculating loglikelihoods: {e}")
        raise RuntimeError(f"Model inference failed: {e}")


def total_loglikelihood(tokens_loglikelihoods: List[Tuple[int, float]]) -> float:
    """
    Calculate the total loglikelihood of a token sequence.

    This function combines individual token loglikelihoods into a sequence probability:
    log(P(t_0)) + log(P(t_1|t_0)) + log(P(t_2|t_0,t_1)) + ... + log(P(t_n|t_0...t_n-1))

    Args:
        tokens_loglikelihoods (List[Tuple[int, float]]): List of token loglikelihoods.

    Returns:
        float: The total loglikelihood of the sequence.

    Raises:
        ValueError: If the input list is empty.
    """
    if not tokens_loglikelihoods:
        raise ValueError("Cannot calculate total loglikelihood for empty input")

    return sum(loglikelihood for _, loglikelihood in tokens_loglikelihoods)


def replace_characters(
    chars_map: Dict[str, List[str]],
    loglikelihoods_list: List[Tuple[int, float]],
    num_to_replace: int = 1,
    model_name: str = DEFAULT_MODEL,
) -> torch.Tensor:
    """
    Replace characters in tokens with highest loglikelihood using homoglyphs.

    This function selects tokens with high loglikelihoods and replaces characters
    with homoglyphs from the provided mapping.

    Args:
        chars_map (Dict[str, List[str]]): Mapping from characters to potential replacements.
        loglikelihoods_list (List[Tuple[int, float]]): List of token loglikelihoods.
        num_to_replace (int, optional): Number of characters to replace. Defaults to 1.
        model_name (str, optional): The model name to use for tokenization.

    Returns:
        torch.Tensor: A new token sequence with characters replaced.

    Raises:
        ValueError: If no suitable characters can be found to replace.
    """
    if not chars_map:
        raise ValueError("Character mapping is empty")

    if not loglikelihoods_list or len(loglikelihoods_list) < 2:
        raise ValueError("Loglikelihoods list is too short")

    tokenizer = get_tokenizer(model_name)

    # Sort tokens by loglikelihood (descending) and skip first token
    sorted_tokens = sorted(loglikelihoods_list[1:], key=lambda x: x[1], reverse=True)

    words_to_replace = []
    replacements_found = 0

    # Find characters to replace
    for word_id, loglikelihood in sorted_tokens:
        try:
            word = tokenizer.decode(word_id)

            # Check each character in the word
            for i, char in enumerate(word):
                if char in chars_map and chars_map[char]:  # Ensure we have replacements
                    # Select a random replacement character
                    replacement = random.choice(chars_map[char])
                    new_word = word[:i] + replacement + word[i + 1 :]
                    encoded_new_word = encode_text(new_word, model_name).tolist()

                    words_to_replace.append((word_id, loglikelihood, new_word, encoded_new_word))
                    replacements_found += 1

                    if replacements_found >= num_to_replace:
                        break

            if replacements_found >= num_to_replace:
                break

        except Exception as e:
            logger.debug(f"Error processing token {word_id}: {e}")
            continue

    if not words_to_replace:
        raise ValueError("Could not find any suitable characters to replace")

    # Generate the new token sequence
    new_tokens_list = []
    for word, loglikelihood in loglikelihoods_list:
        replaced = False
        for word_id, ll, _, encoded_new_word in words_to_replace:
            if word == word_id and loglikelihood == ll:
                new_tokens_list.extend(encoded_new_word)
                replaced = True
                break
        if not replaced:
            new_tokens_list.append(word)

    return torch.tensor(new_tokens_list)


def align_two_token_sequences(reference: Tensor, target: Tensor, fill_token: int = -1) -> Tensor:
    """
    Align two similar token sequences using a variant of the Needleman-Wunsch algorithm.

    This function aligns two token sequences by inserting fill tokens where needed,
    producing a sequence with the same structure as the reference.

    Example:
        reference = [1, 3, 4, 5, 6, 7, 8]
        target =    [1, 2, 4, 5, 9, 8]

        Result:     [1, 2, 4, 5, 9, FILL, 8]

    Args:
        reference (Tensor): The reference token sequence.
        target (Tensor): The target token sequence to align.
        fill_token (int, optional): Token used for filling gaps. Defaults to -1.

    Returns:
        Tensor: The aligned target sequence.
    """
    if reference is None or target is None:
        raise ValueError("Reference and target sequences cannot be None")

    if len(reference) == 0 and len(target) == 0:
        return torch.tensor([])

    # Define alignment scores
    INDEL_SCORE = -1
    MATCH_SCORE = 0
    MISMATCH_SCORE = -1

    # Create the scoring matrix
    n = len(reference)
    m = len(target)
    matrix = torch.zeros(n + 1, m + 1, dtype=torch.long)

    # Initialize the first row and column
    for i in range(n + 1):
        matrix[i][0] = -i
    for j in range(m + 1):
        matrix[0][j] = -j

    # Fill the matrix
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            indel_ref = matrix[i - 1][j] + INDEL_SCORE  # Delete from reference
            indel_target = matrix[i][j - 1] + INDEL_SCORE  # Insert from target
            match_or_mismatch = matrix[i - 1][j - 1] + (
                MATCH_SCORE if reference[i - 1] == target[j - 1] else MISMATCH_SCORE
            )
            matrix[i][j] = max(indel_ref, indel_target, match_or_mismatch)

    # Traceback to find alignment
    aligned = []
    i, j = n, m

    while i > 0 and j > 0:
        if reference[i - 1] == target[j - 1]:
            # Exact match
            aligned.append(target[j - 1])
            i -= 1
            j -= 1
        elif matrix[i - 1][j] > matrix[i][j - 1]:
            # Gap in target
            aligned.append(fill_token)
            i -= 1
        else:
            # Use target token (possibly a mismatch or gap in reference)
            aligned.append(target[j - 1])
            i -= 1
            j -= 1

    # Add remaining elements
    while i > 0:
        aligned.append(fill_token)
        i -= 1
    while j > 0:
        aligned.append(fill_token)
        j -= 1

    # Reverse to get correct order
    aligned.reverse()

    result = torch.tensor(aligned)
    assert len(result) == max(n, m)
    return result


def add_fill_tokens(
    reference: Tensor, target: Tensor, fill_token: int = -1, element_to_fill: Optional[int] = None
) -> Tensor:
    """
    Ensure target sequence matches reference sequence structure by adding fill tokens.

    This function copies elements from the target sequence, adding fill tokens
    wherever the reference sequence has them.

    Args:
        reference (Tensor): The reference token sequence with possible fill tokens.
        target (Tensor): The source token sequence to derive values from.
        fill_token (int, optional): Value identifying fill positions. Defaults to -1.
        element_to_fill (Optional[int], optional): Value to use for filling.
            If None, uses the last valid token. Defaults to None.

    Returns:
        Tensor: The modified target sequence with fill tokens inserted.
    """
    if reference is None or target is None:
        raise ValueError("Reference and target sequences cannot be None")

    if len(reference) == 0:
        return torch.tensor([])

    aligned = []
    j = 0  # Index for target sequence

    # Copy elements from target, adding fill tokens where needed
    for i in range(len(reference)):
        if reference[i] != fill_token:
            if j < len(target):
                aligned.append(target[j])
                j += 1
            else:
                # Target sequence ran out of tokens
                aligned.append(element_to_fill if element_to_fill is not None else aligned[-1])
        else:
            # Insert fill token or specified element
            if aligned and (element_to_fill is None):
                aligned.append(aligned[-1])  # Repeat last token
            else:
                aligned.append(element_to_fill if element_to_fill is not None else fill_token)

    return torch.tensor(aligned)


def get_filled_ranges(sequence: Tensor, fill_token: int = -1) -> List[Tuple[int, int]]:
    """
    Identify ranges of fill tokens in a sequence.

    Args:
        sequence (Tensor): The token sequence to analyze.
        fill_token (int, optional): Value identifying fill positions. Defaults to -1.

    Returns:
        List[Tuple[int, int]]: List of (start, end) indices where fill tokens appear.
    """
    if sequence is None or len(sequence) == 0:
        return []

    ranges = []
    start = None

    # Find continuous ranges of fill tokens
    for i, token in enumerate(sequence):
        if token == fill_token:
            if start is None:
                start = i
        elif start is not None:
            ranges.append((start, i - 1))
            start = None

    # Add final range if it extends to the end
    if start is not None:
        ranges.append((start, len(sequence) - 1))

    return ranges


def get_different_ranges(reference: Tensor, target: Tensor) -> List[Tuple[int, int]]:
    """
    Identify ranges where two sequences differ.

    Args:
        reference (Tensor): The reference token sequence.
        target (Tensor): The target token sequence to compare.

    Returns:
        List[Tuple[int, int]]: List of (start, end) indices where sequences differ.
    """
    if reference is None or target is None:
        raise ValueError("Reference and target sequences cannot be None")

    if len(reference) != len(target):
        raise ValueError(f"Sequences must be the same length: {len(reference)} vs {len(target)}")

    if len(reference) == 0:
        return []

    ranges = []
    start = None

    # Find continuous ranges of differences
    for i, (ref_token, target_token) in enumerate(zip(reference, target)):
        if ref_token != target_token:
            if start is None:
                start = i
        elif start is not None:
            ranges.append((start, i - 1))
            start = None

    # Add final range if it extends to the end
    if start is not None:
        ranges.append((start, len(reference) - 1))

    return ranges


def perform_distributed_replacements(
    text: str, translation_table: Dict[str, List[str]], percentage: float, num_chunks: int = 10
) -> str:
    """
    Perform character replacements distributed across text chunks.

    This function divides text into chunks and replaces a percentage of
    characters in each chunk with homoglyphs from the translation table.

    Args:
        text (str): The input text.
        translation_table (Dict[str, List[str]]): Mapping from characters to replacements.
        percentage (float): Percentage of characters to replace (0.0-1.0).
        num_chunks (int, optional): Number of chunks to divide text into. Defaults to 10.

    Returns:
        str: Text with distributed character replacements.

    Raises:
        ValueError: If percentage is out of range or text/translation_table is empty.
    """
    if not text:
        return text

    if not translation_table:
        return text

    if percentage < 0.0 or percentage > 1.0:
        raise ValueError("Percentage must be between 0.0 and 1.0")

    if num_chunks <= 0:
        num_chunks = 1

    # Ensure we don't create more chunks than characters
    num_chunks = min(num_chunks, len(text))

    # Divide text into chunks
    chunk_size = math.ceil(len(text) / num_chunks)
    chunks = [text[i : i + chunk_size] for i in range(0, len(text), chunk_size)]

    # Process each chunk
    for i, chunk in enumerate(chunks):
        chars = list(chunk)
        replacements_count = math.ceil(len(chars) * percentage)

        # Track positions that have viable replacements
        replaceable_positions = [
            j for j, char in enumerate(chars) if char in translation_table and translation_table[char]
        ]

        if not replaceable_positions:
            # No replaceable characters in this chunk
            continue

        # Select random positions to replace (no duplicates)
        positions_to_replace: List[int] = []
        while len(positions_to_replace) < replacements_count and replaceable_positions:
            pos_idx = random.randint(0, len(replaceable_positions) - 1)
            pos = replaceable_positions.pop(pos_idx)
            positions_to_replace.append(pos)

        # Perform replacements
        for pos in positions_to_replace:
            char = chars[pos]
            replacement = random.choice(translation_table[char])
            chars[pos] = replacement

        chunks[i] = "".join(chars)

    return "".join(chunks)


def combine_attacks(attacks: List[Callable[[str], str]]) -> Callable[[str], str]:
    """
    Combine multiple text transformation functions into a single function.

    Args:
        attacks (List[Callable[[str], str]]): List of text transformation functions.

    Returns:
        Callable[[str], str]: Combined function that applies all transformations in sequence.

    Example:
        ```python
        # Create combined attack that first performs random homoglyph replacement
        # and then adds invisible characters
        combined = combine_attacks([
            lambda text: random_attack(text, percentage=0.05),
            lambda text: add_invisible_chars(text)
        ])

        # Apply the combined attack
        result = combined("Hello world")
        ```
    """
    if not attacks:
        return lambda text: text

    def combined_attack(text: str) -> str:
        """Apply multiple text transformations in sequence."""
        result = text
        for attack in attacks:
            result = attack(result)
        return result

    return combined_attack
