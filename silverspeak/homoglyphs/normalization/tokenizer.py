"""
Tokenizer-based normalization strategy for homoglyph replacement.

This module provides functionality to normalize text by analyzing how potential
homoglyph replacements impact tokenization patterns.

Author: Aldan Creo (ACMC) <os@acmc.fyi>
"""

import logging
from typing import List, Mapping

logger = logging.getLogger(__name__)


def apply_tokenizer_strategy(
    text: str,
    mapping: Mapping[str, List[str]],
    tokenizer_name: str = "google/gemma-3-1b-pt",
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

    # Remove space special prefixes (varies by tokenizer type)
    # First get what's the special prefix for this tokenizer
    # Tokenize 'Hello' and ' Hello' to see if they differ
    hello_with_space_prefix = tokenizer(" Hello", add_special_tokens=False)["input_ids"]
    assert len(hello_with_space_prefix) == 1, "Tokenizer should return a single token for ' Hello'"
    decoded_token = tokenizer.convert_ids_to_tokens(hello_with_space_prefix[0])
    # Space prefix is everything before 'Hello'
    space_prefix = decoded_token[: decoded_token.index("Hello")]
    # Remove all tokens that start with this space prefix
    vocab = [
        token for token in vocab if not token.startswith(space_prefix)
    ]  # We remove instead of replacing because that greatly reduces the number of tokens we have to process (we will account for having removed the space prefix later by always using the space character if it is a possible homoglyph)

    # Initialize normalized text
    normalized_text = ""

    # Process each character with a progress bar
    for i, char in enumerate(text):
        # Check if the character is in mapping
        if char in mapping:
            # Get possible homoglyphs including original character
            possible_chars = [char] + mapping[char]

            # As we're removing space prefixes, if the space ' ' is a possible character, we choose it automatically - this is a common case but could perhaps be made configurable
            if " " in possible_chars:
                logging.debug(f"Character '{char}' at index {i} is a space, using it directly")
                normalized_text += " "
                continue

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
