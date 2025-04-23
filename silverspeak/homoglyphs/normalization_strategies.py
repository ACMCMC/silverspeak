from collections import Counter
from typing import List, Mapping
import unicodedata
import unicodedataplus
import logging
import tqdm
import transformers
import torch


def detect_dominant_script(text: str) -> str:
    """
    Detect the dominant script in the text.

    Args:
        text (str): Text to analyze.

    Returns:
        str: Dominant script in the text.
    """
    script_counts = Counter(unicodedataplus.script(char) for char in text)
    total_count = sum(script_counts.values())
    dominant_script = max(script_counts, key=script_counts.get)
    if script_counts[dominant_script] / total_count < 0.75:
        logging.warning(
            f"The dominant script '{dominant_script}' comprises less than 75% of the total character count. This is unusual, as most texts predominantly consist of characters from a single script. Proceed with caution, as this may affect the reliability of the analysis."
        )
    return dominant_script


def detect_dominant_block(text: str) -> str:
    """
    Detect the dominant Unicode block in the text.

    Args:
        text (str): Text to analyze.

    Returns:
        str: Dominant Unicode block in the text.
    """
    block_counts = Counter(unicodedataplus.block(char) for char in text)
    total_count = sum(block_counts.values())
    dominant_block = max(block_counts, key=block_counts.get)
    if block_counts[dominant_block] / total_count < 0.75:
        logging.warning(
            f"The dominant Unicode block '{dominant_block}' comprises less than 75% of the total character count. This is unusual, as most texts predominantly consist of characters from a single block. Proceed with caution, as this may affect the reliability of the analysis."
        )
    return dominant_block


def apply_dominant_script_strategy(replacer, text: str, **kwargs):
    """
    Normalize text based on the dominant script in the text.

    Args:
        replacer: Instance of HomoglyphReplacer.
        text (str): Text to normalize.

    Returns:
        str: Normalized text.
    """
    dominant_script = detect_dominant_script(text)
    normalization_map = replacer.get_normalization_map_for_script_block_and_category(
        script=dominant_script, **kwargs
    )
    return text.translate(str.maketrans(normalization_map))


def apply_dominant_script_and_block_strategy(replacer, text: str, **kwargs):
    """
    Normalize text based on the dominant script and block in the text.

    Args:
        replacer: Instance of HomoglyphReplacer.
        text (str): Text to normalize.

    Returns:
        str: Normalized text.
    """
    dominant_script = detect_dominant_script(text)
    dominant_block = detect_dominant_block(text)
    normalization_map = replacer.get_normalization_map_for_script_block_and_category(
        script=dominant_script, block=dominant_block, **kwargs
    )
    return text.translate(str.maketrans(normalization_map))


def apply_local_context_strategy(
    text: str,
    normalization_map: Mapping[str, List[str]],
    N: int = 10,
) -> str:
    """
    Translate the text using the provided mapping, but also trying to maximize context matches (i.e. casing, etc.). We keep a sliding window and choose the best match for each character that matches most of the properties of the N characters in the window.

    Args:
        text (str): Text to translate.
        mapping (Mapping[str, List[str]]): Mapping of characters to their replacements.
        context (Optional[Mapping[str, str]]): Context for translation.

    Returns:
        str: Translated text.
    """

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

    # Do not use a translation table here - instead, process the text character by character keeping track of all the properties of the characters in the window
    replaced_text = ""
    for i, char in enumerate(text):
        # Check if the character is in the mapping
        if char in normalization_map:
            # Now, we have a set of possibilities - the set of homoglyphs for this character
            possible_chars = [char] + normalization_map[char]
            # We need to check the context - we will use a sliding window of size N
            # Adjust the context window to always have 10 characters, even at the start or end
            # For char i, we should have i-4 to i + 4
            # To ensure that we always have 10 characters, allow to go out of bounds (i.e. negative indices)
            start = max(0, i - N // 2)
            end = min(len(text), i + N // 2 + 1)
            context_window = text[start:end]
            # If the context window is smaller than N, we need to pad it
            if start == 0:
                context_window = text[:N]
            elif end == len(text):
                context_window = text[-N:]
            else:
                pass  # Nothing to do - we have a full window

            # Get the properties of the characters in the context window
            properties = {
                prop: [PROPERTY_FNS[prop](c) for c in context_window]
                for prop in PROPERTY_FNS
            }
            # Now, we need to find the character that matches the most properties of the characters in the context window
            scores = []  # List to store scores for each possible character
            for possible_char in possible_chars:
                score = sum(
                    PROPERTY_FNS[prop](possible_char) == value
                    for prop, values in properties.items()
                    for value in values
                )
                scores.append((possible_char, score))
            # Sort the list by score in descending order and pick the best character
            best_char, best_score = max(scores, key=lambda x: x[1])
            # If there's a tie in different characters, log a warning
            if len([s for s in scores if s[1] == best_score]) > 1:
                logging.warning(
                    f"Found a tie for the best character for '{char}' (at index {i}) in context '{context_window}': {scores}. Using the first one."
                )
            # If we found a character that matches the properties, we use it
            if best_char:
                replaced_text += best_char
            else:
                # If we didn't find a character that matches the properties, we keep the original character
                replaced_text += char
        # If the character is not in the mapping, we keep it as is
        else:
            replaced_text += char

    return replaced_text


def apply_tokenizer_strategy(
    text: str,
    mapping: Mapping[str, List[str]],
    LONGEST_START_WEIGHT: float = 0.4,
    LONGEST_TOKEN_WEIGHT: float = 0.3,
    NUM_POSSIBLE_TOKENS_WEIGHT: float = 0.2,
    NUM_TOKENS_CONTAINING_CHAR_WEIGHT: float = 0.1,
    **kwargs,
):
    """
    Normalize text using a tokenizer strategy.

    Args:
        text (str): Text to normalize.

    Returns:
        str: Normalized text.
    """
    from transformers import AutoTokenizer, PreTrainedTokenizer

    # Load a tokenizer that supports a lot of languages
    tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(
        # "MaLA-LM/mala-500-10b-v1"
        # "bigscience/bloom"
        "google/gemma-3-1b-it"
    )

    vocab = tokenizer.get_vocab().keys()

    # Order the vocabulary by length
    vocab = sorted(vocab, key=len, reverse=True)

    # For all of the tokens starting with the space prefix, remove the prefix
    # i.e. if the token is "_hello", we want to keep only "hello"

    vocab = [token[1:] if token.startswith("▁") else token for token in vocab]

    normalized_text = ""

    # To select the correct characters, analyze each at a time
    for i, char in tqdm.tqdm(enumerate(text), desc="Normalizing text", total=len(text)):
        # Check if the character is in the mapping
        if char in mapping:
            # We have a set of possibilities - the set of homoglyphs for this character
            possible_chars = [char] + mapping[char]

            # Filter the vocabulary to only include tokens that contain the possible character
            possible_token_starts = {
                char: [
                    # Store only up to the place where the character is in the token (highest index)
                    # i.e. if the character is in the middle of the token, we want to keep only the left side
                    (
                        token[: token.rindex(char)],
                        len(token),
                        token,  # For debugging purposes, keep the original token
                    )  # Keep the original token because it'll be useful later
                    for token in vocab
                    if char in token
                ]
                for char in possible_chars
            }

            # Now, we want to find the biggest possible token that can be formed with the homoglyphs
            # i.e. the biggest token that is in the vocabulary
            # Go over all of the possible tokens and discard all of the ones that could not be formed with the text we have
            possible_token_starts = {
                char: [
                    token_tuple
                    for token_tuple in tokens
                    # Is the start of the token in the final part of the normalized text?
                    if normalized_text.endswith(token_tuple[0])
                ]
                for char, tokens in possible_token_starts.items()
            }

            # Remove all candidates that don't have a single token
            possible_token_starts = {
                char: v for char, v in possible_token_starts.items() if len(v) > 0
            }

            if not possible_token_starts:
                # If there are no possible tokens, we keep the original character
                logging.warning(
                    f"No possible tokens found for character '{char}' (at index {i}) in context '{text}'. Keeping the original character."
                )
                normalized_text += char
                continue

            # Calculate scores for each criterion and aggregate them with weights
            scores = {}
            individual_scores = {}
            normalized_individual_scores = {}

            for char, tokens in possible_token_starts.items():
                # Criterion 1: Average length of starts
                avg_start_length = sum(len(token[0]) for token in tokens) / len(tokens)

                # Criterion 2: Average token length
                avg_token_length = sum(token[1] for token in tokens) / len(tokens)

                # Criterion 3: The one that has the largest number of possible tokens
                num_possible_tokens_score = len(tokens)

                # Criterion 4: Number of tokens containing the character (largest list of possible starts)
                num_tokens_containing_char = len(possible_token_starts[char])

                # Store individual scores
                individual_scores[char] = {
                    "avg_start_length": avg_start_length,
                    "avg_token_length": avg_token_length,
                    "num_possible_tokens_score": num_possible_tokens_score,
                    "num_tokens_containing_char": num_tokens_containing_char,
                }

            # Calculate the maximum values for normalization after individual scores are computed
            max_avg_start_length = (
                max(score["avg_start_length"] for score in individual_scores.values())
                or 1
            )  # Avoid division by zero
            max_avg_token_length = (
                max(score["avg_token_length"] for score in individual_scores.values())
                or 1
            )  # Avoid division by zero
            max_num_possible_tokens = (
                max(
                    score["num_possible_tokens_score"]
                    for score in individual_scores.values()
                )
                or 1
            )  # Avoid division by zero
            max_num_tokens_containing_char = (
                max(
                    score["num_tokens_containing_char"]
                    for score in individual_scores.values()
                )
                or 1
            )  # Avoid division by zero

            for char, scores_dict in individual_scores.items():
                # Normalize individual scores
                longest_start_score = (
                    scores_dict["avg_start_length"] / max_avg_start_length
                )
                longest_token_score = (
                    scores_dict["avg_token_length"] / max_avg_token_length
                )
                num_possible_tokens_score = (
                    scores_dict["num_possible_tokens_score"] / max_num_possible_tokens
                )
                num_tokens_containing_char = (
                    scores_dict["num_tokens_containing_char"]
                    / max_num_tokens_containing_char
                )

                # Store normalized individual scores
                normalized_individual_scores[char] = {
                    "longest_start_score": longest_start_score,
                    "longest_token_score": longest_token_score,
                    "num_possible_tokens_score": num_possible_tokens_score,
                    "num_tokens_containing_char": num_tokens_containing_char,
                }

                # Aggregate scores with parameterized weights
                scores[char] = (
                    LONGEST_START_WEIGHT * longest_start_score
                    + LONGEST_TOKEN_WEIGHT * longest_token_score
                    + NUM_POSSIBLE_TOKENS_WEIGHT * num_possible_tokens_score
                    + NUM_TOKENS_CONTAINING_CHAR_WEIGHT * num_tokens_containing_char
                )

            # Select the character with the highest aggregated score
            best_char = max(scores, key=scores.get)
            normalized_text += best_char
        else:
            # If the character is not in the mapping, we keep it as is
            normalized_text += char

    # Join the normalized tokens back into a single string
    return normalized_text


@torch.inference_mode()
def apply_language_model_strategy(
    text: str,
    mapping: Mapping[str, List[str]],
    language_model: transformers.PreTrainedModel,
    tokenizer: transformers.PreTrainedTokenizer,
    **kwargs,
):
    """
    Normalize text using a language model strategy.

    Args:
        text (str): Text to normalize.

    Returns:
        str: Normalized text.
    """
    normalized_text = text
    return normalized_text
