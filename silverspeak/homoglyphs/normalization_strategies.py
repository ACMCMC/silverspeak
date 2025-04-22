from collections import Counter
from typing import List, Mapping
import unicodedata
import unicodedataplus
import logging
import tqdm


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
    normalization_map = replacer.get_normalization_map_for_script_and_block(
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
    normalization_map = replacer.get_normalization_map_for_script_and_block(
        script=dominant_script, block=dominant_block, **kwargs
    )
    return text.translate(str.maketrans(normalization_map))


def translate_with_context(
    text: str,
    mapping: Mapping[str, List[str]],
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
    replaced_text = []
    for i, char in enumerate(text):
        # Check if the character is in the mapping
        if char in mapping:
            # Now, we have a set of possibilities - the set of homoglyphs for this character
            possible_chars = [char] + mapping[char]
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
                replaced_text.append(best_char)
            else:
                # If we didn't find a character that matches the properties, we keep the original character
                replaced_text.append(char)
        # If the character is not in the mapping, we keep it as is
        else:
            replaced_text.append(char)

    return "".join(replaced_text)


def apply_context_aware_strategy(normalization_map, text, **kwargs):
    """
    Normalize text using a context-aware strategy.

    Args:
        normalization_map: The normalization map to use.
        text (str): Text to normalize.

    Returns:
        str: Normalized text.
    """
    return translate_with_context(text, normalization_map)


def apply_tokenizer_strategy(text: str, mapping: Mapping[str, List[str]], **kwargs):
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

    normalized_text = []

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
                    # Is the start of the token in the text?
                    if text.startswith(token_tuple[0], i - len(token_tuple[0]), i)
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
                normalized_text.append(char)
                continue
            elif len(possible_token_starts) == 1:
                # If there's only one possible token, we just take it
                normalized_text.append(next(iter(possible_token_starts.keys())))
                continue

            # Now that we have all possible tokens, we want to find the one that matches our success criteria:
            # The best char is the one that:
            # 1. Has the longest start
            # 2. Has the longest token
            # 3. Among those possible characters, we want the one that has the largest number of possible tokens
            # 4. If there's a tie, we want to keep the default best char which is the one that had the largest list of possible starts

            # 1. Has the longest start
            max_possible_token_start_length = max(
                max(len(token[0]) for token in tokens)
                for tokens in possible_token_starts.values()
            )

            # Discard all of the possible tokens that are not the longest
            possible_max_len_token_starts = {}
            for char, tokens in possible_token_starts.items():
                for token in tokens:
                    if len(token[0]) == max_possible_token_start_length:
                        possible_max_len_token_starts.setdefault(char, []).append(token)
                    # We can't break here because we need to check all of the tokens - they are sorted by length of the full token, not the length of the start
            # If there's only one candidate, we just take it
            if len(possible_max_len_token_starts) == 1:
                normalized_text.append(next(iter(possible_max_len_token_starts.keys())))
                continue

            # 2. Has the longest token
            # What are the chars with the max length? Since the lists are sorted, we can just take the length of the first one
            max_possible_token_length = max(
                # The length (1) of the first token in the list of possible tokens (0)
                v[0][1]
                for v in possible_max_len_token_starts.values()
            )

            # Discard all of the possible tokens that are not the longest
            for char, tokens in possible_max_len_token_starts.items():
                for token_index, token in enumerate(tokens):
                    if token[1] == max_possible_token_length:
                        # Don't do anything - we want to keep this token
                        pass
                    else:
                        # If the token is not the longest, we discard it and all after itself
                        possible_max_len_token_starts[char] = possible_max_len_token_starts[char][
                            : token_index
                        ]
                        # since the lists are sorted by full token length, we can just break
                        break

            # Discard any possible characters that have no possible tokens
            possible_max_len_token_starts = {
                char: v
                for char, v in possible_max_len_token_starts.items()
                if len(v) > 0
            }

            # If there's only one candidate, we just take it
            if len(possible_max_len_token_starts) == 1:
                normalized_text.append(next(iter(possible_max_len_token_starts.keys())))
                continue

            # 3. Among those possible characters, we want the one that has the largest number of possible tokens

            # Now, we have a list of possible tokens that are the longest
            # We want to find the one that has the largest number of possible tokens
            # i.e. the one that has the largest number of possible tokens
            # We can do this by just taking the length of the list of the tokens of the possible token starts
            max_number_of_possible_tokens = max(
                len(v) for v in possible_max_len_token_starts.values()
            )
            # Now, we want to find the one that has the largest number of possible tokens
            possible_max_len_token_starts = {
                k: v
                for k, v in possible_max_len_token_starts.items()
                if len(v) == max_number_of_possible_tokens
            }
            if len(possible_max_len_token_starts) == 1:
                normalized_text.append(next(iter(possible_max_len_token_starts.keys())))
                continue

            # 4. If there's a tie, we want to keep the default best char which is the one that had the largest list of possible starts
            logging.warning(
                f"Found multiple candidates for the best character for '{char}' (at index {i}) in context '{text}': {possible_max_len_token_starts}. Using the one with the longest token."
            )
            normalized_text.append(
                max(
                    possible_max_len_token_starts.keys(),
                    # Use the length of the first token in the list of all possible tokens (possible_token_starts), not just the list of the longest tokens
                    key=lambda x: len(possible_token_starts[x][0]),
                )
            )
            continue

        else:
            # If the character is not in the mapping, we keep it as is
            normalized_text.append(char)

    # Join the normalized tokens back into a single string
    return "".join(normalized_text)
