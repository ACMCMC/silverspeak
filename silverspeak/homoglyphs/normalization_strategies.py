from collections import Counter
from typing import List, Mapping
import unicodedata
import unicodedataplus
import logging


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


def _generate_homoglyph_replacements(token: str) -> list:
    """
    Generate possible homoglyph replacements for a given token.

    Args:
        token (str): The input token for which homoglyph replacements are to be generated.

    Returns:
        list: A list of possible homoglyph replacements for the input token.
    """
    from search_homolgyphs.unicode_confusables_map import UNICODE_CONFUSABLES_MAP

    # Initialize a list to store possible replacements
    replacements = []

    # Iterate over each character in the token
    for char in token:
        # Check if the character has homoglyphs in the confusables map
        if char in UNICODE_CONFUSABLES_MAP:
            # Replace the character with each of its homoglyphs
            for homoglyph in UNICODE_CONFUSABLES_MAP[char]:
                # Generate a new token with the homoglyph replacement
                new_token = token.replace(char, homoglyph)
                replacements.append(new_token)

    # Return the list of possible replacements
    return replacements


def apply_tokenizer_strategy(text: str, **kwargs):
    """
    Normalize text using a tokenizer strategy.

    Args:
        text (str): Text to normalize.

    Returns:
        str: Normalized text.
    """
    from transformers import AutoTokenizer

    # Load the GPT-2 tokenizer
    tokenizer = AutoTokenizer.from_pretrained("gpt2")

    # Tokenize the input text
    tokens = tokenizer.tokenize(text)

    # Initialize the normalized text
    normalized_text = []

    for token in tokens:
        # Generate possible homoglyph replacements for the token
        possible_replacements = _generate_homoglyph_replacements(token)

        # Find the replacement that produces the best match in the tokenizer's vocabulary
        best_match = max(
            possible_replacements,
            key=lambda replacement: tokenizer.vocab.get(replacement, -1),
        )

        # Append the best match to the normalized text
        normalized_text.append(best_match)

    # Join the normalized tokens back into a single string
    return "".join(normalized_text)
