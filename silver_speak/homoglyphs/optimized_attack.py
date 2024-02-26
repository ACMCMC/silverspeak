# %%
from silver_speak.utils import (
    encode_text,
    get_loglikelihoods_of_tokens,
    decode_tokens,
    convert_ids_to_tokens,
    convert_tokens_to_ids,
    convert_tokens_to_string,
)
from silver_speak.homoglyphs.identical_map import chars_map
import random
import logging

logger = logging.getLogger(__name__)


def optimized_attack(text: str, percentage_to_replace=0.2, random_seed=42) -> str:
    """
    Attack that is much faster than SilverSpeak*.

    It starts by getting the loglikelihoods of the tokens in the text. Then, it replaces the characters in the text that have the highest loglikelihoods with their equivalent characters from the chars_map. However, it doesn't run the text through the model to get the loglikelihoods again.
    """
    encoded_text = encode_text(text)
    loglikelihoods = get_loglikelihoods_of_tokens(encoded_text)
    tokens_strs = convert_ids_to_tokens(encoded_text, skip_special_tokens=True)

    positions_of_tokens_strs_and_loglikelihoods = list(
        enumerate(zip(tokens_strs, map(lambda x: x[1], loglikelihoods)))
    )

    # Sort the tokens by loglikelihood
    sorted_positions_of_tokens_strs_and_loglikelihoods = sorted(
        positions_of_tokens_strs_and_loglikelihoods, key=lambda x: x[1][1], reverse=True
    )

    replaceable_positions = [
        x
        for x in sorted_positions_of_tokens_strs_and_loglikelihoods
        if any(char in chars_map.keys() for char in x[1][0])
    ]

    # Replace the top percentage_to_replace tokens
    num_to_replace = min(
        int(len(sorted_positions_of_tokens_strs_and_loglikelihoods) * percentage_to_replace),
        len(replaceable_positions),
    )
    random.seed(random_seed)
    changes_to_make = []
    changed_text = text
    for i in range(num_to_replace):
        position, (token_str, loglikelihood) = replaceable_positions[i]
        start_of_token = len(
            convert_tokens_to_string([x[1][0] for x in positions_of_tokens_strs_and_loglikelihoods[:position]])
        )
        logger.debug(f"Replacing token {token_str} at position {start_of_token}")
        # Replace the token
        for char_index, char in enumerate(token_str):
            if char in chars_map:
                chosen_char = random.choice(chars_map[char])
                # The position in the text is the sum of the lengths of the tokens before it
                position_in_text = start_of_token + char_index
                # Do that change in the changed_text
                changed_text = (
                    changed_text[:position_in_text]
                    + chosen_char
                    + changed_text[position_in_text + 1:]
                )
                logger.debug(f"Replaced '{text[position_in_text-5:position_in_text]}|{text[position_in_text]}|{text[position_in_text+1:position_in_text+6]}' with {changed_text[position_in_text]} at position {position_in_text}")
                # It's enough to change one character in the token
                break

    # Reconstruct the text - wouldn't work because we have used non-ascii characters
    # and that makes the decoder fail
    # text = convert_tokens_to_string(tokens_strs)

    return changed_text
# %%
if __name__ == "__main__":
    # Test the attack
    test_text = """Dr. Capy Cosmos, a capybara unlike any other, astounded the scientific community with his groundbreaking research in astrophysics. With his keen sense of observation and unparalleled ability to interpret cosmic data, he uncovered new insights into the mysteries of black holes and the origins of the universe. As he peered through telescopes with his large, round eyes, fellow researchers often remarked that it seemed as if the stars themselves whispered their secrets directly to him. Dr. Cosmos not only became a beacon of inspiration to aspiring scientists but also proved that intellect and innovation can be found in the most unexpected of creatures."""
    changed_text = optimized_attack(test_text, percentage_to_replace=0.25, random_seed=42)
    print(changed_text)
