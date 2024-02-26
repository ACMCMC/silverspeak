# %%
import logging
import math
import random
from typing import Dict, List, Tuple

from silver_speak.utils import (decode_tokens, encode_text, get_loglikelihoods_of_tokens,
                    total_loglikelihood)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

SPACES_MAP = [
    "\u2000",
    "\u2002",
    "\u2005\u200A\u2006",
    "\u2006\u2006\u2006",
    "\u2007",
    "\u202f\u2006\u200A",
    "\u205f\u2006\u2006",
]
from silver_speak.identical_map import chars_map

# SPACES_MAP = [
#    "\u2007\u2062",
# ]


def replace_spaces(text):
    # Replaces all spaces in text with a random space from the SPACES_MAP
    perturbed_text = ""
    words = text.split(" ")
    for word in words:
        perturbed_text += word + random.choice(SPACES_MAP)
    return perturbed_text[:-1]  # Remove last space


def convert_to_char_from_hex(hex_num):
    # append 0x to hex num string to convert it.
    hex_num = "0x" + hex_num.strip(" ")
    # get the actual integer of the specific hex num with base 16.
    hex_num = int(hex_num, base=16)
    # finally get the actual character stored for specific hex char representation.
    hex_num = chr(hex_num)
    return hex_num


from silver_speak.utils import (decode_tokens, encode_text,
                                get_loglikelihoods_of_tokens,
                                replace_characters)


def decrease_loglikelihood_replace_characters_by_equivalents(
    chars_map, text, patience=10
):
    encoded_text = encode_text(text)
    loglikelihoods = get_loglikelihoods_of_tokens(encoded_text)
    print(
        f"Mean starting loglikelihood: {sum([x[1] for x in loglikelihoods]) / len(loglikelihoods)}"
    )
    current_loglikelihood = sum([x[1] for x in loglikelihoods]) / len(loglikelihoods)
    global_best_loglikelihood = current_loglikelihood
    global_best_text = encoded_text.tolist()
    current_used_patience = 0
    try:
        while patience > current_used_patience:
            new_tokens_list = replace_characters(
                chars_map, loglikelihoods, num_to_replace=1
            )
            loglikelihoods = get_loglikelihoods_of_tokens(new_tokens_list)
            current_loglikelihood = sum([x[1] for x in loglikelihoods]) / len(
                loglikelihoods
            )
            print(f"Mean loglikelihood: {current_loglikelihood}")
            print(f"New text: {decode_tokens(new_tokens_list)}")
            if current_loglikelihood < global_best_loglikelihood:
                global_best_loglikelihood = current_loglikelihood
                global_best_text = new_tokens_list.tolist()
                current_used_patience = 0
            else:
                current_used_patience += 1
    except ValueError:
        print("No more characters to replace.")

    # Reconstruct the text
    text = decode_tokens(global_best_text)
    return text


def replace_characters_by_equivalents(final_map, text):
    # Replace all chars in text with a random char from the final_map
    rewritten_text = ""
    rewrite = True
    for word in text.split(" "):
        if random.random() < 0.0:
            if random.random() < 0.4 and not rewrite:
                rewrite = not rewrite  # flip the rewrite flag
            else:
                rewrite = not rewrite  # flip the rewrite flag
        if not rewrite:
            rewritten_text += word + " "
            continue
        for char in word:
            if char in final_map.keys():
                rewritten_text += random.choice(final_map[char])
            else:
                # other type of character so write it to file as it is.
                rewritten_text += char
        rewritten_text += " "

    return rewritten_text

