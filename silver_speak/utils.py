# %%
"""
Utility functions for SilverSpeak.
"""

import logging
import math
from typing import List, Literal, Tuple, Union, Callable

import torch
from torch import Tensor
from transformers import AutoModelForCausalLM, AutoTokenizer, GPT2Tokenizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# %%
# Load the BLOOM tokenizer and model
MODEL = "bigscience/bloom-560m"
tokenizer = AutoTokenizer.from_pretrained(MODEL)#, use_fast=False)
model = AutoModelForCausalLM.from_pretrained(MODEL)
if torch.cuda.is_available():
    model.cuda()


# %%
def encode_text(text):
    """Encode text using the chosen tokenizer."""
    input_ids = tokenizer.encode(text, return_tensors="pt")[0]
    return input_ids


def decode_tokens(tokens):
    """Decode tokens using the chosen tokenizer."""
    return tokenizer.decode(tokens)


def convert_ids_to_tokens(tokens, **kwargs) -> List[str]:
    """Get the tokens from the token ids."""
    return tokenizer.convert_ids_to_tokens(tokens, **kwargs)


def convert_tokens_to_ids(tokens, **kwargs) -> List[int]:
    """Get the token ids from the tokens."""
    return tokenizer.convert_tokens_to_ids(tokens, **kwargs)


def convert_tokens_to_string(tokens: List[str]) -> str:
    """Remove special bytes from the text."""
    # Extracted from transformers/models/gpt2/tokenization_gpt2.py
    # text = bytearray([self.byte_decoder[c] for c in text]).decode("utf-8", errors=self.errors)
    return tokenizer.convert_tokens_to_string(tokens)


from torch.nn import CrossEntropyLoss

loss_fct = CrossEntropyLoss(reduction="none")


def get_loglikelihoods_of_tokens(input_ids: torch.Tensor) -> List[Tuple[int, float]]:
    """
    Calculate the loglikelihood of each word in a text using the chosen model.

    Returns a list of tuples (index_id, loglikelihood).
    """
    # Generate predictions
    # Move the input to the model's device
    input_ids = input_ids.to(model.device)
    with torch.no_grad():
        outputs = model(input_ids.unsqueeze(0), output_attentions=True, output_hidden_states=True)

    # Shift so that tokens < n predict n
    # For example, if we have 'This is a text', and we run it through the model, it will predict 'is a text [other token]' from 'This is a text'. We want to compare 'is a text' with 'is a text' to get the loglikelihood, so we remove the first token from the input and the last token from the output.
    shift_logits = outputs["logits"][..., :-1, :].contiguous()
    squeezed_logits = shift_logits.view(
        -1, shift_logits.size(-1)
    )  # Remove the batch dimension (1)
    shift_labels = input_ids[..., 1:].contiguous()  # Flatten the tokens
    loss = loss_fct(squeezed_logits, shift_labels.view(-1))

    # Generate a list of tuples (word, loglikelihood) for each word in the text
    loglikelihoods = [
        (input_ids[0].item(), 0)
    ]  # Add the first token with loglikelihood 0
    for i, word in enumerate(shift_labels):
        loglikelihoods.append((word.item(), -loss[i].item()))

    return loglikelihoods, outputs


def total_loglikelihood(tokens_loglikelihoods: List[Tuple[int, float]]) -> float:
    """
    This function takes a list of the loglikelihoods of a certain set of tokens and gets its conditioned probability, i.e.:
    log(P(t_0)) + log(P(t_1|t_0)) + log(P(t_2|t_1)) + ... + log(P(t_n|t_n-1))
    """
    return sum(loglikelihood for word, loglikelihood in tokens_loglikelihoods)


import random

random.seed(0)
from typing import Dict, List, Tuple


def replace_characters(
    chars_map: Dict[str, List[str]],
    loglikelihoods_list: List[Tuple[int, float]],
    num_to_replace=1,
) -> torch.Tensor:
    """
    Takes the word with the highest loglikelihood and tries to replace one of its characters with an equivalent from the chars_map.
    """
    # Repeat until we find a character to replace or we run out of words. Start by the word with the highest loglikelihood.
    words_to_replace = []
    for word_id, loglikelihood in sorted(
        loglikelihoods_list[1:], key=lambda x: x[1], reverse=True
    ):  # Skip the first word, which is the first token, because it always has loglikelihood 0
        word = tokenizer.decode(word_id)
        # See if there is a character in the word that we can replace
        for i, char in enumerate(word):
            if char in chars_map.keys():
                # Replace the character
                random_chosen_char = random.choice(chars_map[char])
                new_word = word[:i] + random_chosen_char + word[i + 1 :]
                encoded_new_word_tokens = encode_text(new_word).tolist()
                words_to_replace.append(
                    (word_id, loglikelihood, new_word, encoded_new_word_tokens)
                )
                num_to_replace -= 1
                if num_to_replace == 0:
                    break
        if num_to_replace == 0:
            break
    else:
        # We didn't find a character to replace, so raise an error
        raise ValueError("Couldn't find a character to replace.")

    new_tokens_list = []
    # Generate a new list of tokens
    for word, loglikelihood in loglikelihoods_list:
        # Check if there is a word to replace
        for (
            word_2,
            loglikelihood_2,
            new_word,
            encoded_new_word_tokens,
        ) in words_to_replace:
            if word == word_2 and loglikelihood == loglikelihood_2:
                # Append the new word tokens
                new_tokens_list += encoded_new_word_tokens
                break
        ## Remove the word from the list of words to replace
        # words_to_replace.remove((word_2, loglikelihood_2, new_word, encoded_new_word_tokens))
        else:
            # Append the original word tokens
            new_tokens_list.append(word)
    return torch.tensor(new_tokens_list)


def align_two_token_sequences(
    reference: Tensor, target: Tensor, FILL_TOKEN=-1
) -> Tensor:
    """
    Aligns two token sequences.

    We have two token sequences that are *very similar*, but not exactly the same. For example:
    reference = [1, 3, 4, 5, 6, 7, 8]
    target =    [1, 2, 4, 5, 9, 8]

    We want to align them, so that we get:
    reference = [1, 3, 4, 5, 6, 7, 8]
    target =    [1, 2, 4, 5, 9, F, 8]

    Where F is a special token that we use to fill the gap between the two sequences.

    We want to NOT change the different tokens, but we want to add the (FILL) token until we find the same element in both sequences. This way, the sequences will have the same length and we can compare them.
    """
    # We can do this by using the Needleman-Wunsch algorithm
    # However, it is important that when there is a discrepancy, the different tokens are kept at the start of the discrepancy, and if there is a need to add the FILL token, it is added at the end of the discrepancy.

    INDEL_SCORE = -1
    MATCH_SCORE = 0
    MISMATCH_SCORE = -1

    # Create the matrix
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
            # Calculate the score
            indel_score = matrix[i - 1][j] + INDEL_SCORE
            indel_score_2 = matrix[i][j - 1] + INDEL_SCORE
            match_score = matrix[i - 1][j - 1] + (
                MATCH_SCORE if reference[i - 1] == target[j - 1] else MISMATCH_SCORE
            )
            # Choose the maximum score
            matrix[i][j] = max(indel_score, indel_score_2, match_score)

    # Traceback
    aligned = []
    i = n
    j = m
    while i > 0 and j > 0:
        if reference[i - 1] == target[j - 1]:
            aligned.append(reference[i - 1])
            i -= 1
            j -= 1
        elif matrix[i - 1][j] > matrix[i][j - 1]:
            logger.debug(
                f"matrix[i-1][j] > matrix[i][j-1]: {matrix[i-1][j]} > {matrix[i][j-1]}"
            )
            logger.debug(f"Adding FILL")
            aligned.append(FILL_TOKEN)
            i -= 1
        else:
            logger.debug(
                f"matrix[i-1][j] <= matrix[i][j-1]: {matrix[i-1][j]} <= {matrix[i][j-1]}"
            )
            logger.debug(f"Adding the target token: {target[j-1]}")
            aligned.append(target[j - 1])
            i -= 1
            j -= 1

    # Add the remaining elements
    while i > 0:
        aligned.append(FILL_TOKEN)
        i -= 1
    while j > 0:
        aligned.append(FILL_TOKEN)
        j -= 1

    # Reverse the list
    aligned.reverse()

    assert len(aligned) == max(n, m)
    return torch.tensor(aligned)


def add_fill_tokens(
    reference: Tensor, target: Tensor, FILL_TOKEN=-1, ELEMENT_TO_FILL=None
) -> Tensor:
    """
    Takes every element in the reference; if it is not FILL_TOKEN, it adds the corresponding element in the target. If it is FILL_TOKEN, it adds FILL_TOKEN and then moves to the next element in the reference.

    The final sequence will have the same length as the reference.
    """
    aligned = []
    j = 0
    for i in range(len(reference)):
        if reference[i] != FILL_TOKEN:
            aligned.append(target[j])
            j += 1
        else:
            aligned.append(
                ELEMENT_TO_FILL if ELEMENT_TO_FILL is not None else aligned[-1]
            )
    return torch.tensor(aligned)


def get_filled_ranges(sequence: Tensor, FILL_TOKEN=-1) -> List[Tuple[int, int]]:
    """
    Get the ranges of the FILL_TOKEN in a sequence.
    """
    ranges = []
    start = None
    for i, token in enumerate(sequence):
        if token == FILL_TOKEN:
            if start is None:
                start = i
        else:
            if start is not None:
                ranges.append((start, i - 1))
                start = None
    if start is not None:
        ranges.append((start, len(sequence) - 1))
    return ranges

def get_different_ranges(reference: Tensor, target: Tensor) -> List[Tuple[int, int]]:
    """
    Get the ranges of the different tokens in a sequence.
    """
    ranges = []
    start = None
    for i, (ref_token, target_token) in enumerate(zip(reference, target)):
        if ref_token != target_token:
            if start is None:
                start = i
        else:
            if start is not None:
                ranges.append((start, i - 1))
                start = None
    if start is not None:
        ranges.append((start, len(reference) - 1))
    return ranges


def perform_distributed_replacements(text, translation_table, percentage):
    """
    Perform distributed replacements in a text.

    The text is divided into chunks, and for each chunk, we replace a percentage of the characters with a random character from the translation table.

    Args:
        text: The text to replace characters in.
        translation_table: A dictionary with the characters to replace and their possible replacements.
        percentage: The percentage of characters to replace.

    Returns:
        The text with the distributed replacements.
    """
    # Divide the text into chunks
    chunk_size = math.ceil(len(text) / 10)
    chunks = [text[i : i + chunk_size] for i in range(0, len(text), chunk_size)]
    # Replace characters in each chunk
    for i, chunk in enumerate(chunks):
        chunk = list(chunk)
        for j in range(math.ceil(len(chunk) * percentage)):
            index = random.randint(0, len(chunk) - 1)
            char = chunk[index]
            if char in translation_table:
                chunk[index] = random.choice(translation_table[char])
        chunks[i] = "".join(chunk)
    return "".join(chunks)


def combine_attacks(attacks: List[Callable[[str], str]]) -> Callable[[str], str]:
    """
    Combine multiple attacks into one.

    Args:
        attacks: A list of attacks.

    Returns:
        A function that applies the attacks in order.
    """

    def combined_attack(text: str) -> str:
        for attack in attacks:
            text = attack(text)
        return text

    return combined_attack


# %%
