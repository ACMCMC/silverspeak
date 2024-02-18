# %%
"""
Utility functions for SilverSpeak.
"""

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import math
from typing import List, Tuple, Literal

# Load GPT-2 tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("gpt2")
model = AutoModelForCausalLM.from_pretrained("gpt2")
if torch.cuda.is_available():
    model.cuda()

# %%
def encode_text(text):
    """Encode text using GPT-2 tokenizer."""
    input_ids = tokenizer.encode(text, return_tensors="pt")[0]
    if torch.cuda.is_available():
        input_ids = input_ids.cuda()
    return input_ids

def decode_tokens(tokens):
    """Decode tokens using GPT-2 tokenizer."""
    return tokenizer.decode(tokens)

from torch.nn import CrossEntropyLoss
loss_fct = CrossEntropyLoss(reduction='none')

def tokens_loglikelihoods(input_ids: torch.Tensor) -> List[Tuple[str, float]]:
    """Calculate the loglikelihood of each word in a text using GPT-2."""
    # Generate predictions
    with torch.no_grad():
        outputs = model(input_ids)
    
    # Shift so that tokens < n predict n
    # For example, if we have 'This is a text', and we run it through the model, it will predict 'is a text [other token]' from 'This is a text'. We want to compare 'is a text' with 'is a text' to get the loglikelihood, so we remove the first token from the input and the last token from the output.
    shift_logits = outputs['logits'][..., :-1, :].contiguous()
    squeezed_logits = shift_logits.view(-1, shift_logits.size(-1)) # Remove the batch dimension (1)
    shift_labels = input_ids[..., 1:].contiguous() # Flatten the tokens
    loss = loss_fct(squeezed_logits, shift_labels.view(-1))
    
    # Generate a list of tuples (word, loglikelihood) for each word in the text
    loglikelihoods = [(input_ids[0].item(), 0)] # Add the first token with loglikelihood 0
    for i, word in enumerate(shift_labels):
        loglikelihoods.append((word.item(), -loss[i].item()))

    return loglikelihoods

def total_loglikelihood(tokens_loglikelihoods: List[Tuple[str, float]]) -> float:
    """
    This function takes a list of the loglikelihoods of a certain set of tokens and gets its conditioned probability, i.e.:
    log(P(t_0)) + log(P(t_1|t_0)) + log(P(t_2|t_1)) + ... + log(P(t_n|t_n-1))
    """
    return sum(loglikelihood for word, loglikelihood in tokens_loglikelihoods)

# %%
import random
random.seed(0)
from typing import List, Tuple, Dict
def replace_characters(chars_map: Dict[str, List[str]], loglikelihoods_list: List[Tuple[int, float]], num_to_replace = 1) -> torch.Tensor:
    """
    Takes the word with the highest loglikelihood and tries to replace one of its characters with an equivalent from the chars_map. 
    """
    # Repeat until we find a character to replace or we run out of words. Start by the word with the highest loglikelihood.
    words_to_replace = []
    for word_id, loglikelihood in sorted(loglikelihoods_list[1:], key=lambda x: x[1], reverse=True): # Skip the first word, which is the first token, because it always has loglikelihood 0
        word = tokenizer.decode(word_id)
        #print(f'word: {word}, loglikelihood: {loglikelihood}')
        # See if there is a character in the word that we can replace
        for i, char in enumerate(word):
            if char in chars_map.keys():
                # Replace the character
                random_chosen_char = random.choice(chars_map[char])
                new_word = word[:i] + random_chosen_char + word[i+1:]
                encoded_new_word_tokens = encode_text(new_word).tolist()
                words_to_replace.append((word_id, loglikelihood, new_word, encoded_new_word_tokens))
                #print(f'new_word: {new_word}, encoded_new_word_tokens: {encoded_new_word_tokens}')
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
        for word_2, loglikelihood_2, new_word, encoded_new_word_tokens in words_to_replace:
            if word == word_2 and loglikelihood == loglikelihood_2:
                # Append the new word tokens
                new_tokens_list += encoded_new_word_tokens
                break
        ## Remove the word from the list of words to replace
        #words_to_replace.remove((word_2, loglikelihood_2, new_word, encoded_new_word_tokens))
        else:
            # Append the original word tokens
            new_tokens_list.append(word)
    return torch.tensor(new_tokens_list)

# %%
