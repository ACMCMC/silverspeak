# %%
"""
Utility functions for SilverSpeak.
"""

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

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

def loglikelihood(input_ids: torch.Tensor):
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

# %%
import random
random.seed(0)
from typing import List, Tuple, Dict
def replace_one_character(chars_map: Dict[str, List[str]], loglikelihoods_list: List[Tuple[int, float]]) -> torch.Tensor:
    """
    Takes the word with the highest loglikelihood and tries to replace one of its characters with an equivalent from the chars_map. 
    """
    # Repeat until we find a character to replace or we run out of words. Start by the word with the highest loglikelihood.
    for word_id, loglikelihood in sorted(loglikelihoods_list[1:], key=lambda x: x[1], reverse=True): # Skip the first word, which is the first token, because it always has loglikelihood 0
        word = tokenizer.decode(word_id)
        print(f'word: {word}, loglikelihood: {loglikelihood}')
        # See if there is a character in the word that we can replace
        for i, char in enumerate(word):
            if char in chars_map.keys():
                # Replace the character
                random_chosen_char = random.choice(chars_map[char])
                new_word = word[:i] + random_chosen_char + word[i+1:]
                encoded_new_word_tokens = encode_text(new_word).tolist()
                print(f'new_word: {new_word}, encoded_new_word_tokens: {encoded_new_word_tokens}')
                
                # Generate a new list of tokens
                index_of_word = loglikelihoods_list.index((word_id, loglikelihood))
                new_tokens_list = [w for w, l in loglikelihoods_list[:index_of_word]] + encoded_new_word_tokens + [w for w, l in loglikelihoods_list[index_of_word+1:]]
                return torch.tensor(new_tokens_list)

    else:
        # We didn't find a character to replace, so raise an error
        raise ValueError("Couldn't find a character to replace.")

# %%
