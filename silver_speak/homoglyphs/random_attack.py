# %%
from silver_speak.homoglyphs.identical_map import chars_map
import random


def random_attack(text: str, percentage=0.1, random_seed=42) -> str:
    """
    Replaces some characters in the text, randomly choosing which ones, leaving all others unchanged.
    """
    random.seed(random_seed)
    # Replace some characters in the text with their equivalent characters from the chars_map
    num_to_replace = int(len(text) * percentage)
    text = list(text)
    while num_to_replace > 0:
        position = random.randint(0, len(text) - 1)
        char = text[position]
        if char in chars_map:
            text[position] = chars_map[char]
            num_to_replace -= 1
    return "".join(text)
# %%