# %%
import random


def random_attack(original_text, percentage=0.1, random_seed=42):
    """
    Inserts random deletion characters into the text.
    """
    # Use the unicode DEL character
    DELETION_CHAR = "\u007F"
    # Insert the deletion character in random places in the text. The number of characters to insert is a percentage of the length of the text.
    random.seed(random_seed)

    # Insert the deletion characters
    text = original_text
    for i in range(int(len(text) * percentage)):
        position = random.randint(0, len(text))
        text = text[:position] + DELETION_CHAR + text[position:]

    return text
# %%
