from silver_speak.identical_map import chars_map
import random


def random_attack(text: str, probability=0.5, random_seed=42) -> str:
    """
    Replaces some characters in the text, randomly choosing which ones, leaving all others unchanged.
    """
    random.seed(random_seed)
    # Replace some characters in the text
    return "".join(
        (
            char
            if random.random() > probability
            else random.choice(chars_map.get(char, [char]))
        )
        for char in text
    )
