# %%
from silverspeak.homoglyphs.identical_map import chars_map

translation_table = str.maketrans({k: v[0] for k, v in chars_map.items()})


def greedy_attack(text: str) -> str:
    """
    Fastest attack. It replaces all possible characters in the text.
    """
    # Build a translation table from the chars_map
    return text.translate(translation_table)


# %%
