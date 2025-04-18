# %%
import os
import pathlib
import json

import pandas as pd

# Chars map is a CSV file in this directory. Ignore everything after the # symbol.
chars_map = pd.read_csv(
    os.path.join(pathlib.Path(__file__).parent, "unicode_confusables_map.csv"),
    sep=";",
    header=None,
    names=["original", "replacement", "_"],
    comment="#",
)

# Strip the whitespace from the original and replacement columns
chars_map["original"] = chars_map["original"].str.strip()
chars_map["replacement"] = chars_map["replacement"].str.strip()

# Filter out rows where the original column has more than one character
chars_map = chars_map[chars_map["original"].str.split().str.len() == 1]

# Convert all characters in the replacement column
chars_map["replacement"] = chars_map["replacement"].apply(
    lambda x: {chr(int(char, 16)) for char in x.split()}
)

# Now, we have two columns with rows like '0021' and '0306 0307'. We want to map this to their Unicode characters.
# We will use the chr() function to convert the hex to a Unicode character.
chars_map = {
    chr(int(row["original"], 16)): row["replacement"] for _, row in chars_map.iterrows()
}

# Save it as a JSON file
chars_map_json = {k: list(v) for k, v in chars_map.items()}

with open(os.path.join(pathlib.Path(__file__).parent, "unicode_confusables_map.json"), "w") as file:
    json.dump(chars_map_json, file, ensure_ascii=False, indent=4)

# Generate a new map ensuring case consistency
def filter_same_case(original_char, replacements):
    if original_char.islower():
        return {char for char in replacements if not char.isupper()}
    elif original_char.isupper():
        return {char for char in replacements if not char.islower()}
    return replacements

unicode_confusables_same_case_map = {
    original: filter_same_case(original, replacements)
    for original, replacements in chars_map.items()
}

# Save the same-case map as a JSON file
unicode_confusables_same_case_map_json = {k: list(v) for k, v in unicode_confusables_same_case_map.items()}

with open(os.path.join(pathlib.Path(__file__).parent, "unicode_confusables_same_case_map.json"), "w") as file:
    json.dump(unicode_confusables_same_case_map_json, file, ensure_ascii=False, indent=4)
# %%
