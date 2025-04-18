# %%
# Generate a new homoglyph map ensuring case consistency
# This is because OCR may identify that c = C (which is +- true) but it's very evident in plain sight.

# Start by loading the JSON file we saved when generating the OCR map
import json


with open("ocr_chars_map.json", "r") as file:
    combined_homoglyph_dict = json.load(file)


def filter_same_case_homoglyph(original_char, replacements):
    if original_char.islower():
        return {char for char in replacements if not char.isupper()}
    elif original_char.isupper():
        return {char for char in replacements if not char.islower()}
    return replacements


homoglyph_same_case_map = {
    original: filter_same_case_homoglyph(original, replacements)
    for original, replacements in combined_homoglyph_dict.items()
}

# Save the same-case homoglyph map as a JSON file
homoglyph_same_case_map_serializable = {
    k: list(v)
    for k, v in homoglyph_same_case_map.items()
    # Ensure we don't have empty sequences - remove entries where the list is empty
    if len(v) > 0
}

with open("ocr_chars_refined_map.json", "w") as file:
    json.dump(homoglyph_same_case_map_serializable, file, ensure_ascii=False, indent=4)
# %%
