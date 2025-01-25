# %% [markdown]
# # Search homoglyphs
# This script searches for homoglyphs in simplified Chinese, traditional Chinese, Korean and Japanese.
# The original similarity scores are gathered from the "Quantifying Character Similarity with Vision Transformers" paper (https://arxiv.org/pdf/2305.14672)
# %%
import pandas as pd
import pickle
from pathlib import Path
import unicodedata


SIMILARITY_THRESHOLD = 0.95

this_file_dir = "/".join(Path(__file__).resolve().parents[0].parts) + "/"

# Load the dictionary from the pickle file
with open(this_file_dir + "char_char_dist_dict_800_korean.pickle", "rb") as file:
    char_char_dist_dict_korean = pickle.load(file)

# Create a DataFrame from the dictionary
korean_df = pd.DataFrame(char_char_dist_dict_korean)

# Set the diagonal elements to 0 by comparing row and column names
for i in range(len(korean_df)):
    for j in range(len(korean_df.columns)):
        if korean_df.index[i] == korean_df.columns[j]:
            korean_df.iat[i, j] = 0

# Get the indices where the value is > 0.8
indices_korean = korean_df[korean_df > SIMILARITY_THRESHOLD].stack().index.tolist()
print(f"Found {len(indices_korean)} homoglyphs (Korean)")

# Load the dictionary from the pickle file
with open(this_file_dir + "char_char_dist_dict_800_japanese.pickle", "rb") as file:
    char_char_dist_dict_japanese = pickle.load(file)

# Create a DataFrame from the dictionary
japanese_df = pd.DataFrame(char_char_dist_dict_japanese)

# Set the diagonal elements to 0 by comparing row and column names
for i in range(len(japanese_df)):
    for j in range(len(japanese_df.columns)):
        if japanese_df.index[i] == japanese_df.columns[j]:
            japanese_df.iat[i, j] = 0

# Get the indices where the value is > 0.95
indices_japanese = japanese_df[japanese_df > SIMILARITY_THRESHOLD].stack().index.tolist()
print(f"Found {len(indices_japanese)} homoglyphs (Japanese)")


# Load the dictionary from the pickle file
with open(this_file_dir + "char_char_dist_dict_800_s_chinese_expanded_easy.pickle", "rb") as file:
    char_char_dist_dict_s_chinese = pickle.load(file)

# Create a DataFrame from the dictionary
s_chinese_df = pd.DataFrame(char_char_dist_dict_s_chinese)

# Set the diagonal elements to 0 by comparing row and column names
for i in range(len(s_chinese_df)):
    for j in range(len(s_chinese_df.columns)):
        if s_chinese_df.index[i] == s_chinese_df.columns[j]:
            s_chinese_df.iat[i, j] = 0

# Get the indices where the value is > 0.95
indices_s_chinese = s_chinese_df[s_chinese_df > SIMILARITY_THRESHOLD].stack().index.tolist()
print(f"Found {len(indices_s_chinese)} homoglyphs (Simplified Chinese)")

# Load the dictionary from the pickle file
with open(this_file_dir + "char_char_dist_dict_800_t_chinese_expanded_easy.pickle", "rb") as file:
    char_char_dist_dict_t_chinese = pickle.load(file)

# Create a DataFrame from the dictionary
t_chinese_df = pd.DataFrame(char_char_dist_dict_t_chinese)

# Set the diagonal elements to 0 by comparing row and column names
for i in range(len(t_chinese_df)):
    for j in range(len(t_chinese_df.columns)):
        if t_chinese_df.index[i] == t_chinese_df.columns[j]:
            t_chinese_df.iat[i, j] = 0

# Get the indices where the value is > 0.95
indices_t_chinese = t_chinese_df[t_chinese_df > SIMILARITY_THRESHOLD].stack().index.tolist()
print(f"Found {len(indices_t_chinese)} homoglyphs (Traditional Chinese)")

# Function to generate homoglyph dictionary
def generate_homoglyph_dict(indices_list, df):
    homoglyph_dict = {}
    for idx in indices_list:
        char1, char2 = idx
        if char1 not in homoglyph_dict:
            homoglyph_dict[char1] = set()
        if char2 not in homoglyph_dict:
            homoglyph_dict[char2] = set()
        homoglyph_dict[char1].add(char2)
        homoglyph_dict[char2].add(char1)
    return homoglyph_dict

# Generate homoglyph dictionaries for each language
homoglyph_dict_korean = generate_homoglyph_dict(indices_korean, korean_df)
homoglyph_dict_japanese = generate_homoglyph_dict(indices_japanese, japanese_df)
homoglyph_dict_s_chinese = generate_homoglyph_dict(indices_s_chinese, s_chinese_df)
homoglyph_dict_t_chinese = generate_homoglyph_dict(indices_t_chinese, t_chinese_df)

# Combine homoglyph dictionaries
combined_homoglyph_dict = {}
for d in [homoglyph_dict_korean, homoglyph_dict_japanese, homoglyph_dict_s_chinese, homoglyph_dict_t_chinese]:
    for key, value in d.items():
        if key not in combined_homoglyph_dict:
            combined_homoglyph_dict[key] = set()
        combined_homoglyph_dict[key].update(value)

print(f"Combined homoglyph dictionary: {combined_homoglyph_dict}")

# %%
# Save the combined homoglyph dictionary to a JSON file in ocr_chars_map.json
import json
import unicodedata

# Convert sets to lists for JSON serialization
combined_homoglyph_dict_serializable = {key: list(value) for key, value in combined_homoglyph_dict.items()}

with open('ocr_chars_map.json', 'w') as file:
    json.dump(combined_homoglyph_dict_serializable, file, ensure_ascii=False, indent=4)
# %%
