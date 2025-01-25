# %%
import json
import os
import pathlib

# Define the file paths
file_paths = [
    os.path.join(pathlib.Path(__file__).parent, "identical_map.json"),
    os.path.join(pathlib.Path(__file__).parent, "unicode_confusables.json"),
    os.path.join(pathlib.Path(__file__).parent, "ocr_chars_map.json"),
]

# Load the JSON files
combined_map = {}
for file_path in file_paths:
    with open(file_path, "r") as file:
        data = json.load(file)
        for key, value in data.items():
            if key not in combined_map:
                combined_map[key] = set()
            combined_map[key].update(value)

# Convert sets to lists for JSON serialization
combined_map_serializable = {key: list(value) for key, value in combined_map.items()}

# Save the combined map to 'combined_maps.json'
with open(os.path.join(pathlib.Path(__file__).parent, "combined_maps.json"), "w") as file:
    json.dump(combined_map_serializable, file, ensure_ascii=False, indent=4)

# %%
