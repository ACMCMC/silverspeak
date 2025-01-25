import json
import random
import unicodedata
from typing import List, Literal, Mapping, Set
from pathlib import Path


class HomoglyphReplacer:
    def __init__(
        self,
        unicode_categories_to_replace: Set[str] = set(["Ll", "Lm", "Lo", "Lt", "Lu"]),
        types_of_homoglyphs_to_use: List[Literal["identical", "confusables", "ocr"]] = [
            "identical",
            "confusables",
            "ocr",
        ],
        replace_with_priority: bool = False,
        random_seed: int = 42,
    ):
        self.unicode_categories_to_replace = unicode_categories_to_replace
        self.types_of_homoglyphs_to_use = types_of_homoglyphs_to_use
        self.replace_with_priority = replace_with_priority
        self.chars_map: Mapping[str, List[str]] = self._load_chars_map()
        # This object will be used to keep the random state
        self.random_state = random.Random(x=random_seed)

    def _load_chars_map(self):
        files_mapping = {
            "identical": "identical_map.json",
            "confusables": "unicode_confusables_map.json",
            "ocr": "ocr_chars_map.json",
        }
        # Load the JSON files
        chars_map = {}
        for homoglyph_type in self.types_of_homoglyphs_to_use:
            with open(Path(__file__).parent / files_mapping[homoglyph_type], "r") as file:
                data = json.load(file)
                for key, value in data.items():
                    if key not in chars_map:
                        chars_map[key] = []
                    chars_map[key].extend(value)

        if self.replace_with_priority:
            # Only keep the first element in the list
            for key, value in chars_map.items():
                chars_map[key] = [value[0]]

        return chars_map

    def is_replaceable(self, char):
        return (
            char in self.chars_map
            and unicodedata.category(char) in self.unicode_categories_to_replace
        )

    def get_homoglpyh(self, char):
        return self.random_state.choice(self.chars_map[char])
