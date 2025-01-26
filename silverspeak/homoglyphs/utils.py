import json
import random
import unicodedata
from pathlib import Path
from typing import List, Literal, Mapping, Set, Tuple

from uniscripts import get_script_counts, is_script


class HomoglyphReplacer:
    """
    A class to replace characters with their homoglyphs.

    Attributes:
        unicode_categories_to_replace (Set[str]): Unicode categories of characters to replace.
        types_of_homoglyphs_to_use (List[Literal]): Types of homoglyphs to use.
        replace_with_priority (bool): Whether to replace with priority.
        random_state (random.Random): Random state for reproducibility.
        chars_map (Mapping[str, List[str]]): Mapping of characters to their homoglyphs.
        reverse_chars_map (Mapping[str, str]): Reverse mapping of homoglyphs to original characters.
        normalization_table (dict): Table for normalizing text.
    """

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
        """
        Initialize the HomoglyphReplacer.

        Args:
            unicode_categories_to_replace (Set[str]): Unicode categories of characters to replace.
            types_of_homoglyphs_to_use (List[Literal]): Types of homoglyphs to use.
            replace_with_priority (bool): Whether to replace with priority.
            random_seed (int): Seed for random number generator.
        """
        self.unicode_categories_to_replace = unicode_categories_to_replace
        self.types_of_homoglyphs_to_use = types_of_homoglyphs_to_use
        self.replace_with_priority = replace_with_priority
        self.chars_map = self._load_chars_map()
        # This object will be used to keep the random state
        self.random_state = random.Random(x=random_seed)
        self.reverse_chars_map: Mapping[str, str] = self._create_reverse_chars_map()
        self._base_normalization_map = self._create_base_normalization_map()
        self.normalization_translation_tables = {}

    def _load_chars_map(self) -> Mapping[str, List[str]]:
        """
        Load the character mappings from JSON files.

        Returns:
            Mapping[str, List[str]]: Mapping of characters to their homoglyphs.
        """
        files_mapping = {
            "identical": "identical_map.json",
            "confusables": "unicode_confusables_map.json",
            "ocr": "ocr_chars_map.json",
        }
        # Load the JSON files
        chars_map = {}
        for homoglyph_type in self.types_of_homoglyphs_to_use:
            with open(
                Path(__file__).parent / files_mapping[homoglyph_type], "r"
            ) as file:
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

    def _create_reverse_chars_map(self):
        """
        Create a reverse mapping of homoglyphs to original characters.

        Returns:
            Mapping[str, str]: Reverse mapping of homoglyphs to original characters.
        """
        reverse_map = {}
        for key, values in self.chars_map.items():
            for value in values:
                # If there's already a key for this value, don't overwrite it - the first one we find is the highest priority
                if value not in reverse_map:
                    reverse_map[value] = key

        return reverse_map

    def _create_base_normalization_map(self):
        """
        Create a base normalization map for the most common homoglyphs.

        Returns:
            dict: Base normalization map.
        """
        # Don't include any entries for characters that are in NFKD normalization form
        # I.e., only include entries for characters that are not in normalization form (which are the homoglyphs)
        # Also, only include entries for characters that are in list of unicode categories to replace
        base_normalization_map = {
            key: value
            for key, value in self.reverse_chars_map.items()
            if not unicodedata.is_normalized("NFKD", key) and unicodedata.category(key) in self.unicode_categories_to_replace
        }
        return base_normalization_map

    def is_replaceable(self, char: str) -> bool:
        """
        Check if a character is replaceable with a homoglyph.

        Args:
            char (str): Character to check.

        Returns:
            bool: True if the character is replaceable, False otherwise.
        """
        return (
            char in self.chars_map
            and unicodedata.category(char) in self.unicode_categories_to_replace
        )

    def get_homoglpyh(self, char: str) -> str:
        """
        Get a homoglyph for a character.

        Args:
            char (str): Character to replace.

        Returns:
            str: Homoglyph for the character.
        """
        return self.random_state.choice(self.chars_map[char])

    def is_dangerous(self, char: str) -> bool:
        """
        Check if a character is a dangerous homoglyph.

        Args:
            char (str): Character to check.

        Returns:
            bool: True if the character is a dangerous homoglyph, False otherwise.
        """
        return char in self.reverse_chars_map

    def get_original(self, char: str) -> str:
        """
        Get the original character for a homoglyph.

        Args:
            char (str): Homoglyph to replace.

        Returns:
            str: Original character for the homoglyph.
        """
        return self.reverse_chars_map[char]

    def _detect_dominant_script(self, text: str) -> str:
        """
        Detect the dominant script in the text.

        Args:
            text (str): Text to analyze.

        Returns:
            str: Dominant script in the text.
        """
        script_counts = get_script_counts(text)
        return max(script_counts, key=script_counts.get)

    def _get_normalization_table_for_script(self, script: str) -> dict:
        """
        Get the normalization table for a script.

        Args:
            script (str): Script to get the normalization table for.

        Returns:
            dict: Normalization table for the script.
        """
        # Check if it's in our in-memory cache
        if script in self.normalization_translation_tables:
            return self.normalization_translation_tables[script]

        # Generate the normalization table.
        base_normalization_map = self._base_normalization_map

        # Create a dictionary with the NFC entries where the value is in the script we want to normalize (i.e. (this_or_other_script, script) pairs)
        script_normalization_map = {
            key: value
            for key, value in base_normalization_map.items()
            if is_script(value, script)
        }
        script_normalization_translation_table = str.maketrans(script_normalization_map)

        self.normalization_translation_tables[script] = (
            script_normalization_translation_table
        )

        return script_normalization_translation_table

    def normalize(self, text: str) -> str:
        """
        Normalize text by replacing homoglyphs with their original characters,
        based on the dominant script in the text.

        Args:
            text (str): Text to normalize.

        Returns:
            str: Normalized text.
        """
        dominant_script = self._detect_dominant_script(text)
        normalization_table = self._get_normalization_table_for_script(dominant_script)
        return text.translate(normalization_table)
