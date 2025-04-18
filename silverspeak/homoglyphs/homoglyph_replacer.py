import json
import random
import unicodedata
from pathlib import Path
from typing import List, Literal, Mapping, Optional, Set, Tuple
import logging

import unicodedataplus
from collections import Counter

from silverspeak.homoglyphs.utils import (
    _DEFAULT_HOMOGLYPHS_TO_USE,
    _DEFAULT_UNICODE_CATEGORIES_TO_REPLACE,
    TYPES_OF_HOMOGLYPHS,
)


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
        unicode_categories_to_replace: Set[
            str
        ] = _DEFAULT_UNICODE_CATEGORIES_TO_REPLACE,
        types_of_homoglyphs_to_use: List[
            TYPES_OF_HOMOGLYPHS
        ] = _DEFAULT_HOMOGLYPHS_TO_USE,
        replace_with_priority: bool = False,
        random_seed: int = 42,
    ):
        """
        Initialize the HomoglyphReplacer.

        Args:
            unicode_categories_to_replace (Set[str]): Unicode categories of characters to replace.
            types_of_homoglyphs_to_use (List[TYPES_OF_HOMOGLYPHS]): Types of homoglyphs to use.
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
            "ocr_refined": "ocr_chars_refined_map.json",
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

    def _create_reverse_chars_map(self) -> Mapping[str, str]:
        """
        Create a reverse mapping of homoglyphs to original characters.

        Returns:
            Mapping[str, str]: Reverse mapping of homoglyphs to original characters.
        """
        reverse_map: Mapping[str, str] = {}
        for key, values in self.chars_map.items():
            for value in values:
                # If there's already a key for this value, don't overwrite it - the first one we find is the highest priority
                if value not in reverse_map:
                    reverse_map[value] = key

        return reverse_map

    def _create_base_normalization_map(self) -> Mapping[str, str]:
        """
        Create a base normalization map for the most common homoglyphs.

        Returns:
            Mapping[str, str]: Base normalization map.
        """
        base_normalization_map: Mapping[str, str] = {
            key: value
            for key, value in self.reverse_chars_map.items()
            if not unicodedata.is_normalized("NFKD", key)
            and unicodedata.category(key) in self.unicode_categories_to_replace
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

    def _get_script_counts(self, text: str) -> Mapping[str, int]:
        """
        Count the occurrences of each script in the text.

        Args:
            text (str): Text to analyze.

        Returns:
            Mapping[str, int]: Counts of characters in each script.
        """

        script_counts = Counter(unicodedataplus.script(char) for char in text)
        return dict(script_counts)

    def _detect_dominant_script(self, text: str) -> str:
        """
        Detect the dominant script in the text.

        Args:
            text (str): Text to analyze.

        Returns:
            str: Dominant script in the text.
        """
        script_counts = self._get_script_counts(text=text)
        total_count = sum(script_counts.values())
        dominant_script = max(script_counts, key=script_counts.get)
        if script_counts[dominant_script] / total_count < 0.75:
            logging.warning(
                f"The dominant script '{dominant_script}' comprises less than 75% of the total character count. This is unusual, as most texts predominantly consist of characters from a single script. Proceed with caution, as this may affect the reliability of the analysis."
            )
        return dominant_script

    def _get_block_counts(self, text: str) -> Mapping[str, int]:
        """
        Count the number of characters in each Unicode block in the text.

        Args:
            text (str): Text to analyze.

        Returns:
            Mapping[str, int]: Counts of characters in each Unicode block.
        """
        block_counts = Counter(unicodedataplus.block(char) for char in text)
        return dict(block_counts)

    def _detect_dominant_block(self, text: str) -> str:
        """
        Detect the dominant Unicode block in the text.

        Args:
            text (str): Text to analyze.

        Returns:
            str: Dominant Unicode block in the text.
        """
        block_counts = self._get_block_counts(text=text)
        total_count = sum(block_counts.values())
        dominant_block = max(block_counts, key=block_counts.get)
        if block_counts[dominant_block] / total_count < 0.75:
            logging.warning(
                f"The dominant Unicode block '{dominant_block}' comprises less than 75% of the total character count. This is unusual, as most texts predominantly consist of characters from a single block. Proceed with caution, as this may affect the reliability of the analysis."
            )
        return dominant_block

    def _is_script_and_block(
        self, char: str, script: str, block: Optional[str]
    ) -> bool:
        if not block:
            return unicodedataplus.script(char) == script
        else:
            return (
                unicodedataplus.script(char) == script
                and unicodedataplus.block(char) == block
            )

    def _get_normalization_table_for_script_and_block(
        self, script: str, block: str = None, only_replace_non_normalized=False, **kwargs
    ) -> Mapping[int, str]:
        """
        Generate a normalization table for a specific script.

        Args:
            script (str): The script for which the normalization table is generated.
            only_replace_non_normalized (bool): If True, only replace non-normalized characters.

        Returns:
            Mapping[int, str]: A translation table for normalizing text based on the specified script.
        """
        # Check if it's in our in-memory cache
        if script in self.normalization_translation_tables:
            return self.normalization_translation_tables[script]

        # Generate the normalization table.
        # Create a dictionary with the NFKD entries where the value is in the script we want to normalize (i.e. (this_or_other_script, script) pairs)
        script_normalization_map: Mapping[str, str] = {
            key: value
            for key, value in self.reverse_chars_map.items()
            # Keep the NFKD entries where the value is in the desired script
            if
            # The char we normalize into should be normalized
            unicodedata.is_normalized("NFKD", value)
            and (
                # If we activate only_replace_non_normalized, then the char we normalize from should NOT be normalized
                not (
                    unicodedata.is_normalized("NFKD", key)
                    and only_replace_non_normalized
                )
            )
            and unicodedata.category(key) in self.unicode_categories_to_replace
            # The result we get after normalizing should be in the appropriate script and block
            and self._is_script_and_block(char=value, script=script, block=block)
        }

        script_normalization_translation_table: Mapping[int, str] = str.maketrans(
            script_normalization_map
        )

        self.normalization_translation_tables[script] = (
            script_normalization_translation_table
        )

        return script_normalization_translation_table

    def normalize(
        self,
        text: str,
        strategy: Literal[
            "dominant_script", "dominant_script_and_block", "tokenization"
        ] = "dominant_script_and_block",
        **kwargs
    ) -> str:
        """
        Normalize text by replacing homoglyphs with their original characters,
        based on the dominant script in the text.

        Args:
            text (str): Text to normalize.

        Returns:
            str: Normalized text.
        """
        # If the text is empty, return it as is
        if not text:
            return text

        if strategy == "dominant_script":
            dominant_script = self._detect_dominant_script(text=text)
            normalization_table = self._get_normalization_table_for_script_and_block(
                script=dominant_script, **kwargs
            )
            return text.translate(normalization_table)
        if strategy == "dominant_script_and_block":
            dominant_script = self._detect_dominant_script(text=text)
            dominant_block = self._detect_dominant_block(text=text)
            normalization_table = self._get_normalization_table_for_script_and_block(
                script=dominant_script, block=dominant_block, **kwargs
            )
            return text.translate(normalization_table)
        elif strategy == "tokenization":
            raise NotImplementedError()
        else:
            raise NotImplementedError(f"Strategy {strategy} is unknown.")
