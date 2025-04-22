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
    NormalizationStrategies,
    TypesOfHomoglyphs,
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
            TypesOfHomoglyphs
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
        self.reverse_chars_map: Mapping[str, List[str]] = (
            self._create_reverse_chars_map()
        )
        self.base_normalization_map: Mapping[str, List[str]] = (
            self._get_base_normalization_map()
        )
        # self.normalization_translaion_tables = {} # Do not use this. Using a translation table is too naive at times (we may need to consider context for normalization). Instead, keep mappings in a dict and use them when needed.
        self.normalization_translation_maps = {}

    def _load_chars_map(self, ensure_bidirectionality=True) -> Mapping[str, List[str]]:
        """
        Load the character mappings from JSON files.

        Returns:
            Mapping[str, List[str]]: Mapping of characters to their homoglyphs.
        """
        files_mapping = {
            TypesOfHomoglyphs.IDENTICAL: "identical_map.json",
            TypesOfHomoglyphs.CONFUSABLES: "unicode_confusables_map.json",
            TypesOfHomoglyphs.OCR: "ocr_chars_map.json",
            TypesOfHomoglyphs.OCR_REFINED: "ocr_chars_refined_map.json",
        }
        # Load the JSON files
        chars_map = {}
        for homoglyph_type in self.types_of_homoglyphs_to_use:
            with open(
                Path(__file__).parent / files_mapping[homoglyph_type], "r"
            ) as file:
                data = json.load(file)
                for key, values in data.items():
                    if key not in chars_map:
                        # Create an entry for the key if it doesn't exist
                        chars_map[key] = []
                    for v in values:
                        # We only add the values that are not duplicated
                        # We check if the value is in the list of homoglyphs for this character
                        if v not in chars_map[key]:
                            # If it's not there, we add it
                            chars_map[key].append(v)

                        if ensure_bidirectionality:
                            # This means that if we have a mapping from A to B, we also have a mapping from B to A
                            # To ensure that we don't have duplicates, we check if the value is already in the map as a key
                            if v not in chars_map:
                                # If it's not there, we add it because we won't have a duplicate
                                chars_map[v] = [key]
                            else:
                                # If it is, we only add the key if it's not already there
                                if key not in chars_map[v]:
                                    chars_map[v].append(key)

        if self.replace_with_priority:
            # Only keep the first element in the list
            for key, values in chars_map.items():
                chars_map[key] = [values[0]]

        return chars_map

    def _create_reverse_chars_map(self) -> Mapping[str, List[str]]:
        """
        Create a reverse mapping of homoglyphs to original characters.

        Returns:
            Mapping[str, List[str]]: Reverse mapping of homoglyphs to original characters.
        """
        reverse_map: Mapping[str, List[str]] = {}
        for key, values in self.chars_map.items():
            for value in values:
                reverse_map.setdefault(value, []).append(key)

        return reverse_map

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

    def get_original(self, char: str) -> List[str]:
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
        self, char: str, script: Optional[str], block: Optional[str]
    ) -> bool:
        if not block:
            return unicodedataplus.script(char) == script
        else:
            return (
                unicodedataplus.script(char) == script
                and unicodedataplus.block(char) == block
            )

    def _get_base_normalization_map(
        self,
        only_replace_non_normalized: bool = False,
        **kwargs,
    ) -> Mapping[int, str]:
        """
        Generate a base normalization map.

        Args:
            only_replace_non_normalized (bool): If True, only replace non-normalized characters.

        Returns:
            Mapping[int, str]: A normalization map for characters.
        """

        # Generate the normalization table.
        # Create a dictionary with the NFKD entries where the value is in the script we want to normalize (i.e. (this_or_other_script, script) pairs)
        base_normalization_map: Mapping[str, List[str]] = {}
        for key, values in self.reverse_chars_map.items():
            for value in values:
                # Keep the NFKD entries where the value is in the desired script
                if (
                    (
                        # The char we normalize into should be normalized
                        unicodedata.is_normalized("NFKD", value)
                    )
                    and (
                        # If we activate only_replace_non_normalized, then the char we normalize from should NOT be normalized
                        not (
                            unicodedata.is_normalized("NFKD", key)
                            and only_replace_non_normalized
                        )
                    )
                    and (
                        unicodedata.category(key) in self.unicode_categories_to_replace
                    )
                ):
                    base_normalization_map.setdefault(key, []).append(value)

        return base_normalization_map

    def _get_normalization_map_for_script_and_block(
        self,
        script: str,
        block: str = None,
        only_replace_non_normalized=False,
        **kwargs,
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
        if script in self.normalization_translation_maps:
            return self.normalization_translation_maps[script]

        # Generate the normalization table.
        # Create a dictionary with the NFKD entries where the value is in the script we want to normalize (i.e. (this_or_other_script, script) pairs)
        script_normalization_map: Mapping[str, str] = {}
        for key, values in self.reverse_chars_map.items():
            for value in values:
                # Keep the NFKD entries where the value is in the desired script
                if (
                    (
                        # The char we normalize into should be normalized
                        unicodedata.is_normalized("NFKD", value)
                    )
                    and (
                        # If we activate only_replace_non_normalized, then the char we normalize from should NOT be normalized
                        not (
                            unicodedata.is_normalized("NFKD", key)
                            and only_replace_non_normalized
                        )
                    )
                    and (
                        unicodedata.category(key) in self.unicode_categories_to_replace
                    )
                    and (
                        # The result we get after normalizing should be in the appropriate script and block
                        self._is_script_and_block(
                            char=value, script=script, block=block
                        )
                    )
                ):
                    script_normalization_map[key] = value

        self.normalization_translation_maps[script] = script_normalization_map

        return script_normalization_map

    def _translate(self, text: str, map: Mapping[str, List[str]]) -> str:
        """
        Translate the text using the provided mapping.

        Args:
            text (str): Text to translate.
            map (Mapping[str, str]): Mapping of characters to their replacements.

        Returns:
            str: Translated text.
        """
        # Create a translation table
        translation_table = str.maketrans(map)
        # Translate the text using the translation table
        return text.translate(translation_table)

    def _translate_with_context(
        self,
        text: str,
        mapping: Mapping[str, List[str]],
        N: int = 10,
    ) -> str:
        """
        Translate the text using the provided mapping, but also trying to maximize context matches (i.e. casing, etc.). We keep a sliding window and choose the best match for each character that matches most of the properties of the N characters in the window.

        Args:
            text (str): Text to translate.
            mapping (Mapping[str, str]): Mapping of characters to their replacements.
            context (Optional[Mapping[str, str]]): Context for translation.

        Returns:
            str: Translated text.
        """

        PROPERTY_FNS = {
            "script": unicodedataplus.script,
            "block": unicodedataplus.block,
            "category": unicodedataplus.category,
        }

        # Do not use a translation table here - instead, process the text character by character keeping track of all the properties of the characters in the window
        replaced_text = []
        for i, char in enumerate(text):
            # Check if the character is in the mapping
            if char in mapping:
                # Now, we have a set of possibilities - the set of homoglyphs for this character
                possible_chars = [char] + mapping[char]
                # We need to check the context - we will use a sliding window of size N
                # Adjust the context window to always have 10 characters, even at the start or end
                # For char i, we should have i-4 to i + 4
                # To ensure that we always have 10 characters, allow to go out of bounds (i.e. negative indices)
                start = max(0, i - N // 2)
                end = min(len(text), i + N // 2 + 1)
                context_window = text[start:end]
                # If the context window is smaller than N, we need to pad it
                if start == 0:
                    context_window = text[:N]
                elif end == len(text):
                    context_window = text[-N:]
                else:
                    pass # Nothing to do - we have a full window

                # Get the properties of the characters in the context window
                properties = {
                    prop: [PROPERTY_FNS[prop](c) for c in context_window]
                    for prop in PROPERTY_FNS
                }
                # Now, we need to find the character that matches the most properties of the characters in the context window
                scores = []  # List to store scores for each possible character
                for possible_char in possible_chars:
                    score = sum(
                        PROPERTY_FNS[prop](possible_char) == value
                        for prop, values in properties.items()
                        for value in values
                    )
                    scores.append((possible_char, score))
                # Sort the list by score in descending order and pick the best character
                best_char, best_score = max(scores, key=lambda x: x[1])
                # If we found a character that matches the properties, we use it
                if best_char:
                    replaced_text.append(best_char)
                else:
                    # If we didn't find a character that matches the properties, we keep the original character
                    replaced_text.append(char)
            # If the character is not in the mapping, we keep it as is
            else:
                replaced_text.append(char)

        return "".join(replaced_text)

    def normalize(
        self,
        text: str,
        strategy: NormalizationStrategies,
        **kwargs,
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

        if strategy == NormalizationStrategies.DOMINANT_SCRIPT:
            dominant_script = self._detect_dominant_script(text=text)
            normalization_map = self._get_normalization_map_for_script_and_block(
                script=dominant_script, **kwargs
            )
            return self._translate(text, normalization_map)
        if strategy == NormalizationStrategies.DOMINANT_SCRIPT_AND_BLOCK:
            dominant_script = self._detect_dominant_script(text=text)
            dominant_block = self._detect_dominant_block(text=text)
            normalization_map = self._get_normalization_map_for_script_and_block(
                script=dominant_script, block=dominant_block, **kwargs
            )
            return self._translate(text, normalization_map)
        elif strategy == NormalizationStrategies.CONTEXT_AWARE:
            return self._translate_with_context(
                text=text,
                mapping=self.base_normalization_map,
                N=kwargs.get("N", 10),
            )
        elif strategy == NormalizationStrategies.TOKENIZATION:
            raise NotImplementedError()
        else:
            raise NotImplementedError(f"Strategy {strategy} is unknown.")
