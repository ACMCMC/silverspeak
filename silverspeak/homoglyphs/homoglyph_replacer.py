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

from .normalization_strategies import (
    apply_dominant_script_strategy,
    apply_dominant_script_and_block_strategy,
    apply_local_context_strategy,
    apply_tokenizer_strategy,
    apply_language_model_strategy,
)

from .script_block_category_utils import (
    get_script_counts,
    detect_dominant_script,
    get_block_counts,
    detect_dominant_block,
    is_script_and_block,
    is_category,
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

        # TODO: Support multi-character homoglyphs
        # For now, we only support single character homoglyphs
        # Filter out entries with more than one character
        chars_map = {
            key: [value for value in values if len(value) == 1]
            for key, values in chars_map.items()
            if len(key) == 1
        }

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
                    # The char we normalize into should be normalized
                    unicodedata.is_normalized("NFKD", value)
                ) and (
                    # If we activate only_replace_non_normalized, then the char we normalize from should NOT be normalized
                    not (
                        unicodedata.is_normalized("NFKD", key)
                        and only_replace_non_normalized
                    )
                ):
                    base_normalization_map.setdefault(key, []).append(value)

        return base_normalization_map

    def get_normalization_map_for_script_block_and_category(
        self,
        script: str,
        block: str = None,
        unicode_categories_to_replace: Set[
            str
        ] = _DEFAULT_UNICODE_CATEGORIES_TO_REPLACE,
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
                        is_category(
                            text=key,
                            category=unicode_categories_to_replace,
                        )
                    )
                    and (
                        # The result we get after normalizing should be in the appropriate script and block
                        is_script_and_block(text=value, script=script, block=block)
                    )
                ):
                    script_normalization_map[key] = value

        self.normalization_translation_maps[script] = script_normalization_map

        return script_normalization_map

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
            return apply_dominant_script_strategy(replacer=self, text=text, **kwargs)
        elif strategy == NormalizationStrategies.DOMINANT_SCRIPT_AND_BLOCK:
            return apply_dominant_script_and_block_strategy(
                replacer=self, text=text, **kwargs
            )
        elif strategy == NormalizationStrategies.LOCAL_CONTEXT:
            return apply_local_context_strategy(
                normalization_map=self.base_normalization_map, text=text, **kwargs
            )
        elif strategy == NormalizationStrategies.TOKENIZATION:
            return apply_tokenizer_strategy(
                text=text,
                mapping=self.base_normalization_map,
            )
        elif strategy == NormalizationStrategies.LANGUAGE_MODEL:
            import transformers

            lm = transformers.AutoModelForMaskedLM.from_pretrained(
                "google-bert/bert-base-multilingual-cased"
            )
            tokenizer = transformers.AutoTokenizer.from_pretrained(
                "google-bert/bert-base-multilingual-cased"
            )
            return apply_language_model_strategy(
                text=text,
                mapping=self.base_normalization_map,
                language_model=lm,
                tokenizer=tokenizer,
            )
        else:
            raise NotImplementedError(f"Strategy {strategy} is unknown.")
