"""
HomoglyphReplacer: Core component for homoglyph replacement operations

This module provides the HomoglyphReplacer class, which is responsible for replacing characters
with their homoglyphs and normalizing text containing homoglyphs. The class supports multiple
normalization strategies and homoglyph types.

Author: Aldan Creo (ACMC) <os@acmc.fyi>
License: See LICENSE file in the project root
"""

import json
import logging
import random
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Dict, List, Literal, Mapping, Optional, Set, Tuple

import unicodedataplus

from silverspeak.homoglyphs.utils import (
    _DEFAULT_HOMOGLYPHS_TO_USE,
    _DEFAULT_UNICODE_CATEGORIES_TO_REPLACE,
    NormalizationStrategies,
    TypesOfHomoglyphs,
)

from .unicode_scoring import score_homoglyph_for_context

from .script_block_category_utils import (
    detect_dominant_block,
    detect_dominant_script,
    get_block_counts,
    get_script_counts,
    is_category,
    is_script_and_block,
)

# Import normalization strategies from the normalization package
from .normalization import (
    apply_dominant_script_and_block_strategy,
    apply_dominant_script_strategy,
    apply_language_model_strategy,
    apply_local_context_strategy,
    apply_tokenizer_strategy,
    apply_llm_prompt_strategy,
    apply_spell_check_strategy,
    apply_ngram_strategy,
    apply_ocr_confidence_strategy,
    apply_graph_strategy,
)

logger = logging.getLogger(__name__)


class HomoglyphReplacer:
    """
    A class to replace characters with their homoglyphs and normalize homoglyph text.

    This class is the core component of SilverSpeak, providing functionality to:
    1. Replace characters with their visually similar homoglyphs
    2. Normalize text by replacing homoglyphs with their standard characters
    3. Support various normalization strategies based on Unicode script, block, and context

    Attributes:
        unicode_categories_to_replace (Set[str]): Unicode categories of characters to replace.
        types_of_homoglyphs_to_use (List[TypesOfHomoglyphs]): Types of homoglyphs to use.
        replace_with_priority (bool): Whether to replace with priority.
        random_state (random.Random): Random state for reproducibility.
        chars_map (Dict[str, List[str]]): Mapping of characters to their homoglyphs.
        reverse_chars_map (Dict[str, List[str]]): Reverse mapping of homoglyphs to original characters.
        base_normalization_map (Dict[str, List[str]]): Base table for normalizing text.
        normalization_translation_maps (Dict[str, Dict[str, str]]): Cache of normalization maps by script.
    """

    def __init__(
        self,
        unicode_categories_to_replace: Set[str] = _DEFAULT_UNICODE_CATEGORIES_TO_REPLACE,
        types_of_homoglyphs_to_use: List[TypesOfHomoglyphs] = _DEFAULT_HOMOGLYPHS_TO_USE,
        replace_with_priority: bool = False,
        random_seed: int = 42,
    ):
        """
        Initialize the HomoglyphReplacer.

        Args:
            unicode_categories_to_replace (Set[str]): Unicode categories of characters to replace.
                Defaults to typical letter categories (Ll, Lm, Lo, Lt, Lu).
            types_of_homoglyphs_to_use (List[TypesOfHomoglyphs]): Types of homoglyphs to use.
                Defaults to IDENTICAL, CONFUSABLES, and OCR_REFINED.
            replace_with_priority (bool): Whether to prioritize replacements (use only first homoglyph).
                Defaults to False.
            random_seed (int): Seed for random number generator to ensure reproducibility.
                Defaults to 42.
        """
        self.types_of_homoglyphs_to_use = types_of_homoglyphs_to_use
        self.replace_with_priority = replace_with_priority
        self.unicode_categories_to_replace = unicode_categories_to_replace
        self.chars_map: Dict[str, List[str]] = self._load_chars_map()
        # This object will be used to keep the random state
        self.random_state = random.Random(x=random_seed)
        self.reverse_chars_map: Dict[str, List[str]] = self._create_reverse_chars_map()
        self.base_normalization_map: Dict[str, List[str]] = self._get_base_normalization_map()
        # Cache of normalization maps for different scripts
        self.normalization_translation_maps: Dict[str, Dict[str, str]] = {}
        logger.debug(f"HomoglyphReplacer initialized with {len(self.chars_map)} character mappings")

    def _load_chars_map(self, ensure_bidirectionality=True) -> Dict[str, List[str]]:
        """
        Load the character mappings from JSON files.

        This method loads homoglyph maps from the appropriate JSON files based on the
        specified homoglyph types to use. It can also ensure bidirectionality of mappings.

        Args:
            ensure_bidirectionality (bool, optional): If True, ensures that for each mapping
                from character A to B, there is also a mapping from B to A. Defaults to True.

        Returns:
            Dict[str, List[str]]: Mapping of characters to their homoglyphs.
        """
        files_mapping = {
            TypesOfHomoglyphs.IDENTICAL: "identical_map.json",
            TypesOfHomoglyphs.CONFUSABLES: "unicode_confusables_map.json",
            TypesOfHomoglyphs.OCR: "ocr_chars_map.json",
            TypesOfHomoglyphs.OCR_REFINED: "ocr_chars_refined_map.json",
        }
        # Load the JSON files
        chars_map: Dict[str, List[str]] = {}
        for homoglyph_type in self.types_of_homoglyphs_to_use:
            with open(Path(__file__).parent / files_mapping[homoglyph_type], "r") as file:
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
            key: [value for value in values if len(value) == 1] for key, values in chars_map.items() if len(key) == 1
        }

        return chars_map

    def _create_reverse_chars_map(self) -> Dict[str, List[str]]:
        """
        Create a reverse mapping of homoglyphs to original characters.

        This method creates a reverse mapping from the chars_map, so that for each
        homoglyph, we can find the original character(s) it could represent.
        This is essential for normalization operations.

        Returns:
            Dict[str, List[str]]: Reverse mapping of homoglyphs to original characters.
        """
        reverse_map: Dict[str, List[str]] = {}
        for key, values in self.chars_map.items():
            for value in values:
                reverse_map.setdefault(value, []).append(key)

        return reverse_map

    def _get_base_normalization_map(
        self,
        only_replace_non_normalized: bool = False,
        **kwargs,
    ) -> Dict[str, List[str]]:
        """
        Generate a base normalization map for all homoglyphs.

        This method creates a base mapping for normalizing homoglyphs, considering
        whether they are already in a normalized form (NFKD).

        Args:
            only_replace_non_normalized (bool): If True, only replace characters that
                aren't already in normalized form. Defaults to False.
            **kwargs: Additional keyword arguments.

        Returns:
            Dict[str, List[str]]: A normalization map for characters.
        """
        # Generate the normalization table.
        # Create a dictionary with the NFKD entries where the value is in the script we want to normalize (i.e. (this_or_other_script, script) pairs)
        base_normalization_map: Dict[str, List[str]] = {}
        for key, values in self.reverse_chars_map.items():
            for value in values:
                # Keep the NFKD entries where the value is in the desired script
                if (
                    # The char we normalize into should be normalized
                    unicodedata.is_normalized("NFKD", value)
                ) and (
                    # If we activate only_replace_non_normalized, then the char we normalize from should NOT be normalized
                    not (unicodedata.is_normalized("NFKD", key) and only_replace_non_normalized)
                ):
                    base_normalization_map.setdefault(key, []).append(value)

        return base_normalization_map

    def get_normalization_map_for_script_block_and_category(
        self,
        script: str,
        block: Optional[str] = None,
        unicode_categories_to_replace: Set[str] = _DEFAULT_UNICODE_CATEGORIES_TO_REPLACE,
        only_replace_non_normalized=False,
        **kwargs,
    ) -> Dict[str, str]:
        """
        Generate a normalization map for a specific script and block.

        This method creates a mapping for normalizing homoglyphs based on a specific
        Unicode script and optional block, considering character categories.

        Args:
            script (str): The target Unicode script (e.g., 'Latin', 'Cyrillic').
            block (Optional[str], optional): The target Unicode block. Defaults to None.
            unicode_categories_to_replace (Set[str]): Unicode categories of characters to
                consider for replacement. Defaults to _DEFAULT_UNICODE_CATEGORIES_TO_REPLACE.
            only_replace_non_normalized (bool): If True, only replace characters that
                aren't already in normalized form. Defaults to False.
            **kwargs: Additional keyword arguments.

        Returns:
            Dict[str, str]: A normalization map for the specified script and block.
        """
        # Check if it's in our in-memory cache
        if script in self.normalization_translation_maps:
            return self.normalization_translation_maps[script]

        # Generate the normalization table.
        # Create a dictionary with the NFKD entries where the value is in the script we want to normalize (i.e. (this_or_other_script, script) pairs)
        script_normalization_map: Dict[str, str] = {}
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
                        not (unicodedata.is_normalized("NFKD", key) and only_replace_non_normalized)
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

    def get_homoglyph_for_char(
        self,
        char: str,
        same_script: bool = False,
        same_block: bool = False,
        dominant_script: Optional[str] = None,
        dominant_block: Optional[str] = None,
        context: Optional[str] = None,
        context_window_size: int = 10,
    ) -> Optional[str]:
        """
        Get a homoglyph replacement for a character, considering context.

        This method selects an optimal homoglyph replacement for a character by analyzing
        the surrounding context and choosing a replacement that best matches the Unicode
        properties of the surrounding text.

        Args:
            char (str): The character to replace with a homoglyph.
            same_script (bool): Whether to use only homoglyphs from the same Unicode script.
                Defaults to False.
            same_block (bool): Whether to use only homoglyphs from the same Unicode block.
                Defaults to False.
            dominant_script (Optional[str]): The dominant script to use for filtering.
                If None and same_script is True, will be automatically detected.
            dominant_block (Optional[str]): The dominant block to use for filtering.
                If None and same_block is True, will be automatically detected.
            context (Optional[str]): The surrounding text context for property matching.
                If None, context-based selection will not be used.
            context_window_size (int): Size of the context window when context is provided.
                Defaults to 10.

        Returns:
            Optional[str]: A homoglyph replacement, or None if no suitable replacement is found.
        """
        if not char or char not in self.chars_map or not self.chars_map[char]:
            return None

        # Get all possible homoglyph replacements for this character
        all_homoglyphs = self.chars_map[char]

        # Filter by script if requested
        if same_script and dominant_script:
            all_homoglyphs = [h for h in all_homoglyphs if unicodedataplus.script(h) == dominant_script]

        # Filter by block if requested
        if same_block and dominant_block:
            all_homoglyphs = [h for h in all_homoglyphs if unicodedataplus.block(h) == dominant_block]

        # If no homoglyphs remain after filtering, return None
        if not all_homoglyphs:
            return None

        # If context is provided, use it for property matching
        if context:
            # Dictionary of Unicode property extraction functions
            PROPERTY_FNS = {
                "script": unicodedataplus.script,
                "block": unicodedataplus.block,
                "category": unicodedata.category,
                "bidirectional": unicodedata.bidirectional,
                "east_asian_width": unicodedata.east_asian_width,
            }

            # Define property weights for scoring
            PROPERTY_WEIGHTS = {
                "block": 10,  # Highest priority
                "category": 5,  # Medium-high priority
                "script": 3,  # Medium priority
                "bidirectional": 2,  # Medium-low priority
                "east_asian_width": 1,  # Low priority
            }

            # Score each possible homoglyph
            property_scores = []
            for homoglyph in all_homoglyphs:
                if homoglyph == char:
                    continue  # Skip the character itself

                score = 0

                # Extract properties of the homoglyph
                homoglyph_props = {prop: PROPERTY_FNS[prop](homoglyph) for prop in PROPERTY_FNS}

                # Calculate context property matches
                for ctx_char in context:
                    for prop, weight in PROPERTY_WEIGHTS.items():
                        if PROPERTY_FNS[prop](ctx_char) == homoglyph_props[prop]:
                            score += weight

                # Add bonus for specific property combinations in context
                for ctx_char in context:
                    # Block + script combination bonus
                    if (
                        PROPERTY_FNS["block"](ctx_char) == homoglyph_props["block"]
                        and PROPERTY_FNS["script"](ctx_char) == homoglyph_props["script"]
                    ):
                        score += 3

                    # Block + category combination bonus
                    if (
                        PROPERTY_FNS["block"](ctx_char) == homoglyph_props["block"]
                        and PROPERTY_FNS["category"](ctx_char) == homoglyph_props["category"]
                    ):
                        score += 2

                    # Script + category combination bonus
                    if (
                        PROPERTY_FNS["script"](ctx_char) == homoglyph_props["script"]
                        and PROPERTY_FNS["category"](ctx_char) == homoglyph_props["category"]
                    ):
                        score += 2

                property_scores.append((homoglyph, score))

            # Sort by score (highest first) and select the best homoglyph
            if property_scores:
                return max(property_scores, key=lambda x: x[1])[0]

        # If no context or no scored homoglyphs, just return a random one
        return self.random_state.choice(all_homoglyphs)

    def score_homoglyph_for_context(
        self,
        homoglyph: str,
        char: str,
        context: str,
        context_window_size: int = 10,
    ) -> float:
        """
        Score a homoglyph based on how well it matches the surrounding context.

        This method evaluates a potential homoglyph replacement by comparing its Unicode
        properties with those of the surrounding characters, assigning a score that
        reflects how well the homoglyph fits within the given context.

        Args:
            homoglyph (str): The homoglyph to evaluate.
            char (str): The original character that would be replaced.
            context (str): The surrounding text context for property matching.
            context_window_size (int): Size of the context window. Defaults to 10.

        Returns:
            float: A score indicating how well the homoglyph matches the context.
                Higher scores indicate better matches.
        """
        # Delegate to the standalone function from unicode_scoring module
        return score_homoglyph_for_context(
            homoglyph=homoglyph,
            char=char,
            context=context,
            context_window_size=context_window_size
        )

    def normalize(
        self,
        text: str,
        strategy: NormalizationStrategies,
        **kwargs,
    ) -> str:
        """
        Normalize text by replacing homoglyphs with their standard characters.

        This method applies the specified normalization strategy to convert text containing
        homoglyphs back to standard characters. Different strategies consider different
        aspects like dominant script, block, local context, tokenization, or language model.

        Args:
            text (str): Text to normalize.
            strategy (NormalizationStrategies): The normalization strategy to apply.
            **kwargs: Additional arguments passed to the specific strategy implementation.

        Returns:
            str: Normalized text with homoglyphs replaced by standard characters.

        Raises:
            NotImplementedError: If the specified strategy is unknown.
        """
        # If the text is empty, return it as is
        if not text:
            return text

        if strategy == NormalizationStrategies.DOMINANT_SCRIPT:
            return apply_dominant_script_strategy(replacer=self, text=text, **kwargs)

        elif strategy == NormalizationStrategies.DOMINANT_SCRIPT_AND_BLOCK:
            return apply_dominant_script_and_block_strategy(replacer=self, text=text, **kwargs)

        elif strategy == NormalizationStrategies.LOCAL_CONTEXT:
            return apply_local_context_strategy(text=text, normalization_map=self.base_normalization_map, **kwargs)

        elif strategy == NormalizationStrategies.TOKENIZATION:
            return apply_tokenizer_strategy(text=text, mapping=self.base_normalization_map, **kwargs)

        elif strategy == NormalizationStrategies.LANGUAGE_MODEL:
            try:
                import transformers

                model_name = kwargs.get("model_name", "bert-base-multilingual-cased")

                # Only try to load the model if not provided in kwargs
                if "language_model" not in kwargs or "tokenizer" not in kwargs:
                    try:
                        lm = transformers.AutoModelForMaskedLM.from_pretrained(model_name)
                        tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
                        kwargs["language_model"] = lm
                        kwargs["tokenizer"] = tokenizer
                    except Exception as e:
                        logger.error(f"Failed to load language model: {e}")

                return apply_language_model_strategy(text=text, mapping=self.base_normalization_map, **kwargs)
            except ImportError:
                logger.error("Transformers library not available, falling back to dominant script strategy")
                return apply_dominant_script_strategy(replacer=self, text=text, **kwargs)
            except Exception as e:
                logger.error(f"Error applying language model strategy: {e}")
                logger.warning("Falling back to dominant script strategy")
                return apply_dominant_script_strategy(replacer=self, text=text, **kwargs)

        elif strategy == NormalizationStrategies.LLM_PROMPT:
            try:
                import transformers

                model_name = kwargs.get("model_name", "google/gemma-2-1b-it")

                return apply_llm_prompt_strategy(text=text, mapping=self.base_normalization_map, **kwargs)
            except ImportError:
                logger.error("Transformers library not available, falling back to dominant script strategy")
                return apply_dominant_script_strategy(replacer=self, text=text, **kwargs)
            except Exception as e:
                logger.error(f"Error applying LLM prompt strategy: {e}")
                logger.warning("Falling back to dominant script strategy")
                return apply_dominant_script_strategy(replacer=self, text=text, **kwargs)

        elif strategy == NormalizationStrategies.SPELL_CHECK:
            try:
                language = kwargs.get("language", "en")
                logger.info(f"Using spell checking strategy with language: {language}")

                return apply_spell_check_strategy(text=text, mapping=self.base_normalization_map, **kwargs)
            except ImportError as e:
                logger.error(f"Required spell checking libraries not available: {e}")
                logger.warning("Install spell checking dependencies using: poetry install --with spell-check")
                logger.warning("Falling back to local context strategy")
                return apply_local_context_strategy(text=text, normalization_map=self.base_normalization_map, **kwargs)
            except Exception as e:
                logger.error(f"Error applying spell check strategy: {e}")
                logger.warning("Falling back to local context strategy")
                return apply_local_context_strategy(text=text, normalization_map=self.base_normalization_map, **kwargs)

        elif strategy == NormalizationStrategies.NGRAM:
            try:
                logger.info("Using n-gram frequency strategy")

                return apply_ngram_strategy(text=text, mapping=self.base_normalization_map, **kwargs)
            except Exception as e:
                logger.error(f"Error applying n-gram strategy: {e}")
                logger.warning("Falling back to local context strategy")
                return apply_local_context_strategy(text=text, normalization_map=self.base_normalization_map, **kwargs)

        elif strategy == NormalizationStrategies.OCR_CONFIDENCE:
            try:
                logger.info("Using OCR confidence strategy")

                return apply_ocr_confidence_strategy(text=text, mapping=self.base_normalization_map, **kwargs)
            except ImportError as e:
                logger.error(f"Required OCR libraries not available: {e}")
                logger.warning("Install OCR dependencies using: pip install pytesseract pillow")
                logger.warning("Falling back to local context strategy")
                return apply_local_context_strategy(text=text, normalization_map=self.base_normalization_map, **kwargs)
            except Exception as e:
                logger.error(f"Error applying OCR confidence strategy: {e}")
                logger.warning("Falling back to local context strategy")
                return apply_local_context_strategy(text=text, normalization_map=self.base_normalization_map, **kwargs)

        elif strategy == NormalizationStrategies.GRAPH_BASED:
            try:
                logger.info("Using graph-based strategy")

                return apply_graph_strategy(text=text, mapping=self.base_normalization_map, **kwargs)
            except Exception as e:
                logger.error(f"Error applying graph-based strategy: {e}")
                logger.warning("Falling back to local context strategy")
                return apply_local_context_strategy(text=text, normalization_map=self.base_normalization_map, **kwargs)

        else:
            raise NotImplementedError(f"Strategy {strategy} is unknown.")
