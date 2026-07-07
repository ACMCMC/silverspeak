import json
import logging
import random
from pathlib import Path
from typing import Dict, List, Optional, Set

from silverspeak.homoglyphs.script_block_category_utils import char_block, char_script
from silverspeak.homoglyphs.unicode_scoring import score_homoglyphs_for_context_window
from silverspeak.homoglyphs.utils import TypesOfHomoglyphs

logger = logging.getLogger(__name__)


class HomoglyphReplacer:
    def __init__(
        self,
        unicode_categories_to_replace: Set[str],
        types_of_homoglyphs_to_use: List[TypesOfHomoglyphs],
        replace_with_priority: bool,
        random_seed: int,
    ):
        self.types_of_homoglyphs_to_use = types_of_homoglyphs_to_use
        self.replace_with_priority = replace_with_priority
        self.unicode_categories_to_replace = unicode_categories_to_replace
        self.chars_map: Dict[str, List[str]] = self._load_chars_map(ensure_bidirectionality=True)
        self.random_state = random.Random(x=random_seed)

    def _load_chars_map(self, ensure_bidirectionality: bool) -> Dict[str, List[str]]:
        files_mapping = {
            TypesOfHomoglyphs.IDENTICAL: "identical_map.json",
            TypesOfHomoglyphs.CONFUSABLES: "unicode_confusables_map.json",
            TypesOfHomoglyphs.OCR: "ocr_chars_map.json",
            TypesOfHomoglyphs.OCR_REFINED: "ocr_chars_refined_map.json",
        }
        chars_map: Dict[str, List[str]] = {}
        for homoglyph_type in self.types_of_homoglyphs_to_use:
            with open(Path(__file__).parent / files_mapping[homoglyph_type], "r") as file:
                data = json.load(file)
                for key, values in data.items():
                    if key not in chars_map:
                        chars_map[key] = []
                    for v in values:
                        if v not in chars_map[key]:
                            chars_map[key].append(v)
                        if ensure_bidirectionality:
                            if v not in chars_map:
                                chars_map[v] = [key]
                            elif key not in chars_map[v]:
                                chars_map[v].append(key)

        if self.replace_with_priority:
            for key, values in chars_map.items():
                chars_map[key] = [values[0]]

        return {
            key: [value for value in values if len(value) == 1] for key, values in chars_map.items() if len(key) == 1
        }

    def get_homoglyph_for_char(
        self,
        char: str,
        same_script: bool,
        same_block: bool,
        dominant_script: Optional[str],
        dominant_block: Optional[str],
        context: Optional[str],
        context_window_size: int,
    ) -> Optional[str]:
        if not char or char not in self.chars_map or not self.chars_map[char]:
            return None

        all_homoglyphs = self.chars_map[char]
        if same_script and dominant_script:
            all_homoglyphs = [h for h in all_homoglyphs if char_script(char=h) == dominant_script]
        if same_block and dominant_block:
            all_homoglyphs = [h for h in all_homoglyphs if char_block(char=h) == dominant_block]
        if not all_homoglyphs:
            return None

        if context:
            property_scores = []
            for homoglyph in all_homoglyphs:
                if homoglyph == char:
                    continue
                score_dict = score_homoglyphs_for_context_window(
                    homoglyph=homoglyph,
                    char=char,
                    context=context,
                    context_window_size=context_window_size,
                )
                property_scores.append((homoglyph, score_dict.get("total_score", 0.0)))
            if property_scores:
                return max(property_scores, key=lambda x: x[1])[0]

        return self.random_state.choice(all_homoglyphs)
