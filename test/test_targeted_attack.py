#!/usr/bin/env python
"""
Unit tests for the targeted homoglyph attack implementation.

This module contains tests that verify the functionality of the targeted
homoglyph attack, including context-aware homoglyph selection.

Author: Aldan Creo (ACMC) <os@acmc.fyi>
"""

import re
from typing import Dict, List, Optional, Pattern, Union

import pytest
import unicodedataplus

from silverspeak.homoglyphs import HomoglyphReplacer
from silverspeak.homoglyphs.attacks.targeted_attack import targeted_attack
from silverspeak.homoglyphs.utils import (
    _DEFAULT_HOMOGLYPHS_TO_USE,
    _DEFAULT_UNICODE_CATEGORIES_TO_REPLACE,
    TypesOfHomoglyphs,
)


@pytest.fixture
def replacer():
    return HomoglyphReplacer(
        unicode_categories_to_replace=_DEFAULT_UNICODE_CATEGORIES_TO_REPLACE,
        types_of_homoglyphs_to_use=_DEFAULT_HOMOGLYPHS_TO_USE,
        replace_with_priority=False,
        random_seed=42,
    )


def test_get_homoglyph_for_char_basic(replacer):
    result = replacer.get_homoglyph_for_char(
        char="a",
        same_script=False,
        same_block=False,
        dominant_script=None,
        dominant_block=None,
        context=None,
        context_window_size=10,
    )
    assert result is not None
    assert result != "a"

    result = replacer.get_homoglyph_for_char(
        char="丛",
        same_script=False,
        same_block=False,
        dominant_script=None,
        dominant_block=None,
        context=None,
        context_window_size=10,
    )
    assert result is None


def test_get_homoglyph_for_char_with_script_constraint(replacer):
    result = replacer.get_homoglyph_for_char(
        char="a",
        same_script=True,
        same_block=False,
        dominant_script="Latin",
        dominant_block=None,
        context=None,
        context_window_size=10,
    )
    if result:
        assert unicodedataplus.script(result) == "Latin"

    result = replacer.get_homoglyph_for_char(
        char="a",
        same_script=True,
        same_block=False,
        dominant_script="Cyrillic",
        dominant_block=None,
        context=None,
        context_window_size=10,
    )
    if result:
        assert unicodedataplus.script(result) == "Cyrillic"


def test_get_homoglyph_for_char_with_context(replacer):
    result_latin = replacer.get_homoglyph_for_char(
        char="a",
        same_script=False,
        same_block=False,
        dominant_script=None,
        dominant_block=None,
        context="hello world",
        context_window_size=10,
    )
    result_cyrillic = replacer.get_homoglyph_for_char(
        char="a",
        same_script=False,
        same_block=False,
        dominant_script=None,
        dominant_block=None,
        context="привет мир",
        context_window_size=10,
    )
    if result_latin and result_cyrillic:
        assert result_latin != result_cyrillic


def test_targeted_attack_basic():
    """Test basic functionality of the targeted attack."""
    text = "hello world"

    result = targeted_attack(text, percentage=0.3)  # Replace 30% of characters

    # Check that the text has been modified
    assert result != text

    # Count how many characters are different
    diff_count = sum(1 for a, b in zip(text, result) if a != b)

    # Should be around 30% of the characters in "hello world" (11 chars)
    # but not exact due to random selection and eligible character constraints
    assert 0 < diff_count <= 4


def test_targeted_attack_with_percentage():
    """Test targeted attack with different percentages."""
    text = "hello world"

    # Test with 10% replacement rate
    result_low = targeted_attack(text, percentage=0.1)

    # Test with 50% replacement rate
    result_high = targeted_attack(text, percentage=0.5, random_seed=42)

    # Check that the text has been modified in both cases
    assert result_low != text
    assert result_high != text

    # Count differences
    diff_count_low = sum(1 for a, b in zip(text, result_low) if a != b)
    diff_count_high = sum(1 for a, b in zip(text, result_high) if a != b)

    # Higher percentage should result in more replacements
    # Given the random selection, we can't guarantee exact counts,
    # but we can expect higher percentage to generally yield more changes
    assert diff_count_low <= diff_count_high


def test_targeted_attack_with_zero_percentage():
    """Test targeted attack with zero percentage."""
    text = "Hello World"

    result = targeted_attack(text, percentage=0.0)

    # With 0% replacement, text should be unchanged
    assert result == text

    # Test with very small percentage
    result_small = targeted_attack(text, percentage=0.01, random_seed=42)

    # Small percentage on short text might not change anything
    # or might change at most 1 character
    differences = sum(1 for a, b in zip(text, result_small) if a != b)
    assert differences <= 1


def test_targeted_attack_consistency():
    """Test targeted attack produces consistent results with the same seed."""
    text = "Hello World"
    random_seed = 42

    # Run the attack twice with the same seed
    result_1 = targeted_attack(text, percentage=0.3, random_seed=random_seed)

    result_2 = targeted_attack(text, percentage=0.3, random_seed=random_seed)

    # Check that the text has been modified
    assert result_1 != text

    # Check that results are consistent with the same seed
    assert result_1 == result_2


def test_targeted_attack_with_extreme_percentage():
    """Test targeted attack with high percentage value."""
    text = "Hello World"

    result = targeted_attack(text, percentage=1.0, random_seed=42)  # Try to replace all characters

    # Check that the text has been modified
    assert result != text

    # Count replacements
    differences = sum(1 for a, b in zip(text, result) if a != b)

    # Since not all characters might have homoglyphs available,
    # we can't guarantee all characters will be replaced
    # But a significant portion should be
    assert differences > 0


def test_targeted_attack_different_percentages():
    """Test targeted attack with different replacement percentages."""
    text = "This is a sample text for testing different percentages."

    # Set a fixed random seed for consistency
    random_seed = 42

    # Try with small percentage
    result_small = targeted_attack(text, percentage=0.1, random_seed=random_seed)

    # Try with large percentage
    result_large = targeted_attack(text, percentage=0.5, random_seed=random_seed)

    # Both should be different from the original
    assert result_small != text
    assert result_large != text

    # Count the number of character differences
    diff_small = sum(1 for a, b in zip(text, result_small) if a != b)
    diff_large = sum(1 for a, b in zip(text, result_large) if a != b)

    # Larger percentage should result in more character replacements
    # Note: This assumes there are enough replaceable characters in the text
    assert diff_large > diff_small
