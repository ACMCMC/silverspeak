#!/usr/bin/env python
"""
Example showing the targeted homoglyph attack in action.

This script demonstrates how to use the context-aware targeted homoglyph attack,
which replaces characters based on a scoring system that evaluates which homoglyph
replacement best matches the context of the text.

Author: Aldan Creo (ACMC) <os@acmc.fyi>
"""

from silverspeak.homoglyphs import targeted_attack
import re

def print_comparison(original: str, modified: str, title: str = "Comparison"):
    """Print a side-by-side comparison of original and modified text."""
    print("\n" + "=" * 60)
    print(f"{title}:")
    print("-" * 60)
    print(f"Original: {original}")
    print(f"Modified: {modified}")
    
    # Count different characters
    different_chars = sum(1 for a, b in zip(original, modified) if a != b)
    print(f"Different characters: {different_chars}")
    print("=" * 60)

# Example 1: Replace 10% of characters (default)
text = "Hello world! This is a test of the targeted homoglyph attack."

print("\nExample 1: Replace 10% of characters (default)")
modified_text = targeted_attack(
    text=text,
    percentage=0.1,
    context_window_size=8
)
print_comparison(text, modified_text, "10% character replacement with context")

# Example 2: Replace 5% of characters
text = "Hello world! This is a test of the targeted homoglyph attack."

print("\nExample 2: Replace 5% of characters")
modified_text = targeted_attack(
    text=text,
    percentage=0.05,
    context_window_size=8
)
print_comparison(text, modified_text, "5% character replacement with context")

# Example 3: Replace 20% of characters
text = "Hello world! This is a test of the targeted homoglyph attack."

print("\nExample 3: Replace 20% of characters")
modified_text = targeted_attack(
    text=text,
    percentage=0.2,
    context_window_size=8
)
print_comparison(text, modified_text, "20% character replacement with context")

# Example 4: Using a larger context window
text = "Hello world! This is a test of the targeted homoglyph attack."

print("\nExample 4: Using a larger context window")
modified_text = targeted_attack(
    text=text,
    percentage=0.15,
    context_window_size=20  # Use a larger context window
)
print_comparison(text, modified_text, "15% replacement with larger context window")

# Example 5: Text with mixed scripts
text = "Hello мир! こんにちは world!"  # Mixed Latin, Cyrillic, and Japanese

print("\nExample 5: Mixed script text")
modified_text = targeted_attack(
    text=text,
    percentage=0.15,
    context_window_size=8
)
print_comparison(text, modified_text, "15% replacement with mixed script text")

# Example 6: Longer text with more context
text = """The targeted homoglyph attack is designed to strategically replace characters 
in text while considering the surrounding context. This makes the replacements more natural 
and less detectable than random replacements, while still maintaining the visual similarity 
that is the hallmark of homoglyph attacks."""

print("\nExample 6: Longer text with more context")
modified_text = targeted_attack(
    text=text,
    percentage=0.1,
    context_window_size=15  # Larger context window
)
print_comparison(text, modified_text, "10% replacement with larger context window")
