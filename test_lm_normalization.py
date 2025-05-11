#!/usr/bin/env python3
"""
Simple script to test the language model normalization strategy.
"""

import logging
from silverspeak.homoglyphs.normalize import normalize_text
from silverspeak.homoglyphs.utils import NormalizationStrategies

# Configure logging
logging.basicConfig(level=logging.INFO)

# Sample text with homoglyphs
sample_text = "Tһis іs а tеst with ѕome һomoglурhs."  # Contains various homoglyphs

print("Original text:")
print(sample_text)
print()

# Character-level normalization
print("Character-level normalization:")
normalized_text = normalize_text(
    sample_text,
    strategy=NormalizationStrategies.LANGUAGE_MODEL,
    word_level=False,
    model_name="bert-base-uncased"
)
print(normalized_text)
print()

# Word-level normalization
print("Word-level normalization:")
normalized_text = normalize_text(
    sample_text,
    strategy=NormalizationStrategies.LANGUAGE_MODEL,
    word_level=True,
    model_name="bert-base-uncased"
)
print(normalized_text)
