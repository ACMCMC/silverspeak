#!/usr/bin/env python3
"""
Simple script to test the spell checking-based normalization strategy.

To use this script, first install the spell checking dependencies:
    poetry install --with spell-check

For contextual spell checking:
    poetry install --with contextual-spell-check
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

# Test the spell check strategy with English
print("Spell check normalization (English):")
normalized_text = normalize_text(
    sample_text,
    strategy=NormalizationStrategies.SPELL_CHECK,
    language="en",
    distance_threshold=0.7  # Be a bit more lenient for demonstration
)
print(normalized_text)
print()

# Spanish example
spanish_text = "Вuеnоs díаs аmіgо"  # "Buenos días amigo" with some homoglyphs

print("Spanish example with homoglyphs:")
print(spanish_text)
print()

print("Spell check normalization (Spanish):")
normalized_spanish = normalize_text(
    spanish_text,
    strategy=NormalizationStrategies.SPELL_CHECK,
    language="es",
    distance_threshold=0.7
)
print(normalized_spanish)
print()

# Test with custom dictionary for specialized terms
specialized_text = "Маchinе Lеаrning and Аrtificiаl Intеlligеnce"  # With homoglyphs

print("Specialized text with homoglyphs:")
print(specialized_text)
print()

print("Spell check with custom dictionary:")
custom_dict = ["Machine", "Learning", "Artificial", "Intelligence"]
normalized_specialized = normalize_text(
    specialized_text,
    strategy=NormalizationStrategies.SPELL_CHECK,
    custom_dictionary=custom_dict
)
print(normalized_specialized)
