#!/usr/bin/env python3
"""
Simple script to test the LLM prompt-based normalization strategy.
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

# Test the LLM Prompt strategy
print("LLM Prompt normalization (this may take a while to download/run the model):")
normalized_text = normalize_text(
    sample_text,
    strategy=NormalizationStrategies.LLM_PROMPT,
    model_name="google/gemma-2-1b-it",  # Specify a different model if needed
    temperature=0.0  # Set to 0 for deterministic output
)
print(normalized_text)
print()

# Add more text to show comparison
complex_sample = "Tһe quіck brоwn fоx jumрs оvеr tһе lаzу dоg."

print("Complex sample:")
print(complex_sample)
print()

print("LLM Prompt normalization for complex sample:")
normalized_complex = normalize_text(
    complex_sample,
    strategy=NormalizationStrategies.LLM_PROMPT,
    model_name="google/gemma-2-1b-it"
)
print(normalized_complex)
