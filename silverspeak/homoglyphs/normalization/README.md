# SilverSpeak Homoglyph Normalization

This package provides various strategies for normalizing text containing homoglyphs (characters that look visually similar but have different Unicode code points).

## Available Strategies

- **Local Context Strategy**: Normalizes text by analyzing the surrounding characters to choose optimal replacements
- **Tokenizer Strategy**: Uses tokenizer-based heuristics to select homoglyphs that result in more natural tokenization
- **Language Model Strategy**: Uses a masked language model to predict the most likely character at each position
- **Dominant Script Strategy**: Normalizes based on the dominant Unicode script detected in the text
- **Dominant Script and Block Strategy**: Uses both script and Unicode block information for more precise normalization

## Usage

```python
from silverspeak.homoglyphs.normalize import normalize_text
from silverspeak.homoglyphs.utils import NormalizationStrategies

# Normalize using local context (default strategy)
normalized_text = normalize_text("Hеllo wоrld")  # Contains Cyrillic 'е' and 'о'

# Normalize using a specific strategy
normalized_text = normalize_text(
    "Hеllo wоrld", 
    strategy=NormalizationStrategies.DOMINANT_SCRIPT
)

# Use language model strategy with custom model name
normalized_text = normalize_text(
    "Hеllo wоrld",
    strategy=NormalizationStrategies.LANGUAGE_MODEL,
    model_name="bert-base-uncased"
)
```
