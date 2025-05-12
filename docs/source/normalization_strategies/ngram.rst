N-gram Normalization Strategy
==========================

The N-gram normalization strategy leverages character-level n-gram models to identify and correct homoglyphs in text. This approach analyzes the frequency and probability of character sequences to detect anomalies that may indicate the presence of homoglyphs.

Core Concepts
------------

1. **Character N-grams**: The strategy builds and analyzes n-grams (sequences of n characters) from the input text. Typically, it uses a combination of bigrams (n=2), trigrams (n=3), and larger n-grams to capture different levels of character context.

2. **Frequency Analysis**: By comparing the frequency of n-grams in the input text with expected frequencies in a given language, the strategy can identify unusual character sequences that might contain homoglyphs.

3. **Probability Threshold**: Characters that participate in n-grams with probabilities below a specified threshold are candidates for normalization.

Implementation Details
--------------------

The implementation in SilverSpeak uses NLTK for building and analyzing n-gram models. However, a simplified approach is available when NLTK is not installed. The key components include:

1. **CharNgramAnalyzer**: A class that builds and manages character n-gram models for different languages.

2. **LanguageNgramModel**: Represents an n-gram model for a specific language, providing methods to compute n-gram probabilities.

3. **apply_ngram_strategy**: The main function that applies the strategy to normalize text.

Usage
-----

Basic usage with default settings:

.. code-block:: python

    from silverspeak.homoglyphs.normalize import normalize_text
    from silverspeak.homoglyphs.utils import NormalizationStrategies

    normalized_text = normalize_text(
        "Tһis іs а tеst with ѕome һomoglурhs.",
        strategy=NormalizationStrategies.NGRAM
    )
    print(normalized_text)

Customizing the strategy:

.. code-block:: python

    normalized_text = normalize_text(
        "Tһis іs а tеst with ѕome һomoglурhs.",
        strategy=NormalizationStrategies.NGRAM,
        language="english",       # Specify language
        n_values=[2, 3, 4, 5],    # Specify n-gram sizes
        threshold=0.001           # Probability threshold
    )
    print(normalized_text)

Dependencies
-----------

For optimal performance, the N-gram strategy requires NLTK:

.. code-block:: bash

    poetry install --with ngram-analysis

The strategy will still work without NLTK, but will use a simplified approach that may be less effective.
