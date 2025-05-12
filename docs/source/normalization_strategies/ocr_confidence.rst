OCR Confidence Normalization Strategy
=================================

The OCR Confidence normalization strategy simulates Optical Character Recognition (OCR) processes to identify and correct homoglyphs in text. It relies on the principle that homoglyphs often cause OCR systems to have lower confidence in their recognition, which can be leveraged to detect potential substitution candidates.

Core Concepts
------------

1. **OCR Confidence Scores**: The strategy uses OCR confidence scores to evaluate each character. Characters with low confidence scores are more likely to be homoglyphs.

2. **Confusion Matrices**: Historical OCR confusion data is used to identify frequently confused character pairs, which often correspond to homoglyph relationships.

3. **Character Rendering**: When Tesseract OCR is available, the strategy can render characters to images and process them with OCR to obtain real-time confidence scores.

Implementation Details
--------------------

The implementation in SilverSpeak offers two approaches:

1. **Tesseract OCR-based**: Uses Pytesseract and Pillow to render and analyze characters in real-time.

2. **Confusion Matrix-based**: Uses pre-computed OCR confusion data from the SilverSpeak resources, which works even without OCR dependencies.

The main components include:

1. **OCRConfidenceAnalyzer**: A class that manages the OCR processing and confidence analysis.

2. **apply_ocr_confidence_strategy**: The main function that applies the strategy to normalize text.

Usage
-----

Basic usage with default settings:

.. code-block:: python

    from silverspeak.homoglyphs.normalize import normalize_text
    from silverspeak.homoglyphs.utils import NormalizationStrategies

    normalized_text = normalize_text(
        "Tһis іs а tеst with ѕome һomoglурhs.",
        strategy=NormalizationStrategies.OCR_CONFIDENCE
    )
    print(normalized_text)

Customizing the strategy:

.. code-block:: python

    normalized_text = normalize_text(
        "Tһis іs а tеst with ѕome һomoglурhs.",
        strategy=NormalizationStrategies.OCR_CONFIDENCE,
        confidence_threshold=0.8,    # Higher confidence threshold
        use_tesseract=False,         # Use only confusion matrix approach
        font_name="Arial",           # Optional font for rendering
        font_size=24                 # Optional font size
    )
    print(normalized_text)

Dependencies
-----------

For full functionality including real-time OCR analysis, the OCR Confidence strategy requires:

.. code-block:: bash

    pip install pytesseract pillow

Additionally, Tesseract OCR must be installed on the system. The strategy will still work without these dependencies, falling back to the confusion matrix-based approach.
