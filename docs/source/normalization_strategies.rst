Normalization Strategies
========================

The `normalization` package provides a suite of methods to normalize text by leveraging various linguistic and contextual properties. These strategies are designed to handle homoglyphs and other text normalization challenges effectively. Below is a detailed explanation of each strategy:

1. **Dominant Script Strategy**:

   This strategy identifies the dominant script in the input text and applies normalization based on it. The dominant script is determined by analyzing the frequency of scripts in the text. If the most frequent script constitutes less than 75% of the total characters, a warning is logged, as this may indicate mixed-script text. The normalization map is then applied to replace characters based on the dominant script.

   .. code-block:: python

      from silverspeak.homoglyphs.normalization import apply_dominant_script_strategy

      normalized_text = apply_dominant_script_strategy(replacer, text="example text")
      print(normalized_text)

2. **Dominant Script and Block Strategy**:

   This strategy extends the dominant script strategy by also considering the dominant Unicode block. The dominant block is identified similarly to the script, and a combined normalization map is applied. This approach provides finer granularity in normalization by accounting for both script and block-level properties.

   .. code-block:: python

      from silverspeak.homoglyphs.normalization import apply_dominant_script_and_block_strategy

      normalized_text = apply_dominant_script_and_block_strategy(replacer, text="example text")
      print(normalized_text)

3. **Local Context Strategy**:

   This strategy uses an advanced sliding window approach to normalize text based on sophisticated local context analysis. For each character, it evaluates possible replacements by analyzing their Unicode properties (script, block, plane, category, bidirectional, east_asian_width) and comparing them with the properties of surrounding characters in the context window. The strategy uses a weighted scoring system where Unicode block has the highest importance, followed by plane and script. This ensures that replacement characters not only look similar but also have compatible Unicode properties that maintain visual and semantic coherence with their context.

   .. code-block:: python

      from silverspeak.homoglyphs.normalization import apply_local_context_strategy

      normalized_text = apply_local_context_strategy(
          text="example text", 
          normalization_map={},
          N=10  # Context window size
      )
      print(normalized_text)

4. **Tokenizer Strategy**:

   This strategy leverages a pre-trained tokenizer to normalize text. It evaluates possible replacements for each character by analyzing their compatibility with tokens in the tokenizer's vocabulary. Scores are computed based on criteria such as token length, number of possible tokens, and token frequency. The replacement with the highest aggregated score is selected.

   .. code-block:: python

      from silverspeak.homoglyphs.normalization import apply_tokenizer_strategy

      normalized_text = apply_tokenizer_strategy(text="example text", mapping={})
      print(normalized_text)

5. **Language Model Strategy**:

   This strategy utilizes a pre-trained language model to normalize text. The model predicts the most likely replacement for each character based on its context in the text. This approach is particularly powerful for handling complex normalization scenarios where linguistic context plays a crucial role.

   .. code-block:: python

      from silverspeak.homoglyphs.normalization import apply_language_model_strategy
      from transformers import AutoTokenizer, AutoModelForMaskedLM

      tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")
      model = AutoModelForMaskedLM.from_pretrained("bert-base-multilingual-cased")

      normalized_text = apply_language_model_strategy(text="example text", mapping={}, language_model=model, tokenizer=tokenizer)
      print(normalized_text)

6. **LLM Prompt Strategy**:

   This strategy leverages large language models (LLMs) with a prompt-based approach to fix homoglyphs in text. The model is prompted to identify and correct homoglyphs, providing a flexible and powerful approach that can handle complex linguistic patterns without requiring explicit character-by-character mapping.

   .. code-block:: python

      from silverspeak.homoglyphs.normalize import normalize_text
      from silverspeak.homoglyphs.utils import NormalizationStrategies

      # Make sure you have the required dependencies
      # poetry install --with transformers

      normalized_text = normalize_text(
          "Tһis іs а tеst with ѕome һomoglурhs.",
          strategy=NormalizationStrategies.LLM_PROMPT,
          model_name="google/gemma-2-1b-it"  # Optional, specify a different model
      )
      print(normalized_text)

7. **Spell Check Strategy**:

   This strategy uses spell checking algorithms to identify and correct words containing homoglyphs. It supports multiple languages and can be customized with language-specific dictionaries or user-provided word lists. This approach is particularly effective for text where homoglyphs cause words to be misspelled.

   .. code-block:: python

      from silverspeak.homoglyphs.normalize import normalize_text
      from silverspeak.homoglyphs.utils import NormalizationStrategies

      # Make sure you have the required dependencies
      # poetry install --with spell-check

      # Basic usage with English (default language)
      normalized_text = normalize_text(
          "Tһis іs а tеst with ѕome һomoglурhs.",
          strategy=NormalizationStrategies.SPELL_CHECK
      )
      print(normalized_text)

8. **N-gram Strategy**:

   This strategy uses character n-gram frequency analysis to identify and correct homoglyphs. It builds n-gram models of different sizes (bigrams, trigrams, etc.) and identifies unlikely character sequences that might indicate homoglyphs. This approach is particularly effective for detecting contextually unusual characters.

   .. code-block:: python

      from silverspeak.homoglyphs.normalize import normalize_text
      from silverspeak.homoglyphs.utils import NormalizationStrategies

      # Make sure you have the required dependencies
      # poetry install --with ngram-analysis

      normalized_text = normalize_text(
          "Tһis іs а tеst with ѕome һomoglурhs.",
          strategy=NormalizationStrategies.NGRAM,
          language="english",  # Optional language parameter
          threshold=0.01       # Optional probability threshold
      )
      print(normalized_text)

9. **OCR Confidence Strategy**:

   This strategy uses OCR confidence scores to detect and correct homoglyphs. It simulates OCR processing on characters and uses confidence scores to identify potential homoglyphs. Characters with lower OCR confidence are candidates for normalization. The strategy can use either Tesseract OCR or pre-computed confusion matrices.

   .. code-block:: python

      from silverspeak.homoglyphs.normalize import normalize_text
      from silverspeak.homoglyphs.utils import NormalizationStrategies

      # Make sure you have the required dependencies for full functionality
      # pip install pytesseract pillow

      normalized_text = normalize_text(
          "Tһis іs а tеst with ѕome һomoglурhs.",
          strategy=NormalizationStrategies.OCR_CONFIDENCE,
          confidence_threshold=0.7,  # Optional confidence threshold
          use_tesseract=True         # Optional flag to use Tesseract OCR
      )
      print(normalized_text)

10. **Graph-based Network Strategy**:

    This strategy uses a graph representation of character relationships to normalize text. Characters are represented as nodes in a graph with edges representing visual or contextual similarity. The strategy applies graph algorithms to find the optimal path for normalizing homoglyphs to their standard forms.

    .. code-block:: python

       from silverspeak.homoglyphs.normalize import normalize_text
       from silverspeak.homoglyphs.utils import NormalizationStrategies

       # Make sure you have the required dependencies
       # poetry install --with graph-analysis

       normalized_text = normalize_text(
           "Tһis іs а tеst with ѕome һomoglурhs.",
           strategy=NormalizationStrategies.GRAPH_BASED,
           similarity_threshold=0.8  # Optional similarity threshold
       )
       print(normalized_text)

      # Using with a different language
      spanish_text = normalize_text(
          "Вuеnоs díаs аmіgо",  # "Buenos días amigo" with homoglyphs
          strategy=NormalizationStrategies.SPELL_CHECK,
          language="es"  # Spanish
      )
      print(spanish_text)
      
      # Advanced usage with custom dictionary words
      custom_text = normalize_text(
          "SіlvеrSреаk is а lіbrаrу for homoglурh dеtеctіon",
          strategy=NormalizationStrategies.SPELL_CHECK,
          custom_words=["SilverSpeak", "homoglyph"],  # Adds these words to the dictionary
          distance=2  # Maximum edit distance for corrections (default is 2)
      )
      print(custom_text)
      
      # Using contextual spell checking (requires additional dependencies)
      # poetry install --with contextual-spell-check
      contextual_text = normalize_text(
          "Tһe cat is јumріng оn tһe fеnсe",
          strategy=NormalizationStrategies.SPELL_CHECK,
          use_contextual=True  # Uses neuspell for contextual corrections
      )
      print(contextual_text)

Each of these strategies is designed to address specific text normalization challenges, providing flexibility and precision in handling diverse linguistic scenarios.