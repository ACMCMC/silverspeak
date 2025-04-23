Normalization Strategies
========================

The `normalization_strategies.py` module provides a suite of methods to normalize text by leveraging various linguistic and contextual properties. These strategies are designed to handle homoglyphs and other text normalization challenges effectively. Below is a detailed explanation of each strategy:

1. **Dominant Script Strategy**:

   This strategy identifies the dominant script in the input text and applies normalization based on it. The dominant script is determined by analyzing the frequency of scripts in the text. If the most frequent script constitutes less than 75% of the total characters, a warning is logged, as this may indicate mixed-script text. The normalization map is then applied to replace characters based on the dominant script.

   .. code-block:: python

      from silverspeak.homoglyphs.normalization_strategies import apply_dominant_script_strategy

      normalized_text = apply_dominant_script_strategy(replacer, text="example text")
      print(normalized_text)

2. **Dominant Script and Block Strategy**:

   This strategy extends the dominant script strategy by also considering the dominant Unicode block. The dominant block is identified similarly to the script, and a combined normalization map is applied. This approach provides finer granularity in normalization by accounting for both script and block-level properties.

   .. code-block:: python

      from silverspeak.homoglyphs.normalization_strategies import apply_dominant_script_and_block_strategy

      normalized_text = apply_dominant_script_and_block_strategy(replacer, text="example text")
      print(normalized_text)

3. **Local Context Strategy**:

   This strategy uses a sliding window approach to normalize text based on local context. For each character, it evaluates possible replacements by comparing their properties (e.g., script, block, category) with those of surrounding characters in the window. The replacement that best matches the context is selected. If multiple replacements have the same score, a warning is logged, and the first one is chosen.

   .. code-block:: python

      from silverspeak.homoglyphs.normalization_strategies import apply_local_context_strategy

      normalized_text = apply_local_context_strategy(text="example text", normalization_map={})
      print(normalized_text)

4. **Tokenizer Strategy**:

   This strategy leverages a pre-trained tokenizer to normalize text. It evaluates possible replacements for each character by analyzing their compatibility with tokens in the tokenizer's vocabulary. Scores are computed based on criteria such as token length, number of possible tokens, and token frequency. The replacement with the highest aggregated score is selected.

   .. code-block:: python

      from silverspeak.homoglyphs.normalization_strategies import apply_tokenizer_strategy

      normalized_text = apply_tokenizer_strategy(text="example text", mapping={})
      print(normalized_text)

5. **Language Model Strategy**:

   This strategy utilizes a pre-trained language model to normalize text. The model predicts the most likely replacement for each character based on its context in the text. This approach is particularly powerful for handling complex normalization scenarios where linguistic context plays a crucial role.

   .. code-block:: python

      from silverspeak.homoglyphs.normalization_strategies import apply_language_model_strategy
      from transformers import AutoTokenizer, AutoModelForMaskedLM

      tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")
      model = AutoModelForMaskedLM.from_pretrained("bert-base-multilingual-cased")

      normalized_text = apply_language_model_strategy(text="example text", mapping={}, language_model=model, tokenizer=tokenizer)
      print(normalized_text)

Each of these strategies is designed to address specific text normalization challenges, providing flexibility and precision in handling diverse linguistic scenarios.