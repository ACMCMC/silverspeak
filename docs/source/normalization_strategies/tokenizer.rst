Tokenizer Strategy
==================

The Tokenizer Strategy is a normalization approach that leverages a pre-trained tokenizer to ensure compatibility with downstream tokenized processing. This strategy evaluates possible replacements for each character based on their compatibility with tokens in the tokenizer's vocabulary.

Overview
--------

This strategy is particularly useful for applications where text is processed in tokenized form, such as machine translation, language modeling, or text generation. By aligning the normalization process with the tokenizer's vocabulary, this strategy ensures that the output text is both syntactically valid and semantically meaningful.

Implementation Details
-----------------------

1. **Tokenizer Initialization**:
   The following code snippet demonstrates how to initialize a pre-trained tokenizer. The tokenizer provides a comprehensive vocabulary for evaluating possible replacements.

   .. code-block:: python

      from transformers import AutoTokenizer
      tokenizer = AutoTokenizer.from_pretrained("bigscience/bloom")
      vocab = tokenizer.get_vocab().keys()
      vocab = sorted(vocab, key=len, reverse=True)

   In this step, the `transformers` library is used to load a tokenizer from the `bigscience/bloom` model. The tokenizer's vocabulary is sorted by token length to facilitate efficient processing during the replacement step.

2. **Token Filtering**:
   This snippet filters the vocabulary to include only tokens that contain the character or its possible replacements. This ensures that the replacements are compatible with the tokenizer's vocabulary.

   .. code-block:: python

      possible_token_starts = {
          char: [
              (token[: token.rindex(char)], len(token), token)
              for token in vocab if char in token
          ]
          for char in possible_chars
      }

   Here, the `possible_token_starts` dictionary maps each character to a list of tokens that contain it. For each token, the substring up to the character's last occurrence, the token's length, and the full token are stored. This information is used to evaluate the compatibility of each replacement.

3. **Scoring and Aggregation**:
   The following code snippet demonstrates how to compute scores for each possible replacement based on criteria such as token length and frequency. These scores are aggregated to determine the best replacement.

   .. code-block:: python

      scores[char] = (
          LONGEST_START_WEIGHT * longest_start_score
          + LONGEST_TOKEN_WEIGHT * longest_token_score
          + NUM_POSSIBLE_TOKENS_WEIGHT * num_possible_tokens_score
          + NUM_TOKENS_CONTAINING_CHAR_WEIGHT * num_tokens_containing_char
      )

   The scoring function assigns weights to different criteria, such as the length of the token's prefix (`longest_start_score`), the token's overall length (`longest_token_score`), the number of possible tokens (`num_possible_tokens_score`), and the number of tokens containing the character (`num_tokens_containing_char`). These weights can be adjusted to prioritize specific aspects of the replacement process.

4. **Selection**:
   This snippet selects the replacement with the highest aggregated score and applies it to the text.

   .. code-block:: python

      best_char = max(scores, key=scores.get)
      normalized_text += best_char

   The `max` function is used to identify the character with the highest score. This character is then appended to the `normalized_text` string, which stores the final output.

Example Usage
-------------

The following example demonstrates how to normalize a text using the Tokenizer Strategy. It applies the strategy to ensure compatibility with downstream tokenized processing.

.. code-block:: python

   text = "Example text with homoglyphs."
   normalization_map = {"a": ["α", "а"], "e": ["е", "ε"]}
   normalized_text = apply_tokenizer_strategy(text, normalization_map)
   print(normalized_text)

   In this example, the `apply_tokenizer_strategy` function is used to normalize the input text. The function leverages the tokenizer's vocabulary to evaluate possible replacements and select the most compatible option for each character.

Key Considerations
-------------------
- The choice of tokenizer significantly impacts the effectiveness of this strategy. A tokenizer with a diverse vocabulary is recommended.
- This strategy is ideal for texts intended for tokenized processing, such as machine translation or language modeling.
- The scoring function can be customized to prioritize specific criteria, such as token length or frequency, based on the application's requirements.