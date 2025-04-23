Dominant Script and Block Strategy
==================================

The Dominant Script and Block Strategy extends the Dominant Script Strategy by incorporating the dominant Unicode block into the normalization process. This approach provides finer granularity in character normalization by considering both script and block-level properties.

Overview
--------

This strategy is particularly useful for texts where both the script and the Unicode block play a significant role in character representation. By combining these two dimensions, the strategy ensures that homoglyphs are replaced in a manner that respects both the script and the block-level context of the text.

Implementation Details
-----------------------

1. **Script and Block Detection**:
   The following code snippet demonstrates how to detect the dominant Unicode block in a text. It calculates the frequency of each block and identifies the most common one. If the dominant block constitutes less than 75% of the total characters, a warning is logged. This step ensures that the text is predominantly written within a single block, which is essential for effective normalization.

   .. code-block:: python

      def detect_dominant_block(text: str) -> str:
          block_counts = Counter(unicodedataplus.block(char) for char in text)
          total_count = sum(block_counts.values())
          dominant_block = max(block_counts, key=block_counts.get)
          if block_counts[dominant_block] / total_count < 0.75:
              logging.warning(
                  f"The dominant Unicode block '{dominant_block}' comprises less than 75% of the total character count."
              )
          return dominant_block

   This function uses the `unicodedataplus` library to determine the Unicode block of each character in the text. The `Counter` class from the `collections` module is used to count the occurrences of each block, and the most frequent block is identified as the dominant block.

2. **Normalization Map Retrieval**:
   Once the dominant script and block are identified, a combined normalization map is retrieved. This map defines the replacements for homoglyphs based on both the script and block-level properties.

   .. code-block:: python

      def apply_dominant_script_and_block_strategy(replacer, text: str, **kwargs):
          dominant_script = detect_dominant_script(text)
          dominant_block = detect_dominant_block(text)
          normalization_map = replacer.get_normalization_map_for_script_block_and_category(
              script=dominant_script, block=dominant_block, **kwargs
          )
          return text.translate(str.maketrans(normalization_map))

   In this function, the `get_normalization_map_for_script_block_and_category` method of the `HomoglyphReplacer` class is used to retrieve the combined normalization map. The `str.translate` method is then employed to replace characters in the text based on the map.

Example Usage
-------------

The following example demonstrates how to normalize a text using the Dominant Script and Block Strategy. It initializes a `HomoglyphReplacer` instance and applies the strategy to replace homoglyphs based on both the dominant script and block.

.. code-block:: python

   from homoglyph_replacer import HomoglyphReplacer

   text = "Some example text with homoglyphs."
   replacer = HomoglyphReplacer()
   normalized_text = apply_dominant_script_and_block_strategy(replacer, text)
   print(normalized_text)

   In this example, the `HomoglyphReplacer` class is used to manage the normalization process. The `apply_dominant_script_and_block_strategy` function is called with the input text and the replacer instance, and the normalized text is printed.

Key Considerations
-------------------
- This strategy is suitable for texts where both script and Unicode block play a significant role in character representation.
- If the dominant script or block constitutes less than 75% of the total characters, a warning is logged.
- This approach provides finer granularity in normalization by considering both script and block-level properties.