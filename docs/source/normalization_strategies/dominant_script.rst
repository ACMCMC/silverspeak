Dominant Script Strategy
========================

The Dominant Script Strategy is a normalization approach that identifies the most frequently occurring script in a given text and applies normalization rules based on this dominant script. This strategy is particularly effective for texts predominantly written in a single script, such as Latin, Cyrillic, or Arabic.

Overview
--------

This strategy is designed to address the challenge of mixed-script text, where characters from different scripts may appear together. By identifying the dominant script, the strategy ensures that homoglyphs are replaced in a manner consistent with the primary script of the text. This is particularly useful for applications such as text normalization, data preprocessing, and security.

Implementation Details
-----------------------

1. **Script Detection**:
   The following code snippet demonstrates how to detect the dominant script in a text. It calculates the frequency of each script and identifies the most common one. If the dominant script constitutes less than 75% of the total characters, a warning is logged. This step ensures that the text is predominantly written in a single script, which is a prerequisite for effective normalization.

   .. code-block:: python

      def detect_dominant_script(text: str) -> str:
          script_counts = Counter(unicodedataplus.script(char) for char in text)
          total_count = sum(script_counts.values())
          dominant_script = max(script_counts, key=script_counts.get)
          if script_counts[dominant_script] / total_count < 0.75:
              logging.warning(
                  f"The dominant script '{dominant_script}' comprises less than 75% of the total character count."
              )
          return dominant_script

   This function uses the `unicodedataplus` library to determine the script of each character in the text. The `Counter` class from the `collections` module is used to count the occurrences of each script, and the most frequent script is identified as the dominant script.

2. **Normalization Map Retrieval**:
   Once the dominant script is identified, a normalization map tailored to that script is retrieved. This map defines the replacements for homoglyphs based on the script's characteristics.

   .. code-block:: python

      def apply_dominant_script_strategy(replacer, text: str, **kwargs):
          dominant_script = detect_dominant_script(text)
          normalization_map = replacer.get_normalization_map_for_script_block_and_category(
              script=dominant_script, **kwargs
          )
          return text.translate(str.maketrans(normalization_map))

   In this function, the `get_normalization_map_for_script_block_and_category` method of the `HomoglyphReplacer` class is used to retrieve the normalization map. The `str.translate` method is then employed to replace characters in the text based on the map.

Example Usage
-------------

The following example demonstrates how to normalize a text using the Dominant Script Strategy. It initializes a `HomoglyphReplacer` instance and applies the strategy to replace homoglyphs based on the dominant script.

.. code-block:: python

   from homoglyph_replacer import HomoglyphReplacer

   text = "Some example text with homoglyphs."
   replacer = HomoglyphReplacer()
   normalized_text = apply_dominant_script_strategy(replacer, text)
   print(normalized_text)

   In this example, the `HomoglyphReplacer` class is used to manage the normalization process. The `apply_dominant_script_strategy` function is called with the input text and the replacer instance, and the normalized text is printed.

Key Considerations
-------------------
- This strategy assumes that the text is predominantly written in a single script. If the dominant script constitutes less than 75% of the total characters, a warning is logged.
- The normalization map is tailored to the identified script, ensuring consistency in character representation.
- This approach is most effective for texts with a clear dominant script. For mixed-script texts, additional preprocessing may be required.