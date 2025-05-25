Local Context Strategy
======================

The Local Context Strategy is a sophisticated normalization approach that leverages a sliding window mechanism to evaluate the local context of each character in the text. This strategy ensures that the selected replacement for a character aligns with the properties of its surrounding characters, thereby maintaining contextual consistency.

Overview
--------

This strategy is designed to address the challenge of mixed-script text, where characters from different scripts may appear together. By analyzing the local context of each character, the strategy ensures that replacements are contextually appropriate and consistent with the surrounding text. This approach is particularly useful for applications such as text normalization, data preprocessing, and security.

Implementation Details
-----------------------

1. **Sliding Window Context**:
   The strategy uses a sliding window to capture the local context around each character. The window dynamically adjusts at the boundaries of the text to ensure sufficient context.

   .. code-block:: python

      start = max(0, i - N // 2)
      end = min(len(text), i + N // 2 + 1)
      context_window = text[start:end]

      # Ensure we have a sufficiently sized window
      if len(context_window) < min(N, len(text)):
          if start == 0:
              context_window = text[: min(N, len(text))]
          elif end == len(text):
              context_window = text[-min(N, len(text)) :]

   The `start` and `end` indices define the boundaries of the sliding window. The window size is determined by the parameter `N` (default 10), which specifies the number of characters to include in the context. The strategy ensures that the context window is always optimally sized, even at the edges of the text.

2. **Advanced Unicode Property Scoring**:
   The strategy uses the `score_homoglyphs_for_context_window` function from the unicode_scoring module to evaluate each possible replacement. This function analyzes multiple Unicode properties with different weights:

   .. code-block:: python

      PROPERTIES = {
          "script": {"fn": unicodedataplus.script, "weight": 2},
          "block": {"fn": unicodedataplus.block, "weight": 5},
          "plane": {"fn": lambda c: ord(c) >> 16, "weight": 3},
          "category": {"fn": unicodedata.category, "weight": 2},
          "bidirectional": {"fn": unicodedata.bidirectional, "weight": 2},
          "east_asian_width": {"fn": unicodedata.east_asian_width, "weight": 1},
      }

   Each property is assigned a weight based on its importance for visual similarity and contextual consistency. The Unicode block has the highest weight (5) as it's most indicative of visual similarity, while east_asian_width has the lowest weight (1).

3. **Context-Aware Scoring**:
   For each possible replacement character, the strategy calculates how well its Unicode properties match those of the surrounding characters in the context window:

   .. code-block:: python

      score_dict = score_homoglyphs_for_context_window(
          homoglyph=possible_char,
          char=char,
          context=context_window,
          context_window_size=N,
          PROPERTIES=PROPERTIES
      )
      score = score_dict.get("total_score", 0.0)

   The scoring function returns a detailed breakdown of scores for each property as well as a total aggregated score. Characters that better match the Unicode properties of their surrounding context receive higher scores.

4. **Selection and Tie Handling**:
   The replacement with the highest total score is selected. The strategy includes sophisticated tie handling and detailed logging:

   .. code-block:: python

      best_char, best_score = max(scores, key=lambda x: x[1])
      
      # Log detailed scoring information for debugging
      logging.debug(
          f"Character '{char}' at index {i}: chosen '{best_char}' with total score {best_score}. "
          f"Detailed scores: {best_detailed_score}. Context: '{context_window}'"
      )

      # Handle ties between multiple characters
      ties = [s[0] for s in scores if s[1] == best_score]
      if len(ties) > 1 and len(set(ties)) > 1:
          logging.debug(f"Found a tie for the best character for '{char}' at index {i}. Options: {ties}")

   The strategy provides extensive logging for debugging purposes, including detailed score breakdowns and information about ties between multiple replacement options.

Example Usage
-------------

The following example demonstrates how to normalize text using the Local Context Strategy. It leverages the advanced scoring system to maintain contextual consistency:

.. code-block:: python

   from silverspeak.homoglyphs.normalization import apply_local_context_strategy

   text = "Exаmple tеxt with һomoglуphs."  # Contains Cyrillic homoglyphs
   normalization_map = {
       "а": ["a"],  # Cyrillic 'а' to Latin 'a'
       "е": ["e"],  # Cyrillic 'е' to Latin 'e'
       "һ": ["h"],  # Cyrillic 'һ' to Latin 'h'
       "у": ["y"],  # Cyrillic 'у' to Latin 'y'
   }
   
   normalized_text = apply_local_context_strategy(
       text, 
       normalization_map, 
       N=10  # Context window size
   )
   print(normalized_text)  # Output: "Example text with homoglyphs."

This example shows how the strategy analyzes the Unicode properties of surrounding characters to make contextually appropriate replacements. The function uses sophisticated scoring to ensure that replacements maintain visual and semantic coherence with the surrounding text.

**Alternative Usage via normalize_text**:

.. code-block:: python

   from silverspeak.homoglyphs import normalize_text
   from silverspeak.homoglyphs.utils import NormalizationStrategies

   text = "Exаmple tеxt with һomoglуphs."
   normalized_text = normalize_text(
       text, 
       strategy=NormalizationStrategies.LOCAL_CONTEXT,
       N=10  # Optional: specify context window size
   )
   print(normalized_text)

Key Considerations
-------------------

**Performance and Complexity:**

- The sliding window size `N` significantly impacts both performance and accuracy. A larger window provides more context but increases computational complexity.
- The sophisticated scoring system makes this strategy more computationally intensive than simple mapping strategies, but produces higher quality results.

**Effectiveness:**

- This strategy is particularly effective for mixed-script texts where maintaining visual consistency is crucial.
- The weighted Unicode property system ensures that visually similar characters (same block, script) are preferred over less similar ones.
- The context-aware approach helps maintain the overall "feel" and readability of the text.

**Customization:**

- The Unicode properties and their weights can be customized by modifying the `PROPERTIES` parameter in the scoring function.
- The context window size can be adjusted based on the specific requirements of your application.
- Debug logging can be enabled to understand the scoring decisions and identify potential issues.

**Debugging and Monitoring:**

- The strategy provides detailed logging at the DEBUG level, showing score breakdowns for each character replacement decision.
- Tie situations are automatically detected and logged, helping identify ambiguous cases.
- Error handling ensures that scoring failures don't break the normalization process.