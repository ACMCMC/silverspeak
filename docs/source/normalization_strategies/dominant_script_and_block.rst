Dominant Script and Block Strategy
==================================

The Dominant Script and Block Strategy is an advanced normalization approach that extends the Dominant Script Strategy by incorporating both the dominant Unicode script and the dominant Unicode block into the normalization process. This dual-criteria approach provides significantly finer granularity and higher precision in character normalization, making it particularly effective for complex homoglyph scenarios.

Overview
--------

This strategy recognizes that homoglyphs often occur not just across different scripts, but also across different Unicode blocks within the same script or between closely related scripts. By analyzing both the script distribution and the Unicode block distribution, the strategy can make more informed decisions about appropriate character replacements.

The approach is especially valuable for:
- Texts containing characters from closely related scripts (e.g., Latin and Cyrillic)
- Documents with technical content that may include symbols from various Unicode blocks
- Security applications where precision in homoglyph detection is critical
- Mixed-language documents where script alone might be insufficient for proper normalization

Implementation Details
-----------------------

1. **Dual-Criteria Detection System**:
   The strategy performs parallel analysis of both Unicode scripts and Unicode blocks across all characters in the input text:

   .. code-block:: python

      def detect_dominant_script(text: str) -> str:
          script_counts = Counter(unicodedataplus.script(char) for char in text)
          total_count = sum(script_counts.values())
          dominant_script = max(script_counts.keys(), key=lambda k: script_counts[k])
          
          if script_counts[dominant_script] / total_count < 0.75:
              logging.warning(
                  f"The dominant script '{dominant_script}' comprises less than 75% of the total character count."
              )
          return dominant_script

      def detect_dominant_block(text: str) -> str:
          block_counts = Counter(unicodedataplus.block(char) for char in text)
          total_count = sum(block_counts.values())
          dominant_block = max(block_counts.keys(), key=lambda k: block_counts[k])
          
          if block_counts[dominant_block] / total_count < 0.75:
              logging.warning(
                  f"The dominant Unicode block '{dominant_block}' comprises less than 75% of the total character count."
              )
          return dominant_block

   Both detection functions use the same statistical approach as the single-criterion strategy, but operate on different Unicode properties. The 75% threshold is applied independently to both script and block analysis.

2. **Enhanced Normalization Map Generation**:
   The strategy generates a more precise normalization map by combining both script and block information:

   .. code-block:: python

      def apply_dominant_script_and_block_strategy(replacer, text: str, **kwargs) -> str:
          if not text:
              logging.warning("Empty text provided for normalization")
              return ""
          
          if not replacer:
              raise ValueError("No replacer provided for normalization")
          
          dominant_script = detect_dominant_script(text)
          dominant_block = detect_dominant_block(text)
          
          if dominant_script == "Unknown" or dominant_block == "Unknown":
              logging.warning("Unable to determine dominant script/block, normalization may be suboptimal")
          
          normalization_map = replacer.get_normalization_map_for_script_block_and_category(
              script=dominant_script, 
              block=dominant_block, 
              **kwargs
          )
          
          if not normalization_map:
              logging.warning(f"No normalization map available for script '{dominant_script}' and block '{dominant_block}'")
              return text
          
          return text.translate(str.maketrans(normalization_map))

   The combined approach allows the `HomoglyphReplacer` to select from a more targeted set of character mappings, resulting in higher precision and reduced false positives.

3. **Error Handling and Fallback Mechanisms**:
   The strategy includes sophisticated error handling to manage edge cases:

   .. code-block:: python

      # Detection of ambiguous cases
      if dominant_script == "Unknown" or dominant_block == "Unknown":
          logging.warning("Unable to determine dominant script/block, normalization may be suboptimal")
      
      # Handling missing normalization maps
      if not normalization_map:
          logging.warning(f"No normalization map available for script '{dominant_script}' and block '{dominant_block}'")
          return text

   These mechanisms ensure robust operation even when dealing with unusual or edge-case input texts.

Example Usage
-------------

The following examples demonstrate how to normalize text using the Dominant Script and Block Strategy, showcasing its enhanced precision compared to single-criterion approaches:

.. code-block:: python

   from silverspeak.homoglyphs.normalization import apply_dominant_script_and_block_strategy
   from silverspeak.homoglyphs import HomoglyphReplacer

   # Text with mixed Cyrillic and Latin characters in different blocks
   text = "Examрle tеxt with symbоls like ∑ and α."  # Contains Cyrillic + Greek symbols
   
   # Initialize the replacer
   replacer = HomoglyphReplacer()
   
   # Apply dual-criteria normalization
   normalized_text = apply_dominant_script_and_block_strategy(replacer, text)
   print(normalized_text)  # More precise normalization than script-only approach

This example shows how the strategy handles complex cases where characters from multiple Unicode blocks appear together.

**Alternative Usage via normalize_text**:

.. code-block:: python

   from silverspeak.homoglyphs import normalize_text
   from silverspeak.homoglyphs.utils import NormalizationStrategies

   text = "Examрle tеxt with symbоls like ∑ and α."
   normalized_text = normalize_text(
       text, 
       strategy=NormalizationStrategies.DOMINANT_SCRIPT_AND_BLOCK
   )
   print(normalized_text)

**Comparison with Single-Criterion Strategy**:

.. code-block:: python

   # Same text normalized with different strategies
   text = "Mathеmatical ехprеssion: ∫f(x)dx = F(b) - F(a)"  # Mixed scripts and symbols
   
   # Script-only normalization
   script_result = normalize_text(text, strategy=NormalizationStrategies.DOMINANT_SCRIPT)
   
   # Script + Block normalization  
   script_block_result = normalize_text(text, strategy=NormalizationStrategies.DOMINANT_SCRIPT_AND_BLOCK)
   
   print(f"Script only: {script_result}")
   print(f"Script + Block: {script_block_result}")
   # The script+block approach provides more nuanced handling of mathematical symbols

**Advanced Usage with Parameters**:

.. code-block:: python

   # Preserve case and filter by category
   normalized_text = apply_dominant_script_and_block_strategy(
       replacer, 
       text, 
       preserve_case=True,
       category="L"  # Letters only
   )

Key Considerations
-------------------

**Enhanced Precision and Accuracy:**

- **Dual-Criteria Analysis**: By analyzing both script and Unicode block, this strategy achieves significantly higher precision than single-criterion approaches.
- **Reduced False Positives**: The combination of criteria helps distinguish between legitimate mixed-content and potential homoglyph attacks.
- **Context Sensitivity**: Better handling of technical documents, mathematical expressions, and multilingual content.

**Performance Characteristics:**

- **Computational Overhead**: Slightly higher computational cost than single-criterion strategies due to dual analysis, but still O(n) complexity.
- **Memory Usage**: Minimal additional memory overhead for storing both script and block counts.
- **Precision vs. Speed Trade-off**: The slight performance cost is generally justified by significantly improved accuracy.

**Use Case Optimization:**

- **Technical Documents**: Excellent for scientific papers, mathematical texts, and technical documentation with symbols.
- **Multilingual Content**: Better handling of documents that legitimately contain characters from multiple scripts/blocks.
- **Security Applications**: Higher precision reduces false positives in security-critical applications.
- **Mixed-Script Languages**: Effective for languages that legitimately use characters from multiple Unicode blocks.

**Limitations and Considerations:**

- **Complexity Threshold**: May be overkill for simple, single-script texts where the basic dominant script strategy would suffice.
- **Block Granularity**: Some Unicode blocks are quite large and may not provide the expected precision in all cases.
- **Edge Cases**: Texts with highly diverse Unicode block usage may not benefit from this approach.

**Best Practices:**

- **Strategy Selection**: Use this strategy when dealing with texts that may contain technical symbols, mathematical notation, or mixed-script content.
- **Threshold Monitoring**: Monitor both script and block thresholds; consistent warnings may indicate the need for a different strategy.
- **Validation**: Test with representative samples of your target content to ensure the strategy provides the expected precision.
- **Fallback Planning**: Have fallback strategies ready for edge cases where dual-criteria analysis doesn't provide clear results.

**Security and Detection Benefits:**

- **Attack Sophistication**: This strategy can detect more sophisticated homoglyph attacks that attempt to exploit Unicode block boundaries.
- **Precision Filtering**: Higher precision reduces the risk of blocking legitimate multilingual or technical content.
- **Audit Trail**: Detailed logging of both script and block analysis provides better forensic capabilities for security investigations.