Usage
=====

SilverSpeak is a Python library designed to manipulate text using homoglyphs for security and adversarial purposes. Below are examples of its usage, with detailed explanations to help users understand the purpose and functionality of each operation:

1. **Performing a Greedy Attack**:

   A greedy attack replaces characters in the input text with visually similar homoglyphs to create an adversarial version of the text. This can be used to test the robustness of text-based systems, such as spam filters or OCR systems, against homoglyph-based attacks.

   .. code-block:: python

      from silverspeak.homoglyphs import greedy_attack

      # Define the input text to be attacked
      text = "example text"

      # Perform a greedy attack by replacing characters with homoglyphs
      attacked_text = greedy_attack.perform_attack(text)

      # Print the adversarially modified text
      print(attacked_text)

   In this example, the `perform_attack` function systematically replaces characters in the input text with homoglyphs from a predefined mapping. The resulting text may look visually similar to the original but is designed to confuse text-processing systems.

2. **Normalizing Text**:

   Text normalization replaces homoglyphs in the input text with their canonical equivalents. This is useful for ensuring consistency in text representation, especially when processing mixed-script text that may contain homoglyphs.

   .. code-block:: python

      from silverspeak.homoglyphs import normalize_text

      # Define the input text containing homoglyphs
      text = "exаmple"  # Note: homoglyph 'а' (Cyrillic) is used instead of 'a' (Latin)

      # Normalize the text by replacing homoglyphs with their canonical equivalents
      normalized_text = normalize_text(text)

      # Print the normalized text
      print(normalized_text)

   In this example, the `normalize_text` function identifies homoglyphs in the input text and replaces them with their standard Unicode equivalents. This ensures that the text is represented in a consistent and predictable manner.

3. **Replacing Homoglyphs**:

   Homoglyph replacement involves substituting characters in the input text with homoglyphs from a predefined mapping. This can be used to create visually obfuscated text for security or artistic purposes.

   .. code-block:: python

      from silverspeak.homoglyphs import HomoglyphReplacer

      # Initialize a HomoglyphReplacer instance
      replacer = HomoglyphReplacer()

      # Define the input text to be modified
      text = "example"

      # Replace characters in the text with homoglyphs
      replaced_text = replacer.replace(text)

      # Print the modified text
      print(replaced_text)

   In this example, the `HomoglyphReplacer` class provides a flexible interface for replacing characters in the input text with homoglyphs. The specific homoglyphs used for replacement are determined by the replacer's configuration.

Refer to the API reference for detailed documentation of each module, including additional parameters and advanced usage scenarios.