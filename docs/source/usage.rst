Usage
=====

SilverSpeak is a Python library designed to manipulate text using homoglyphs for security and adversarial purposes. Below are examples of its usage:

1. **Performing a Greedy Attack**:

   .. code-block:: python

      from silverspeak.homoglyphs import greedy_attack

      text = "example text"
      attacked_text = greedy_attack.perform_attack(text)
      print(attacked_text)

2. **Normalizing Text**:

   .. code-block:: python

      from silverspeak.homoglyphs import normalize_text

      normalized_text = normalize_text("exаmple")  # Note: homoglyph 'а' (Cyrillic)
      print(normalized_text)

3. **Replacing Homoglyphs**:

   .. code-block:: python

      from silverspeak.homoglyphs import HomoglyphReplacer

      replacer = HomoglyphReplacer()
      replaced_text = replacer.replace("example")
      print(replaced_text)

Refer to the API reference for detailed documentation of each module.