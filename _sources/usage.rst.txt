Usage
=====

SilverSpeak provides tools for homoglyph-based text manipulation and analysis. Below is an example of how to use the library:

.. code-block:: python

   from silverspeak.homoglyphs import greedy_attack

   text = "example text"
   attacked_text = greedy_attack.perform_attack(text)
   print(attacked_text)

Refer to the API reference for detailed usage of each module.