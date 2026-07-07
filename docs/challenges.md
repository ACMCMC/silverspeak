# Challenges in Normalizing Texts with Homoglyphs

Homoglyphs are visually similar characters from different scripts or encodings. Normalizing them is inherently ambiguous: without context, you cannot always know which character the author intended.

SilverSpeak v3 uses the HKB fast pipeline to pick canonical replacements from a ranked graph. It returns ambiguity metadata instead of guessing when candidates tie. Choose `min_score` and `score_margin` for your use case.
