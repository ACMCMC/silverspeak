from silverspeak.homoglyphs.utils import HomoglyphReplacer


def normalize_text(text: str, unicode_categories_to_replace=["Ll", "Lm", "Lo", "Lt", "Lu"], types_of_homoglyphs_to_use=["identical", "confusables", "ocr"], replace_with_priority=False) -> str:
    replacer = HomoglyphReplacer(unicode_categories_to_replace=unicode_categories_to_replace, types_of_homoglyphs_to_use=types_of_homoglyphs_to_use, replace_with_priority=replace_with_priority)
    return replacer.normalize(text)
