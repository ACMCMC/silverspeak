import unicodedata

_ZERO_WIDTH = {"\u200b", "\u200c", "\u200d", "\ufeff"}
_BIDI_OVERRIDE = {chr(c) for c in range(0x202A, 0x202F)}
_BIDI_ISOLATE = {chr(c) for c in range(0x2066, 0x206A)}
_STRIP_CHARS = _ZERO_WIDTH | _BIDI_OVERRIDE | _BIDI_ISOLATE


def _is_tags_block(char: str) -> bool:
    code = ord(char)
    return 0xE0000 <= code <= 0xE007F


def strip_format_chars(text: str) -> str:
    out = []
    for char in text:
        if char in _STRIP_CHARS:
            continue
        if _is_tags_block(char=char):
            continue
        if unicodedata.category(char) == "Cf" and char in _ZERO_WIDTH:
            continue
        out.append(char)
    return "".join(out)
