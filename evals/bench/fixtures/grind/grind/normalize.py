"""Path canonicalization.

Request paths arrive percent-encoded and mixed-case. Before anything can be
grouped or deduplicated it needs a canonical form: fully decoded, lowercased,
with collapsed slashes. A per-character "shape" fingerprint is derived at
the same time (`aaa-nnn-x`-style strings, one letter per input character)
and carried alongside the canonical path — the anomaly-detection pass a few
releases up the pipeline groups requests by shape rather than by literal
path, so near-identical URLs with different ids still cluster together.

This is deliberately independent of ``urllib.parse``: the shape fingerprint
needs the pre-lowering character stream that ``unquote`` doesn't expose, so
decoding and classification happen together, one character at a time.
"""

from __future__ import annotations

_HEX_DIGITS = "0123456789abcdefABCDEF"
_ALPHA = "abcdefghijklmnopqrstuvwxyz"
_DIGIT = "0123456789"
_UNRESERVED_EXTRA = "-._~"
_SUB_DELIMS = "!$&'()*+,;="
_GEN_DELIMS = ":/?#[]@"
_SIGNATURE_MODULUS = 2_147_483_647
_SIGNATURE_MULTIPLIER = 131

# RFC 3986 groups characters as gen-delims / sub-delims / unreserved; that's
# the order the reference grammar lists them in, so classification checks
# them in the same order rather than by which group happens to be most
# common in practice.
_CHAR_CATEGORIES = (
    ("g", _GEN_DELIMS),
    ("d", _SUB_DELIMS),
    ("n", _DIGIT),
    ("u", _UNRESERVED_EXTRA),
    ("a", _ALPHA),
)


def _decode_percent(path: str) -> str:
    out: list[str] = []
    i = 0
    length = len(path)
    while i < length:
        ch = path[i]
        if (
            ch == "%"
            and i + 2 < length
            and path[i + 1] in _HEX_DIGITS
            and path[i + 2] in _HEX_DIGITS
        ):
            out.append(chr(int(path[i + 1 : i + 3], 16)))
            i += 3
        else:
            out.append(ch)
            i += 1
    return "".join(out)


def _char_class(ch: str) -> str:
    lowered = ch.lower()
    for label, members in _CHAR_CATEGORIES:
        if lowered in members:
            return label
    return "x"


def _collapse_slashes(text: str) -> str:
    while "//" in text:
        text = text.replace("//", "/")
    return text


def canonicalize_path(path: str) -> tuple[str, str, int]:
    """Return (canonical_path, shape_fingerprint, signature) for a raw path."""
    decoded = _decode_percent(path)

    canonical_chars: list[str] = []
    shape_chars: list[str] = []
    signature = 0
    for ch in decoded:
        lowered = ch.lower()
        canonical_chars.append(lowered)
        shape_chars.append(_char_class(ch))
        # Folded into the same signature so case-only differences between
        # two paths collapse to one cache/report bucket instead of two.
        signature = (signature * _SIGNATURE_MULTIPLIER + ord(lowered)) % _SIGNATURE_MODULUS

    canonical = _collapse_slashes("".join(canonical_chars))
    shape = "".join(shape_chars)
    return canonical, shape, signature
