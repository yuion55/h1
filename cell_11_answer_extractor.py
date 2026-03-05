# cell_11_answer_extractor.py
"""
CTRL-MATH v5 — Answer Normalization (AnswerExtractor)

Handles: \\boxed{}, ANSWER: field, fractions, powers, factorials,
         keyword extraction, last-integer fallback.
Returns (primary, candidates, source, confidence).
Never raises — returns (0, [], "failed", 0.0) on all exceptions.
All extraction < 1ms.
"""

from __future__ import annotations

import math
import re
from typing import List, Tuple


# ── Modular inverse helper ─────────────────────────────────────────────────────

def _modinv(a: int, m: int) -> int:
    """Extended Euclidean modular inverse. Returns a^-1 mod m."""
    g, x, _ = _extended_gcd(a % m, m)
    if g != 1:
        return 0
    return x % m


def _extended_gcd(a: int, b: int) -> Tuple[int, int, int]:
    if a == 0:
        return b, 0, 1
    g, x, y = _extended_gcd(b % a, a)
    return g, y - (b // a) * x, x


# ── Extraction patterns ────────────────────────────────────────────────────────

_MODULI = [998_244_353, 10**9 + 7, 10**9 + 9, 1000]

_RE_BOXED    = re.compile(r"\\boxed\{([^}]+)\}")
_RE_ANSWER   = re.compile(r"ANSWER\s*[:=]\s*(-?\d+(?:/\d+)?)", re.IGNORECASE)
_RE_FRAC_TEX = re.compile(r"\\frac\{(\d+)\}\{(\d+)\}")
_RE_FRAC     = re.compile(r"(-?\d+)\s*/\s*(\d+)")
_RE_POWER    = re.compile(r"(-?\d+)\s*\^[\{]?(\d+)[\}]?")
_RE_FACT     = re.compile(r"(\d+)\s*!")
_RE_INTEGER  = re.compile(r"-?\d+")

_KEYWORDS_LAST = ["answer", "result", "therefore", "thus", "hence", "equals"]


class AnswerExtractor:
    """
    Normalises raw LLM output to a single integer answer.

    Extraction priority:
      1. \\boxed{}
      2. ANSWER: field
      3. \\frac{p}{q} — modular inverse mod standard moduli
      4. p/q plain fraction
      5. a^b power
      6. n! factorial
      7. Integer after answer keyword
      8. Last integer in text
    """

    ANSWER_MAX = 99999  # AIMO3 Progress Prize maximum answer value

    def extract(
        self, text, problem_modulus: int = 0
    ) -> Tuple[int, List[int], str, float]:
        """
        Extract integer answer from text.

        Args:
            text: Raw LLM output to extract from.
            problem_modulus: When > 1, fractions are resolved via modular
                inverse (e.g. for competition problems asking for answers
                mod p). When 0, non-integer fractions return 0.

        Returns:
            (primary_answer, candidates, source_name, confidence)
        Never raises. Accepts None or non-string input gracefully.
        """
        if text is None or not isinstance(text, str):
            return 0, [], "failed", 0.0
        try:
            return self._extract_impl(text, problem_modulus)
        except Exception:
            return 0, [], "failed", 0.0

    def _extract_impl(
        self, text: str, problem_modulus: int = 0
    ) -> Tuple[int, List[int], str, float]:
        candidates: List[int] = []

        # 1. \\boxed{}
        for m in _RE_BOXED.finditer(text):
            inner = m.group(1).strip()
            v     = self._parse_value(inner, problem_modulus)
            if v is not None:
                candidates.append(v)
                primary = max(0, v)
                return primary, candidates, "boxed", 0.95

        # 2. ANSWER: field
        m = _RE_ANSWER.search(text)
        if m:
            v = self._parse_value(m.group(1), problem_modulus)
            if v is not None:
                candidates.append(v)
                primary = max(0, v)
                return primary, candidates, "answer_field", 0.90

        # 3. \\frac{p}{q}
        for m in _RE_FRAC_TEX.finditer(text):
            p, q = int(m.group(1)), int(m.group(2))
            if q != 0:
                v = self._fraction_to_int(p, q, problem_modulus)
                if v:
                    candidates.append(v)

        # 4. p/q plain fraction
        for m in _RE_FRAC.finditer(text):
            p, q = int(m.group(1)), int(m.group(2))
            if q != 0:
                v = self._fraction_to_int(p, q, problem_modulus)
                if v:
                    candidates.append(v)

        # 5. a^b power
        for m in _RE_POWER.finditer(text):
            base, exp = int(m.group(1)), int(m.group(2))
            if 0 <= exp <= 60:
                try:
                    candidate = int(base ** exp)
                    candidates.append(abs(candidate))
                except Exception:
                    pass

        # 6. n! factorial
        for m in _RE_FACT.finditer(text):
            n = int(m.group(1))
            if 0 <= n <= 20:
                candidates.append(math.factorial(n))

        if candidates:
            primary = max(0, candidates[0])
            return primary, candidates, "expression", 0.75

        # 7. Integer after answer keyword
        text_lower = text.lower()
        for kw in _KEYWORDS_LAST:
            idx = text_lower.rfind(kw)
            if idx != -1:
                after = text[idx:]
                m     = _RE_INTEGER.search(after)
                if m:
                    v = int(m.group())
                    candidates.append(v)
                    primary = max(0, v)
                    return primary, candidates, f"keyword_{kw}", 0.60

        # 8. Last integer fallback
        all_ints = _RE_INTEGER.findall(text)
        if all_ints:
            v = int(all_ints[-1])
            candidates.append(v)
            primary = max(0, v)
            return primary, candidates, "last_integer", 0.40

        return 0, [], "not_found", 0.0

    def _parse_value(self, s: str, problem_modulus: int = 0) -> int | None:
        """Try to parse s as an integer directly, or via fraction/power."""
        s = s.strip()
        # Direct integer
        try:
            return int(s)
        except ValueError:
            pass
        # Fraction
        m = _RE_FRAC.match(s)
        if m:
            p, q = int(m.group(1)), int(m.group(2))
            if q != 0:
                return self._fraction_to_int(p, q, problem_modulus) or None
        return None

    def _fraction_to_int(self, p: int, q: int, problem_modulus: int = 0) -> int:
        """
        Convert fraction p/q to integer via:
          - exact division if q divides p
          - modular inverse only when problem_modulus > 1 is explicitly provided

        When problem_modulus is 0, non-integer fractions return 0 to avoid
        producing spurious large modular-inverse values as answers.
        """
        if q == 0:
            return 0
        if p % q == 0:
            return p // q
        if problem_modulus > 1:
            inv = _modinv(q, problem_modulus)
            if inv != 0:
                return (p * inv) % problem_modulus
        return 0  # Not an integer; don't guess
