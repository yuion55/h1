# cell_04_engine_v4.py
"""
CTRL-MATH v4 — Updated Transform Engine
(after cell_04f_geometry.py)

TransformEngineV4 orchestrates:
  Phase 1: 4 fast regex shortcuts (no LLM call)
  Phase 2: LLM extraction → route to deterministic solver if confidence >= 0.75
"""

from __future__ import annotations

import re
from typing import Any, Dict, Optional

import numpy as np

# TransformResult is defined in cell_04_transform_engine and re-exported here
from cell_04_transform_engine import TransformResult  # noqa: F401

try:
    from cell_02a_numba_nt import (
        vp_jit, vp_factorial_jit,
        lte_odd_minus_jit, lte_odd_plus_jit, lte_p2_minus_jit,
        fib_jit,
    )
except ImportError:
    def vp_jit(n, p): return 0
    def vp_factorial_jit(n, p): return 0
    def lte_odd_minus_jit(a, b, n, p): return 0
    def lte_odd_plus_jit(a, b, n, p): return 0
    def lte_p2_minus_jit(a, b, n): return 0
    def fib_jit(n, mod=0): return 0

try:
    from cell_05_cyclotomic import PHI_LEQ_8
except ImportError:
    PHI_LEQ_8 = None

try:
    from cell_03_mog_parser import MathState
except ImportError:
    from dataclasses import dataclass, field
    @dataclass
    class MathState:  # type: ignore[no-redef]
        problem_text: str = ""
        domain: Any = None
        modulus: int = 10**9 + 7
        variables: dict = field(default_factory=dict)
        constraints: list = field(default_factory=list)
        budget_remaining: int = 50
        facts: dict = field(default_factory=dict)
        answer: Any = None
        solved: bool = False

try:
    from cell_04a_extractor import LLMExtractor
except ImportError:
    LLMExtractor = None  # type: ignore[assignment,misc]

try:
    from cell_04b_linear_recurrence import LinearRecurrenceSolver
except ImportError:
    LinearRecurrenceSolver = None  # type: ignore[assignment,misc]

try:
    from cell_04c_combinatorics import CombinatoricsSolver, ensure_tables
except ImportError:
    CombinatoricsSolver = None  # type: ignore[assignment,misc]
    def ensure_tables(mod): pass

try:
    from cell_04d_number_theory import NumberTheorySolver
except ImportError:
    NumberTheorySolver = None  # type: ignore[assignment,misc]

try:
    from cell_04e_gf_solver import GFSolver
except ImportError:
    GFSolver = None  # type: ignore[assignment,misc]

try:
    from cell_04f_geometry import GeometrySolver
except ImportError:
    GeometrySolver = None  # type: ignore[assignment,misc]


# ── Confidence threshold ──────────────────────────────────────────────────────
CONFIDENCE_THRESHOLD: float = 0.75


# ── TransformEngineV4 ─────────────────────────────────────────────────────────
class TransformEngineV4:
    """
    CTRL-MATH v4 transform engine.

    Phase 1: 4 fast regex shortcuts — _try_lte_vp, _try_fibonacci,
             _try_digit_sum, _try_cyclotomic — run BEFORE any LLM call.
    Phase 2: LLM extraction → route to deterministic solver when
             confidence >= CONFIDENCE_THRESHOLD and category != "unknown".
    """

    def __init__(self, llm_model=None, llm_tokenizer=None):
        # Create LLM extractor if model is provided
        self._extractor = None
        if llm_model is not None and LLMExtractor is not None:
            self._extractor = LLMExtractor(llm_model, llm_tokenizer)

        # Map category names to solver classes
        self._solvers: Dict[str, Any] = {}
        if LinearRecurrenceSolver is not None:
            self._solvers["linear_recurrence"] = LinearRecurrenceSolver
        if CombinatoricsSolver is not None:
            self._solvers["combinatorics"] = CombinatoricsSolver
        if NumberTheorySolver is not None:
            self._solvers["number_theory"] = NumberTheorySolver
        if GFSolver is not None:
            self._solvers["generating_function"] = GFSolver
            self._solvers["polynomial_coeff"] = GFSolver
        if GeometrySolver is not None:
            self._solvers["geometry"] = GeometrySolver

        # Pre-build combinatorics tables
        ensure_tables(998_244_353)

    def apply(self, state: "MathState") -> Optional[TransformResult]:
        """
        Try all transforms in order.
        Phase 1 (no LLM): fast regex shortcuts.
        Phase 2 (LLM): extraction + deterministic solver.
        """
        text = state.problem_text

        # ── Phase 1: fast regex shortcuts ────────────────────────────────────
        for shortcut in (
            self._try_lte_vp,
            self._try_fibonacci,
            self._try_digit_sum,
            self._try_cyclotomic,
        ):
            result = shortcut(state, text)
            if result is not None and result.solved:
                return result

        # ── Phase 2: LLM extraction ───────────────────────────────────────────
        if self._extractor is None:
            return None

        extraction = self._extractor.extract(text)
        if (extraction.confidence < CONFIDENCE_THRESHOLD
                or extraction.category == "unknown"):
            return None

        solver_cls = self._solvers.get(extraction.category)
        if solver_cls is None:
            return None

        try:
            return solver_cls.solve(extraction.params, state.modulus)
        except Exception:
            return None

    # ── Phase 1 shortcuts ─────────────────────────────────────────────────────

    def _try_lte_vp(self, state: "MathState", text: str) -> Optional[TransformResult]:
        """
        Apply Lifting The Exponent lemma using JIT functions.
        Detects patterns like v_p(a^n ± b^n).
        """
        m = re.search(r'v_?(\d+)\s*\(\s*(\d+)\^n?\s*-\s*(\d+)\^n?\s*\)', text)
        if m:
            p = int(m.group(1))
            a = int(m.group(2))
            b = int(m.group(3))
            n_match = re.search(r'n\s*=\s*(\d+)', text)
            if n_match:
                n = int(n_match.group(1))
                if p == 2:
                    val = lte_p2_minus_jit(np.int64(a), np.int64(b), np.int64(n))
                else:
                    val = lte_odd_minus_jit(np.int64(a), np.int64(b),
                                            np.int64(n), np.int64(p))
                return TransformResult(
                    solved=True, answer=int(val) % state.modulus,
                    reduced_state=state,
                    certificate={"a": a, "b": b, "n": n, "p": p, "vp": int(val)},
                    transform_name="lte_vp_jit_v4",
                )
        return None

    def _try_fibonacci(self, state: "MathState", text: str) -> Optional[TransformResult]:
        """Solve Fibonacci-type problems using JIT matrix exponentiation."""
        m = re.search(r'[Ff]ibonacci.*?F_?\{?(\d+)\}?', text)
        if not m:
            m = re.search(r'F_?\{?(\d+)\}?.*?[Ff]ibonacci', text)
        if m:
            n = int(m.group(1))
            mod = state.modulus
            val = fib_jit(np.int64(n), np.int64(mod))
            return TransformResult(
                solved=True, answer=int(val),
                reduced_state=state,
                certificate={"n": n, "F_n": int(val)},
                transform_name="fibonacci_jit_v4",
            )
        return None

    def _try_digit_sum(self, state: "MathState", text: str) -> Optional[TransformResult]:
        """
        Apply digit sum / base-p analysis using vp_factorial_jit.
        Detects patterns involving digit sums in various bases.
        """
        if 'digit' not in text.lower() and 'base' not in text.lower():
            return None

        m_base = re.search(r'base\s+(\d+)', text, re.IGNORECASE)
        m_num = re.search(r'digit\s+sum.*?(\d+)', text, re.IGNORECASE)
        if not m_num:
            m_num = re.search(r'(\d+).*?digit\s+sum', text, re.IGNORECASE)

        if m_num:
            n = int(m_num.group(1))
            p = int(m_base.group(1)) if m_base else 10
            mod = state.modulus
            if p >= 2 and n > 0:
                vp_nfact = vp_factorial_jit(np.int64(n), np.int64(p))
                digit_sum = int(n - (p - 1) * vp_nfact)
                return TransformResult(
                    solved=True, answer=digit_sum % mod,
                    reduced_state=state,
                    certificate={"n": n, "base": p, "digit_sum": digit_sum},
                    transform_name="digit_sum_jit_v4",
                )
        return None

    def _try_cyclotomic(self, state: "MathState", text: str) -> Optional[TransformResult]:
        """
        Apply cyclotomic polynomial analysis using precomputed PHI_LEQ_8 table.
        No SymPy calls — uses only the precomputed table for n <= 8.
        """
        if ('cyclotomic' not in text.lower()
                and 'root of unity' not in text.lower()):
            return None
        if PHI_LEQ_8 is None:
            return None

        mod = state.modulus
        m = re.search(r'[Pp]hi_?\{?(\d+)\}?|cyclotomic.*?(\d+)', text)
        if m:
            n = int(m.group(1) or m.group(2))
            if n in PHI_LEQ_8:
                coeffs = PHI_LEQ_8[n]
                m_eval = re.search(
                    r'evaluate.*?at\s+(\d+)|at\s+x\s*=\s*(\d+)', text)
                if m_eval:
                    x = int(m_eval.group(1) or m_eval.group(2))
                    val = 0
                    x_pow = 1
                    for c in coeffs:
                        val += int(c) * x_pow
                        x_pow *= x
                    answer = val % mod
                else:
                    answer = (len(coeffs) - 1) % mod
                return TransformResult(
                    solved=True, answer=answer,
                    reduced_state=state,
                    certificate={"n": n, "degree": len(coeffs) - 1,
                                 "coeffs": coeffs.tolist()},
                    transform_name="cyclotomic_phi_table_v4",
                )
        return None
