# cell_14_verification_ladder.py
"""
CTRL-MATH v5 — 4-Level Verification Pipeline

Level 0: Range check          (< 1μs)
Level 1: SymPy check          (< 100ms)
Level 2: Z3 check             (< 5s)
Level 3: Lean 4 proof         (≤ 30s)

Integrates with KalmanBeliefState from cell_08_kalman.py (Kalman belief filter).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class VerificationResult:
    """Result from any verification level."""
    passed:     bool
    level:      int
    elapsed_ms: float
    message:    str   = ""
    confidence: float = 0.0


class VerificationLadder:
    """
    4-level hierarchical verifier.

    Each level only runs if the previous level passed (or is skipped).
    The ladder stops at the first definitive pass.
    """

    # Competition answer bounds — configurable via constructor
    _ANSWER_LO = 0
    _ANSWER_HI = 999  # AIMO3 Progress Prize default

    def __init__(
        self,
        lean_repl=None,
        z3_checker=None,
        kalman_state=None,
        lean_timeout: int = 30,
        answer_lo: int = 0,
        answer_hi: int = 999,
    ):
        self.lean       = lean_repl
        self.z3         = z3_checker
        self.kalman     = kalman_state
        self.lean_timeout = lean_timeout
        self._ANSWER_LO = answer_lo
        self._ANSWER_HI = answer_hi

    # ── Public API ─────────────────────────────────────────────────────────────

    def verify(
        self,
        answer: int,
        problem: str       = "",
        sympy_expr: str    = "",
        z3_formula: str    = "",
        z3_bounds: dict    = None,
        lean_statement: str = "",
        lean_tactic: str   = "",
    ) -> VerificationResult:
        """
        Run verification ladder until a definitive result or exhaustion.
        Returns the highest-confidence result achieved.
        """
        # Level 0 always runs
        r0 = self._level0(answer)
        if not r0.passed:
            return r0

        # Level 1
        if sympy_expr:
            r1 = self._level1(answer, sympy_expr, problem)
            if r1.passed:
                self._update_kalman("level1_verified", 0.85)
                return r1
            if r1.confidence > 0.9:   # definitive failure
                return r1

        # Level 2
        if z3_formula and self.z3 is not None:
            r2 = self._level2(answer, z3_formula, z3_bounds or {})
            if r2.passed:
                self._update_kalman("level2_verified", 0.95)
                return r2

        # Level 3
        if lean_statement and self.lean is not None:
            r3 = self._level3(answer, lean_statement, lean_tactic)
            if r3.passed:
                self._update_kalman("level3_verified", 1.0)
                return r3
            return r3

        # Default: level 0 pass with low confidence
        r0.confidence = 0.5
        return r0

    # ── Levels ─────────────────────────────────────────────────────────────────

    def _level0(self, answer: int) -> VerificationResult:
        """
        Range check: answer_lo ≤ answer ≤ answer_hi.
        < 1μs.
        """
        t0 = time.perf_counter()
        try:
            answer_int = int(answer)
            passed     = self._ANSWER_LO <= answer_int <= self._ANSWER_HI
        except Exception:
            passed = False
        elapsed = (time.perf_counter() - t0) * 1000.0
        return VerificationResult(
            passed     = passed,
            level      = 0,
            elapsed_ms = elapsed,
            message    = "range_ok" if passed else f"out of range: {answer}",
            confidence = 0.3 if passed else 0.0,
        )

    def _level1(
        self, answer: int, sympy_expr: str, problem: str = ""
    ) -> VerificationResult:
        """
        SymPy symbolic check.
        < 100ms.
        """
        t0 = time.perf_counter()
        try:
            import sympy as sp
            expr   = sp.sympify(sympy_expr, evaluate=True)
            result = int(sp.simplify(expr))
            passed = result == int(answer)
        except Exception as e:
            elapsed = (time.perf_counter() - t0) * 1000.0
            return VerificationResult(
                passed     = False,
                level      = 1,
                elapsed_ms = elapsed,
                message    = str(e),
                confidence = 0.0,
            )
        elapsed = (time.perf_counter() - t0) * 1000.0
        return VerificationResult(
            passed     = passed,
            level      = 1,
            elapsed_ms = elapsed,
            message    = "sympy_ok" if passed else f"sympy got {result} ≠ {answer}",
            confidence = 0.85 if passed else 0.95,
        )

    def _level2(
        self, answer: int, z3_formula: str, z3_bounds: Dict[str, Tuple[int, int]]
    ) -> VerificationResult:
        """
        Z3 SAT check.
        < 5s.
        """
        t0 = time.perf_counter()
        if self.z3 is None:
            elapsed = (time.perf_counter() - t0) * 1000.0
            return VerificationResult(
                passed=False, level=2, elapsed_ms=elapsed,
                message="z3_checker not available", confidence=0.0,
            )
        try:
            results = self.z3.check_all([(z3_formula, z3_bounds)])
            passed  = bool(results[0]) if results else False
        except Exception as e:
            elapsed = (time.perf_counter() - t0) * 1000.0
            return VerificationResult(
                passed=False, level=2, elapsed_ms=elapsed,
                message=str(e), confidence=0.0,
            )
        elapsed = (time.perf_counter() - t0) * 1000.0
        return VerificationResult(
            passed     = passed,
            level      = 2,
            elapsed_ms = elapsed,
            message    = "z3_sat" if passed else "z3_unsat",
            confidence = 0.95 if passed else 0.80,
        )

    def _level3(
        self, answer: int, lean_statement: str, lean_tactic: str
    ) -> VerificationResult:
        """
        Lean 4 formal proof.
        ≤ 30s.
        """
        t0 = time.perf_counter()
        if self.lean is None:
            elapsed = (time.perf_counter() - t0) * 1000.0
            return VerificationResult(
                passed=False, level=3, elapsed_ms=elapsed,
                message="lean_repl not available", confidence=0.0,
            )
        try:
            lean_result = self.lean.verify_answer(
                answer, lean_statement, lean_tactic
            )
            passed  = lean_result.success
            message = lean_result.error_msg if not passed else "lean_ok"
        except Exception as e:
            elapsed = (time.perf_counter() - t0) * 1000.0
            return VerificationResult(
                passed=False, level=3, elapsed_ms=elapsed,
                message=str(e), confidence=0.0,
            )
        elapsed = (time.perf_counter() - t0) * 1000.0
        return VerificationResult(
            passed     = passed,
            level      = 3,
            elapsed_ms = elapsed,
            message    = message,
            confidence = 1.0 if passed else 0.0,
        )

    # ── Kalman integration ─────────────────────────────────────────────────────

    def _update_kalman(self, fact_name: str, confidence: float) -> None:
        """Update KalmanBeliefState if available."""
        if self.kalman is None:
            return
        try:
            import numpy as np
            self.kalman.update_batch([fact_name], np.array([confidence]))
        except Exception:
            pass
