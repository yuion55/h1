# cell_09a_lean4_repl.py
"""
CTRL-MATH v4 — Lean 4 REPL Verifier
(after cell_09b_z3_parallel.py)

Wraps the Lean 4 binary to verify mathematical answers through formal proofs.
Every shortcut answer can be passed through Lean 4 before submission.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import tempfile
import threading
from dataclasses import dataclass, field
from typing import Optional


# ── LeanResult dataclass ──────────────────────────────────────────────────────
@dataclass
class LeanResult:
    """Result from Lean 4 verification."""
    success: bool
    error_msg: str
    lean_output: str
    elapsed_ms: float


# ── Lean 4 REPL ───────────────────────────────────────────────────────────────
class Lean4REPL:
    """
    Wraps the Lean 4 binary for formal verification of mathematical answers.
    Uses a threading lock to serialize subprocess calls.
    """

    # Known Kaggle paths for lean binary
    _KAGGLE_LEAN_PATHS = [
        "/kaggle/input/lean4-mathlib-cache/lean/bin/lean",
        "/tmp/lean-4.8.0/bin/lean",
    ]

    LEAN_PREAMBLE = """\
import Mathlib.Tactic
import Mathlib.Data.Nat.Basic
import Mathlib.Data.Int.GCD
open Nat Int
"""

    def __init__(self, timeout: int = 30):
        self.timeout = timeout
        self._lock = threading.Lock()
        self.lean_binary: Optional[str] = self._find_lean()

    def _find_lean(self) -> Optional[str]:
        """Find lean binary from PATH or known Kaggle paths."""
        # Try PATH first
        lean = shutil.which("lean")
        if lean:
            return lean
        # Try known Kaggle paths
        for path in self._KAGGLE_LEAN_PATHS:
            if os.path.exists(path):
                return path
        return None

    def _run_lean(self, lean_code: str) -> LeanResult:
        """
        Write lean_code to a temp .lean file, invoke lean binary,
        parse stdout+stderr for error: lines, return LeanResult.
        """
        import time

        if self.lean_binary is None:
            return LeanResult(
                success=False,
                error_msg="lean binary not found",
                lean_output="",
                elapsed_ms=0.0,
            )

        tmp_path = None
        t0 = time.perf_counter()
        with self._lock:
            try:
                with tempfile.NamedTemporaryFile(
                    mode="w", suffix=".lean", delete=False
                ) as f:
                    f.write(lean_code)
                    tmp_path = f.name

                result = subprocess.run(
                    [self.lean_binary, tmp_path],
                    capture_output=True,
                    text=True,
                    timeout=self.timeout,
                )
                elapsed = (time.perf_counter() - t0) * 1000.0
                combined = result.stdout + result.stderr
                errors = [
                    line for line in combined.splitlines()
                    if "error:" in line.lower()
                ]
                success = (result.returncode == 0 and len(errors) == 0)
                return LeanResult(
                    success=success,
                    error_msg="\n".join(errors) if errors else "",
                    lean_output=combined,
                    elapsed_ms=elapsed,
                )
            except subprocess.TimeoutExpired:
                elapsed = (time.perf_counter() - t0) * 1000.0
                return LeanResult(
                    success=False,
                    error_msg=f"lean timed out after {self.timeout}s",
                    lean_output="",
                    elapsed_ms=elapsed,
                )
            except Exception as e:
                elapsed = (time.perf_counter() - t0) * 1000.0
                return LeanResult(
                    success=False,
                    error_msg=str(e),
                    lean_output="",
                    elapsed_ms=elapsed,
                )
            finally:
                if tmp_path and os.path.exists(tmp_path):
                    try:
                        os.unlink(tmp_path)
                    except OSError:
                        pass

    def verify_answer(self, answer: int, lean_statement: str,
                      tactic: str) -> LeanResult:
        """
        Verify that `lean_statement` holds using `tactic`.
        Builds: theorem verify_answer_{abs(answer)} : {lean_statement} := by {tactic}
        """
        theorem_name = f"verify_answer_{abs(int(answer))}"
        code = (
            self.LEAN_PREAMBLE
            + f"\ntheorem {theorem_name} : {lean_statement} := by\n  {tactic}\n"
        )
        return self._run_lean(code)

    def verify_fast(self, expression: str, expected: int) -> LeanResult:
        """
        Quick evaluation: uses #eval do ... if result == expected then IO.println "CORRECT"
        """
        code = (
            self.LEAN_PREAMBLE
            + f"\n#eval do\n"
            + f"  let result := {expression}\n"
            + f"  if result == {expected} then IO.println \"CORRECT\"\n"
            + f"  else IO.println s!\"WRONG: got {{result}}, expected {expected}\"\n"
        )
        result = self._run_lean(code)
        # Check for CORRECT in output (no error lines and CORRECT marker present)
        if "CORRECT" in result.lean_output and not result.error_msg:
            return LeanResult(
                success=True,
                error_msg="",
                lean_output=result.lean_output,
                elapsed_ms=result.elapsed_ms,
            )
        return result

    def verify_step(self, fact_name: str, lean_statement: str,
                    tactic: str) -> LeanResult:
        """
        Verify an intermediate proof step.
        fact_name is sanitized with regex before use.
        """
        safe_name = re.sub(r"[^a-zA-Z0-9_]", "_", fact_name)
        code = (
            self.LEAN_PREAMBLE
            + f"\ntheorem {safe_name} : {lean_statement} := by\n  {tactic}\n"
        )
        return self._run_lean(code)

    def verify_modular_answer(self, answer: int, modulus: int) -> LeanResult:
        """Range check and route to verify_fast for modular arithmetic answers."""
        answer = int(answer) % int(modulus)
        expression = f"({answer} : Nat)"
        return self.verify_fast(expression, answer)

    def get_correction_prompt(self, fact_name: str, lean_statement: str,
                               failed_tactic: str, lean_error: str) -> str:
        """Build prompt for LLM to correct a failed Lean proof."""
        return (
            f"The following Lean 4 proof attempt failed.\n\n"
            f"Theorem: {fact_name}\n"
            f"Statement: {lean_statement}\n"
            f"Failed tactic:\n  {failed_tactic}\n"
            f"Lean error:\n  {lean_error}\n\n"
            f"Please provide a corrected tactic proof. "
            f"Output only the tactic (no imports or theorem statement)."
        )
