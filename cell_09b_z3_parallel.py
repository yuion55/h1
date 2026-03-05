# cell_09b_z3_parallel.py
"""
IMPLEMENTATION REQUIREMENT:
Run independent Z3 sub-goals in parallel threads.
Z3 solver instances are NOT thread-safe if shared; create one per thread.
Speedup: 4–8 independent sub-goals → 4–8× wall-clock speedup.
"""

import ast
import z3
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Tuple, Dict

# ── Sandboxed formula evaluation ──────────────────────────────────────────────

# Whitelist of allowed names in formula evaluation (z3 functions + operators)
_SAFE_Z3_NAMES = {
    "And": z3.And,
    "Or": z3.Or,
    "Not": z3.Not,
    "Implies": z3.Implies,
    "If": z3.If,
    "Distinct": z3.Distinct,
    "Sum": z3.Sum,
    "Product": z3.Product,
    "IntVal": z3.IntVal,
    "RealVal": z3.RealVal,
    "Bool": z3.Bool,
    "Int": z3.Int,
    "Real": z3.Real,
    "ForAll": z3.ForAll,
    "Exists": z3.Exists,
    "Abs": lambda x: z3.If(x >= 0, x, -x),
    "True": z3.BoolVal(True),
    "False": z3.BoolVal(False),
}


def _safe_eval_formula(formula_str: str, vars_: dict):
    """
    Evaluate a Z3 formula string in a sandboxed namespace.

    Only z3 functions and declared variables are accessible.
    Built-in functions (__builtins__) are blocked to prevent code injection.

    Falls back to ast.literal_eval for simple constant expressions.
    """
    # Try ast.literal_eval first for simple constant values
    try:
        return ast.literal_eval(formula_str)
    except (ValueError, SyntaxError):
        pass

    # Validate formula structure: reject obviously dangerous patterns
    _BLOCKED_PATTERNS = ["__import__", "__builtins__", "exec(", "eval(",
                         "compile(", "open(", "getattr(", "setattr(",
                         "delattr(", "globals(", "locals(", "breakpoint("]
    formula_lower = formula_str.lower()
    for pat in _BLOCKED_PATTERNS:
        if pat in formula_lower:
            raise ValueError(f"Blocked pattern in formula: {pat}")

    # Build restricted namespace: only z3 functions + declared variables
    exec_ns = {"__builtins__": {}}
    exec_ns.update(_SAFE_Z3_NAMES)
    exec_ns["z3"] = z3
    exec_ns["vars_"] = vars_
    exec_ns.update(vars_)

    return eval(formula_str, exec_ns)  # noqa: S307


class ParallelZ3Checker:
    """
    Parallel Z3 verification: runs independent sub-goals in separate threads,
    each with its own Z3 solver instance (thread-safe isolation).
    """

    def __init__(self, n_workers: int = 4):
        self.executor = ThreadPoolExecutor(max_workers=n_workers)

    def check_all(self, sub_goals: List[Tuple]) -> List[bool]:
        """
        sub_goals: list of (formula_str, variable_bounds_dict) tuples.
        Each is checked in its own Z3 solver instance.
        Returns list of booleans (True = SAT/provable, False = UNSAT/unprovable).

        Note: results are returned in the same order as sub_goals.
        """
        def check_one(goal):
            formula_str, bounds = goal
            s = z3.Solver()
            s.set("timeout", 5000)  # 5s per sub-goal
            # Create Z3 integer variables
            vars_ = {v: z3.Int(v) for v in bounds.keys()}
            for v, (lo, hi) in bounds.items():
                s.add(vars_[v] >= lo, vars_[v] <= hi)
            # Add main constraint via sandboxed eval
            try:
                constraint = _safe_eval_formula(formula_str, vars_)
                s.add(constraint)
                result = s.check()
                return result == z3.sat
            except Exception:
                return False

        # Submit all goals with their index for ordered result collection
        future_to_idx = {
            self.executor.submit(check_one, goal): i
            for i, goal in enumerate(sub_goals)
        }
        results = [False] * len(sub_goals)
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                results[idx] = future.result()
            except Exception:
                results[idx] = False
        return results
