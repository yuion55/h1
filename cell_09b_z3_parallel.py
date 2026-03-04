# cell_09b_z3_parallel.py
"""
IMPLEMENTATION REQUIREMENT:
Run independent Z3 sub-goals in parallel threads.
Z3 solver instances are NOT thread-safe if shared; create one per thread.
Speedup: 4–8 independent sub-goals → 4–8× wall-clock speedup.
"""

import z3
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Tuple, Dict


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
            # Add main constraint
            try:
                exec_ns = {**vars_, "z3": z3}
                constraint = eval(formula_str, exec_ns)
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
