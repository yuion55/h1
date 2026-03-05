# cell_12_time_allocator.py
"""
CTRL-MATH v5 — Competition Time Management

BudgetState dataclass + TimeAllocator class.
Difficulty estimation: easy/medium/hard based on keyword density, quantifiers, variables.
Hard cutoff at 3× budget, reserve 10% for retry pass.
allocate_budget() < 100μs.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


# ── BudgetState ───────────────────────────────────────────────────────────────

@dataclass
class BudgetState:
    """Tracks time and attempt budget for a single problem."""
    problem_id:      str
    allocated_sec:   float
    start_time:      float = field(default_factory=time.perf_counter)
    attempts:        int   = 0
    solved:          bool  = False
    answer:          int   = 0
    difficulty:      str   = "medium"     # easy / medium / hard
    retried:         bool  = False

    def elapsed(self) -> float:
        return time.perf_counter() - self.start_time

    def remaining(self) -> float:
        return max(0.0, self.allocated_sec - self.elapsed())

    def over_budget(self, factor: float = 1.0) -> bool:
        return self.elapsed() > self.allocated_sec * factor


# ── Difficulty keyword sets ───────────────────────────────────────────────────

_HARD_KEYWORDS = {
    "prove", "proof", "show that", "determine all", "find all",
    "number of", "how many", "maximum", "minimum", "optimal",
    "polynomial", "infinite", "sequence", "recurrence", "modular",
    "congruent", "prime", "divisible", "divisor", "greatest", "least",
}
_MEDIUM_KEYWORDS = {
    "find", "compute", "evaluate", "calculate", "simplify",
    "factor", "solve", "root", "sum", "product",
}
_QUANTIFIER_RE = re.compile(
    r"\b(for all|for every|there exists|infinitely many|at least|at most)\b",
    re.IGNORECASE,
)
_VARIABLE_RE = re.compile(r"\b[a-zA-Z]\b")


# ── TimeAllocator ─────────────────────────────────────────────────────────────

class TimeAllocator:
    """
    Allocates time budgets across multiple competition problems.

    Strategy:
      - Total time split among N problems
      - 10% reserved for a retry pass over failed problems
      - Hard cutoff at 3× individual budget
      - Easy → less time, Hard → more time
    """

    RETRY_RESERVE    = 0.10   # 10% of total for retries
    HARD_CUTOFF_MULT = 3.0    # abandon after 3× budget
    EASY_MULT        = 0.6
    MEDIUM_MULT      = 1.0
    HARD_MULT        = 1.6

    def __init__(
        self,
        total_seconds: float = 9 * 3600.0,   # 9 hours (typical Kaggle limit)
        n_problems:    int   = 50,
    ):
        self.total_seconds = total_seconds
        self.n_problems    = n_problems
        self._states:   Dict[str, BudgetState] = {}
        self._history:  List[Tuple[str, bool, float]] = []  # (id, solved, elapsed)

    # ── Core methods ──────────────────────────────────────────────────────────

    def allocate_budget(self, problem_id: str, problem_text: str) -> BudgetState:
        """
        Allocate a time budget for problem_id.
        < 100μs.
        """
        difficulty = self.estimate_difficulty(problem_text)
        base_sec   = (
            self.total_seconds * (1 - self.RETRY_RESERVE)
        ) / self.n_problems

        mult = {
            "easy":   self.EASY_MULT,
            "medium": self.MEDIUM_MULT,
            "hard":   self.HARD_MULT,
        }.get(difficulty, self.MEDIUM_MULT)

        allocated = base_sec * mult
        state = BudgetState(
            problem_id    = problem_id,
            allocated_sec = allocated,
            difficulty    = difficulty,
        )
        self._states[problem_id] = state
        return state

    def should_abandon(self, state: BudgetState) -> bool:
        """Return True if problem should be abandoned (3× budget exceeded)."""
        return state.over_budget(self.HARD_CUTOFF_MULT)

    def time_remaining_for(self, state: BudgetState) -> float:
        """Return remaining seconds for this problem."""
        return state.remaining()

    def record_result(
        self, state: BudgetState, solved: bool, answer: int = 0
    ) -> None:
        """Record outcome of a problem attempt."""
        state.solved  = solved
        state.answer  = answer
        self._history.append((state.problem_id, solved, state.elapsed()))

    def get_retry_candidates(self, top_k: int = 10) -> List[str]:
        """
        Return problem IDs that failed and have retry time budget available.
        Uses reserved 10% of total time.
        """
        retry_budget   = self.total_seconds * self.RETRY_RESERVE
        used_in_retry  = 0.0
        candidates     = []

        # Sort by elapsed time ascending (cheapest first)
        failed = [
            (pid, elapsed)
            for pid, solved, elapsed in self._history
            if not solved
        ]
        failed.sort(key=lambda x: x[1])

        for pid, elapsed in failed[:top_k]:
            if used_in_retry + elapsed > retry_budget:
                break
            candidates.append(pid)
            used_in_retry += elapsed

        return candidates

    def estimate_difficulty(self, problem_text: str) -> str:
        """
        Estimate problem difficulty as 'easy', 'medium', or 'hard'.

        Heuristics:
          - hard keyword density
          - number of universal/existential quantifiers
          - number of distinct single-letter variables
        """
        text_lower = problem_text.lower()

        # Count hard keyword hits
        hard_hits   = sum(1 for kw in _HARD_KEYWORDS if kw in text_lower)
        medium_hits = sum(1 for kw in _MEDIUM_KEYWORDS if kw in text_lower)

        # Count quantifiers
        q_count = len(_QUANTIFIER_RE.findall(problem_text))

        # Count distinct variables
        var_count = len(set(_VARIABLE_RE.findall(problem_text)))

        # Score
        hard_score   = hard_hits * 2 + q_count * 3 + max(0, var_count - 2)
        medium_score = medium_hits

        if hard_score >= 6:
            return "hard"
        if hard_score >= 2 or medium_score >= 3:
            return "medium"
        return "easy"
