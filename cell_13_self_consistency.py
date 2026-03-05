# cell_13_self_consistency.py
"""
CTRL-MATH v5 — Self-Consistency Voting

Vectorized majority vote using np.unique + np.argmax.
K=3 default; escalates to K=5 if confidence < 0.5.
vote() < 100μs.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np


class SelfConsistencyChecker:
    """
    Aggregates multiple independent rollout answers via majority vote.

    Usage:
        checker = SelfConsistencyChecker(k=3)
        answer, confidence = checker.check(answers)
        if checker.should_escalate(confidence):
            # run more rollouts
    """

    ESCALATE_K       = 5
    ESCALATE_THRESH  = 0.5

    def __init__(self, k: int = 3):
        self.k = k

    def check(
        self, answers: List[int]
    ) -> Tuple[int, float, bool]:
        """
        Run majority vote over answers.

        Returns:
            (best_answer, confidence, should_escalate)
        """
        if not answers:
            return 0, 0.0, True

        best, confidence = self.vote(answers)
        escalate = self.should_escalate(confidence)
        return best, confidence, escalate

    def vote(self, answers: List[int]) -> Tuple[int, float]:
        """
        Vectorized majority vote.
        < 100μs.

        Returns (majority_answer, fraction_of_votes_for_majority).
        """
        if not answers:
            return 0, 0.0

        arr              = np.array(answers, dtype=np.int64)
        values, counts   = np.unique(arr, return_counts=True)
        best_idx         = int(np.argmax(counts))
        best_answer      = int(values[best_idx])
        confidence       = float(counts[best_idx]) / len(answers)
        return best_answer, confidence

    def should_escalate(self, confidence: float) -> bool:
        """Return True if confidence is below threshold → run more rollouts."""
        return confidence < self.ESCALATE_THRESH

    def check_with_escalation(
        self,
        initial_answers: List[int],
        extra_answers: Optional[List[int]] = None,
    ) -> Tuple[int, float]:
        """
        If initial confidence is too low and extra_answers provided,
        merge them and re-vote.
        """
        best, conf, escalate = self.check(initial_answers)

        if escalate and extra_answers:
            merged = initial_answers + extra_answers
            best, conf = self.vote(merged)

        return best, conf
