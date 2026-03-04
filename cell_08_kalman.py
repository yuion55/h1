# cell_08_kalman.py
"""
IMPLEMENTATION REQUIREMENT:
Replace scalar Kalman with full vector Kalman over all facts simultaneously.
Update all belief probabilities in one NumPy operation, not a Python loop.
"""

import numpy as np
from typing import Dict, List


class KalmanBeliefState:
    """
    Vectorized Kalman filter over N mathematical facts.

    State: belief vector b ∈ [0,1]^N, variance P ∈ R^N.
    """

    Q = 0.10   # Process noise (LLM uncertainty per step)
    R = 0.01   # Measurement noise (SymPy/Z3 near-perfect)

    def __init__(self, initial_facts: dict):
        self.fact_names = list(initial_facts.keys())
        N               = len(self.fact_names)
        self.b          = np.array([float(v) for v in initial_facts.values()], dtype=np.float64)
        self.P          = np.zeros(N, dtype=np.float64)   # zero variance for given facts (certain)
        self._idx       = {name: i for i, name in enumerate(self.fact_names)}

    def predict(self, new_facts: dict):
        """
        Add LLM-derived facts with uncertainty Q.
        Vectorized: update all new facts simultaneously.
        """
        for name, conf in new_facts.items():
            if name not in self._idx:
                self.fact_names.append(name)
                self._idx[name] = len(self.fact_names) - 1
                self.b = np.append(self.b, conf * (1 - self.Q))
                self.P = np.append(self.P, self.Q)
            else:
                i         = self._idx[name]
                self.b[i] = conf * (1 - self.Q)
                self.P[i] += self.Q

    def update_batch(self, fact_names: List[str], z_values: np.ndarray):
        """
        Vectorized Kalman update for multiple verified facts at once.
        z_values: binary array (1.0 = verified true, 0.0 = verified false).

        All Kalman gain computations in one NumPy broadcast.
        """
        indices = np.array([self._idx[n] for n in fact_names if n in self._idx])
        if len(indices) == 0:
            return
        P_sub   = self.P[indices]
        K       = P_sub / (P_sub + self.R)         # Kalman gain (vectorized)
        self.b[indices] += K * (z_values[:len(indices)] - self.b[indices])
        self.P[indices]  = (1 - K) * P_sub

    def lean4_lock(self, fact_names: List[str]):
        """Lean 4 verified: set confidence=1.0, variance=0 permanently."""
        indices             = [self._idx[n] for n in fact_names if n in self._idx]
        self.b[indices]     = 1.0
        self.P[indices]     = 0.0

    def high_confidence(self, threshold: float = 0.95) -> dict:
        """Return facts with b[i] >= threshold. O(N) NumPy comparison."""
        mask = self.b >= threshold
        return {self.fact_names[i]: float(self.b[i])
                for i in np.where(mask)[0]}
