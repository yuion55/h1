# cell_08_prm.py
"""
CTRL-MATH v5 — Process Reward Model
Combines symbolic scoring (Numba JIT) and LLM scoring in a single batch call.

Combined score: 0.6 * llm_score + 0.4 * symbolic_score
"""

from __future__ import annotations

import re
from typing import List, Optional

import numpy as np
from numba import njit

# ── Known theorems and contradiction patterns ─────────────────────────────────

KNOWN_THEOREMS = [
    "fermat_little",
    "euler_totient",
    "chinese_remainder",
    "bezout",
    "lagrange_four_squares",
    "wilson",
    "quadratic_reciprocity",
    "cauchy_schwarz",
    "am_gm",
    "pigeonhole",
    "inclusion_exclusion",
    "stars_and_bars",
    "vieta",
    "newton_identities",
    "ptolemy",
    "power_mean",
    "jensen",
    "burnside",
    "cayley",
    "lucas",
    "catalan_ballot",
    "bertrand_postulate",
    "dirichlet_pigeonhole",
    "mobius_inversion",
    "legendre_symbol",
]

CONTRADICTION_PATTERNS = [
    r"0\s*=\s*[1-9]",    # 0 = nonzero
    r"[1-9]\s*=\s*0",    # nonzero = 0
    r"(\w+)\s*>\s*\1",   # x > x
    r"(\w+)\s*<\s*\1",   # x < x
    r"negative.*factorial",
    r"sqrt.*negative",
]

# Feature indices for symbolic scoring
_FEAT_HAS_EQUALS   = 0
_FEAT_HAS_THEOREM  = 1
_FEAT_CONTRADICTION= 2
_FEAT_HAS_NUMBER   = 3
_FEAT_STEP_LENGTH  = 4
_N_FEATURES        = 5


@njit(cache=True)
def symbolic_score_jit(features: np.ndarray) -> float:
    """
    Fast symbolic score from pre-extracted features.
    < 1μs per step.

    features[0] = has_equals (0/1)
    features[1] = has_known_theorem (0/1)
    features[2] = has_contradiction (0/1)  → penalize
    features[3] = has_number (0/1)
    features[4] = step_length_norm ∈ [0,1]
    """
    score = 0.0
    # Reward: equals sign means concrete progress
    score += 0.3 * features[0]
    # Reward: uses a known theorem
    score += 0.3 * features[1]
    # Penalize: contradiction detected
    score -= 0.8 * features[2]
    # Reward: concrete numeric result
    score += 0.2 * features[3]
    # Reward: reasonably-length step (not too short, not too long)
    length_score = features[4]
    if length_score > 0.02 and length_score < 0.9:
        score += 0.1
    # Clamp to [0, 1]
    if score < 0.0:
        score = 0.0
    if score > 1.0:
        score = 1.0
    return score


def _extract_symbolic_features(step: str) -> np.ndarray:
    """
    Extract symbolic features from a single solution step string.
    Returns np.ndarray of shape (_N_FEATURES,) dtype float64.
    """
    features = np.zeros(_N_FEATURES, dtype=np.float64)
    step_lower = step.lower()

    # has_equals (also match ≡, ≥, ≤)
    features[_FEAT_HAS_EQUALS] = 1.0 if re.search(r"[=≡≥≤]", step) else 0.0

    # has_known_theorem — partial word matching on each token of theorem name
    for thm in KNOWN_THEOREMS:
        # Try full phrase first, then individual tokens (≥4 chars)
        tokens = [t for t in thm.split("_") if len(t) >= 4]
        phrase = thm.replace("_", " ")
        if phrase in step_lower:
            features[_FEAT_HAS_THEOREM] = 1.0
            break
        if tokens and all(t in step_lower for t in tokens):
            features[_FEAT_HAS_THEOREM] = 1.0
            break
        # Single distinctive token match (e.g. "fermat" in text)
        if tokens and any(t in step_lower for t in tokens[:1]):
            features[_FEAT_HAS_THEOREM] = 1.0
            break

    # has_contradiction
    for pat in CONTRADICTION_PATTERNS:
        if re.search(pat, step, re.IGNORECASE):
            features[_FEAT_CONTRADICTION] = 1.0
            break

    # has_number
    features[_FEAT_HAS_NUMBER] = 1.0 if re.search(r"\d+", step) else 0.0

    # step_length_norm (clamp between 0 and 500 chars)
    features[_FEAT_STEP_LENGTH] = min(len(step), 500) / 500.0

    return features


class ProcessRewardModel:
    """
    Process Reward Model that scores each solution step.

    For each step, computes:
      combined = 0.6 * llm_score + 0.4 * symbolic_score

    LLM scoring uses a single model.forward() call for all K steps (batched).
    """

    LLM_WEIGHT      = 0.6
    SYMBOLIC_WEIGHT = 0.4

    def __init__(self, model=None, tokenizer=None, device: str = "cuda"):
        self.model     = model
        self.tokenizer = tokenizer
        self.device    = device

    def score_batch(self, steps: List[str]) -> np.ndarray:
        """
        Score a list of solution steps.
        Returns np.ndarray of shape (len(steps),) ∈ [0,1].
        """
        if not steps:
            return np.zeros(0, dtype=np.float64)

        n = len(steps)

        # Symbolic scores (vectorized via njit)
        feat_matrix = np.stack(
            [_extract_symbolic_features(s) for s in steps], axis=0
        )  # (n, _N_FEATURES)

        sym_scores = np.array(
            [symbolic_score_jit(feat_matrix[i]) for i in range(n)],
            dtype=np.float64,
        )

        # LLM scores
        if self.model is not None:
            llm_scores = self._llm_score_batch(steps)
        else:
            llm_scores = np.full(n, 0.5, dtype=np.float64)

        combined = self.LLM_WEIGHT * llm_scores + self.SYMBOLIC_WEIGHT * sym_scores
        return np.clip(combined, 0.0, 1.0)

    def _llm_score_batch(self, steps: List[str]) -> np.ndarray:
        """
        Score all steps in a single model.forward() call.

        Builds a batch of reward-prompts, runs one forward pass,
        extracts the logit for token "1" (good) vs "0" (bad) as score.
        """
        try:
            import torch

            prompts = [
                f"Rate this math solution step (0=wrong, 1=correct):\n{s}\nRating:"
                for s in steps
            ]
            enc = self.tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
            ).to(self.device)

            with torch.no_grad():
                logits = self.model(**enc).logits[:, -1, :]  # (n, vocab)

            tok_0 = self.tokenizer.encode("0", add_special_tokens=False)
            tok_1 = self.tokenizer.encode("1", add_special_tokens=False)

            id0 = tok_0[0] if tok_0 else 0
            id1 = tok_1[0] if tok_1 else 1

            p0 = logits[:, id0].float()
            p1 = logits[:, id1].float()
            scores_tensor = torch.softmax(torch.stack([p0, p1], dim=1), dim=1)[:, 1]
            return scores_tensor.cpu().numpy().astype(np.float64)

        except Exception:
            return np.full(len(steps), 0.5, dtype=np.float64)


# ── Warm up JIT ───────────────────────────────────────────────────────────────
def _warmup_jit() -> None:
    _f = np.zeros(_N_FEATURES, dtype=np.float64)
    _f[0] = 1.0
    symbolic_score_jit(_f)


_warmup_jit()
