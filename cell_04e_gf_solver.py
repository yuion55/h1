# cell_04e_gf_solver.py
"""
CTRL-MATH v4 — Generating Function Solver
(after cell_04d_number_theory.py)

Implements rational GF coefficient extraction and polynomial power-mod truncation,
routing to Kitamasa for the linear recurrence case.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict

import numpy as np
from numba import njit

try:
    from cell_02a_numba_nt import poly_mul_ntt
except ImportError:
    def poly_mul_ntt(f: np.ndarray, g: np.ndarray, mod: int = None) -> np.ndarray:
        n = 1
        while n < len(f) + len(g) - 1:
            n <<= 1
        fa = np.zeros(n, dtype=np.int64); fa[:len(f)] = f
        ga = np.zeros(n, dtype=np.int64); ga[:len(g)] = g
        h = np.fft.irfft(np.fft.rfft(fa) * np.fft.rfft(ga), n=n)
        h = np.round(h).astype(np.int64)[:len(f) + len(g) - 1]
        if mod:
            h = h % mod
        return h

try:
    from cell_04b_linear_recurrence import kitamasa_flint
except ImportError:
    def kitamasa_flint(c, init, n, mod):
        c = list(c); init = list(init); k = len(c)
        if n < k:
            return int(init[n]) % mod
        a = [int(x) % mod for x in init]
        for i in range(k, n + 1):
            nxt = sum(c[j] * a[-k + j] for j in range(k)) % mod
            a.append(nxt)
            if len(a) > k + 1:
                a.pop(0)
        return a[-1]

try:
    from cell_04_transform_engine import TransformResult
except ImportError:
    @dataclass
    class TransformResult:  # type: ignore[no-redef]
        solved: bool
        answer: Any
        reduced_state: Any
        certificate: Dict[str, Any]
        transform_name: str


# ── Polynomial mod over Z/p ───────────────────────────────────────────────────
@njit(cache=True)
def poly_mod_jit(a: np.ndarray, m: np.ndarray, mod: int) -> np.ndarray:
    """
    Reduce polynomial a mod m over Z/mod. Ascending coefficient order.
    Trims trailing zeros. Uses Fermat inverse for leading coefficient.
    """
    a = a.copy() % mod
    deg_m = len(m) - 1
    if deg_m < 0:
        return np.zeros(1, dtype=np.int64)
    # Lead inverse
    lead = m[deg_m] % mod
    lead_inv = np.int64(1)
    b = np.int64(lead % mod)
    e = np.int64(mod - 2)
    while e > 0:
        if e & 1:
            lead_inv = lead_inv * b % mod
        b = b * b % mod
        e >>= 1

    while len(a) > deg_m:
        deg_a = len(a) - 1
        if a[deg_a] == 0:
            a = a[:deg_a]
            continue
        coef = a[deg_a] * lead_inv % mod
        for i in range(deg_m + 1):
            a[deg_a - deg_m + i] = (a[deg_a - deg_m + i] - coef * m[i]) % mod
        a = a[:deg_a]

    # Trim trailing zeros, ensure minimum length 1
    end = len(a)
    while end > 1 and a[end - 1] == 0:
        end -= 1
    return a[:end]


# ── Polynomial power-mod with truncation ─────────────────────────────────────
def poly_power_mod_trunc(base_poly: np.ndarray, power: int,
                         mod_poly: np.ndarray, mod: int,
                         max_degree: int) -> np.ndarray:
    """
    Compute base_poly^power mod mod_poly over Z/mod,
    truncating at max_degree after each multiplication step.
    Uses poly_mul_ntt from cell_02a_numba_nt.
    """
    base_poly = np.asarray(base_poly, dtype=np.int64) % mod
    mod_poly  = np.asarray(mod_poly, dtype=np.int64) % mod

    # result = 1 (constant polynomial)
    result = np.zeros(max(len(mod_poly), 2), dtype=np.int64)
    result[0] = np.int64(1)
    base = base_poly.copy()

    exp = int(power)
    while exp > 0:
        if exp & 1:
            product = poly_mul_ntt(result, base)
            product = product % mod
            # Truncate
            if len(product) > max_degree + 1:
                product = product[:max_degree + 1]
            result = _poly_mod_py(product, mod_poly, mod)
        product = poly_mul_ntt(base, base)
        product = product % mod
        if len(product) > max_degree + 1:
            product = product[:max_degree + 1]
        base = _poly_mod_py(product, mod_poly, mod)
        exp >>= 1

    return result


def _poly_mod_py(a: np.ndarray, m: np.ndarray, mod: int) -> np.ndarray:
    """Python wrapper for polynomial reduction."""
    a = a.copy() % mod
    deg_m = len(m) - 1
    if deg_m <= 0:
        return a[:0] if deg_m < 0 else np.zeros(1, dtype=np.int64)
    lead_inv = pow(int(m[deg_m]), mod - 2, mod)
    while len(a) > deg_m:
        deg_a = len(a) - 1
        if int(a[deg_a]) == 0:
            a = a[:deg_a]
            continue
        coef = int(a[deg_a]) * lead_inv % mod
        for i in range(deg_m + 1):
            a[deg_a - deg_m + i] = (int(a[deg_a - deg_m + i]) - coef * int(m[i])) % mod
        a = a[:deg_a]
    result = np.zeros(deg_m, dtype=np.int64)
    for i in range(min(len(a), deg_m)):
        result[i] = int(a[i]) % mod
    return result


# ── Rational GF coefficient extraction ───────────────────────────────────────
def rational_gf_coefficient(P: np.ndarray, Q: np.ndarray,
                             n: int, mod: int) -> int:
    """
    Extract [x^n] P(x) / Q(x) mod p by:
    1. Normalizing Q[0] to 1 (divide by Q[0] using Fermat inverse).
    2. Extracting recurrence coefficients c from Q: c[i] = -Q[i+1] mod p.
    3. Computing initial values a_0..a_{k-1} by power series division.
    4. Routing to kitamasa_flint(c, init, n, mod).
    """
    P = np.asarray(P, dtype=np.int64) % mod
    Q = np.asarray(Q, dtype=np.int64) % mod
    k = len(Q) - 1  # order of recurrence

    if k == 0:
        # P(x) / Q[0] — just a polynomial
        q0_inv = pow(int(Q[0]), mod - 2, mod)
        if n < len(P):
            return int(P[n]) * q0_inv % mod
        return 0

    # Step 1: normalize so Q[0] = 1
    q0_inv = pow(int(Q[0]), mod - 2, mod)
    Q = Q * q0_inv % mod
    P = P * q0_inv % mod

    # Step 2: recurrence coefficients c[i] = -Q[i+1] mod p
    # a[n] = c[1]*a[n-1] + ... + c[k]*a[n-k] for n >= deg(P)+1
    c = np.array([(-int(Q[i + 1])) % mod for i in range(k)], dtype=np.int64)
    # c[0] corresponds to a[n-1], c[k-1] to a[n-k]

    # Step 3: compute initial values a[0..k-1] by power series P/Q
    # We expand P(x)/Q(x) = sum_{i>=0} a[i] x^i
    # Q * A = P  (in formal power series)
    # Q[0]*a[n] + Q[1]*a[n-1] + ... + Q[k]*a[n-k] = P[n] (P[n]=0 for n>deg P)
    # Since Q[0]=1: a[n] = P[n] - Q[1]*a[n-1] - ... - Q[min(k,n)]*a[n-k]
    init = np.zeros(k, dtype=np.int64)
    for i in range(k):
        p_i = int(P[i]) if i < len(P) else 0
        s = p_i
        for j in range(1, i + 1):
            if j <= k:
                s = (s - int(Q[j]) * int(init[i - j])) % mod
        init[i] = int(s) % mod

    # Step 4: route to kitamasa
    return kitamasa_flint(c, init, n, mod)


# ── GFSolver ──────────────────────────────────────────────────────────────────
class GFSolver:
    """Dispatches generating function problems to the appropriate solver."""

    @staticmethod
    def solve(params: dict, mod: int) -> "TransformResult":
        """
        Solve a GF problem.

        params keys:
          - "denominator" → rational GF coefficient extraction
          - "operations"  → polynomial product powers
        """
        actual_mod = int(params.get("modulus", mod))
        n = int(params.get("n", 0))

        if "denominator" in params:
            P = np.array(params.get("numerator", [1]), dtype=np.int64)
            Q = np.array(params["denominator"], dtype=np.int64)
            answer = rational_gf_coefficient(P, Q, n, actual_mod)
            return TransformResult(
                solved=True, answer=answer, reduced_state=None,
                certificate={"method": "rational_gf", "n": n},
                transform_name="gf_solver_v4",
            )

        elif "operations" in params:
            # Polynomial product-power: compute [x^n] prod_i base_i(x)^{exp_i}
            operations = params["operations"]  # list of {poly: [...], power: int}
            max_degree = n + 1
            result = np.zeros(max_degree + 1, dtype=np.int64)
            result[0] = np.int64(1)
            for op in operations:
                base_poly = np.array(op["poly"], dtype=np.int64)
                power = int(op.get("power", 1))
                mod_poly = np.zeros(max_degree + 2, dtype=np.int64)
                mod_poly[max_degree + 1] = np.int64(1)  # x^{n+1}
                powered = poly_power_mod_trunc(base_poly, power, mod_poly,
                                               actual_mod, max_degree)
                product = poly_mul_ntt(result, powered)
                product = product % actual_mod
                if len(product) > max_degree + 1:
                    product = product[:max_degree + 1]
                result = _poly_mod_py(product, mod_poly, actual_mod)
            answer = int(result[n]) if n < len(result) else 0
            return TransformResult(
                solved=True, answer=answer, reduced_state=None,
                certificate={"method": "poly_power_mod", "n": n},
                transform_name="gf_solver_v4",
            )

        return TransformResult(
            solved=False, answer=None, reduced_state=None,
            certificate={"error": "unknown GF params"},
            transform_name="gf_solver_v4",
        )
