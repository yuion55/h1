# cell_04c_combinatorics.py
"""
CTRL-MATH v4 — Combinatorics Solver
(after cell_04b_linear_recurrence.py)

Implements all common combinatorial functions with Numba JIT acceleration:
binomial, Catalan, Stirling numbers, derangements, partitions, Bell numbers,
inclusion-exclusion. GPU batch via CuPy where available.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict

import numpy as np
from numba import njit, prange

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

try:
    import cupy as cp  # type: ignore
    _CUPY_AVAILABLE = True
except ImportError:
    _CUPY_AVAILABLE = False

# ── Module-level globals ──────────────────────────────────────────────────────
COMB_MOD: int = 998_244_353
COMB_MAX_N: int = 1_000_000

_FACT = np.ones(COMB_MAX_N + 1, dtype=np.int64)
_INV_FACT = np.ones(COMB_MAX_N + 1, dtype=np.int64)
_TABLES_BUILT: bool = False
_TABLES_MOD: int = -1


# ── Factorial table builder ───────────────────────────────────────────────────
@njit(cache=True)
def _build_fact_table(fact: np.ndarray, inv_fact: np.ndarray,
                      N: int, mod: int) -> None:
    """
    Build factorial and inverse-factorial tables up to N (inclusive) mod p.
    Uses Fermat's little theorem for inverses: (n!)^{-1} = (n!)^{p-2} mod p.
    """
    fact[0] = np.int64(1)
    for i in range(1, N + 1):
        fact[i] = fact[i - 1] * np.int64(i) % mod

    # Compute (N!)^{p-2} mod p
    base = fact[N]
    exp = np.int64(mod - 2)
    result = np.int64(1)
    b = base % mod
    while exp > 0:
        if exp & 1:
            result = result * b % mod
        b = b * b % mod
        exp >>= 1
    inv_fact[N] = result

    for i in range(N - 1, -1, -1):
        inv_fact[i] = inv_fact[i + 1] * np.int64(i + 1) % mod


def ensure_tables(mod: int) -> None:
    """Build tables if not built or if mod changed."""
    global _TABLES_BUILT, _TABLES_MOD, _FACT, _INV_FACT
    if _TABLES_BUILT and _TABLES_MOD == mod:
        return
    _build_fact_table(_FACT, _INV_FACT, COMB_MAX_N, np.int64(mod))
    _TABLES_BUILT = True
    _TABLES_MOD = mod


# Build tables at import time
ensure_tables(COMB_MOD)


# ── Binomial coefficient ──────────────────────────────────────────────────────
@njit(cache=True)
def binom_fast(n: int, k: int, mod: int,
               fact: np.ndarray, inv_fact: np.ndarray) -> int:
    """
    Compute C(n, k) mod p.
    - O(1) table lookup for n <= COMB_MAX_N.
    - O(log_p n) Lucas theorem for larger n (iterates base-p digits).
    """
    _COMB_MAX = np.int64(1_000_000)
    if k < 0 or k > n:
        return np.int64(0)
    if k == 0 or k == n:
        return np.int64(1)
    if n <= _COMB_MAX:
        return fact[n] * inv_fact[k] % mod * inv_fact[n - k] % mod
    # Lucas theorem: C(n, k) = prod C(n_i, k_i) mod p for base-p digits
    result = np.int64(1)
    nn, kk = np.int64(n), np.int64(k)
    while nn > 0 or kk > 0:
        ni = nn % mod
        ki = kk % mod
        nn //= mod
        kk //= mod
        if ki > ni:
            return np.int64(0)
        if ni <= _COMB_MAX:
            result = result * fact[ni] % mod * inv_fact[ki] % mod * inv_fact[ni - ki] % mod
        else:
            # ni is still large — recurse via the loop (handled next iteration)
            # For very large ni, use direct computation
            top = np.int64(1)
            bot = np.int64(1)
            for j in range(ki):
                top = top * ((ni - j) % mod) % mod
                bot = bot * (j + 1) % mod
            # bot_inv via Fermat
            bot_inv = np.int64(1)
            b = bot
            e = np.int64(mod - 2)
            while e > 0:
                if e & 1:
                    bot_inv = bot_inv * b % mod
                b = b * b % mod
                e >>= 1
            result = result * top % mod * bot_inv % mod
    return result


# ── Catalan numbers ───────────────────────────────────────────────────────────
@njit(cache=True)
def catalan_jit(n: int, mod: int,
                fact: np.ndarray, inv_fact: np.ndarray) -> int:
    """
    C_n = C(2n, n) / (n+1) mod p.
    < 1μs after JIT warm-up.
    """
    if n == 0:
        return np.int64(1)
    binom_val = fact[2 * n] * inv_fact[n] % mod * inv_fact[n] % mod
    # Multiply by (n+1)^{-1} mod p
    inv_np1 = np.int64(1)
    b = np.int64((n + 1) % mod)
    e = np.int64(mod - 2)
    while e > 0:
        if e & 1:
            inv_np1 = inv_np1 * b % mod
        b = b * b % mod
        e >>= 1
    return binom_val * inv_np1 % mod


# ── Stirling numbers of the second kind ──────────────────────────────────────
@njit(parallel=True, cache=True)
def stirling2_batch(n_vals: np.ndarray, k: int, mod: int,
                    fact: np.ndarray, inv_fact: np.ndarray) -> np.ndarray:
    """
    Compute S(n, k) for each n in n_vals in parallel.
    S(n, k) = (1/k!) * sum_{j=0}^{k} (-1)^{k-j} * C(k, j) * j^n.
    Uses modular exponentiation for j^n.
    """
    result = np.empty(len(n_vals), dtype=np.int64)
    kk = np.int64(k)
    for idx in prange(len(n_vals)):
        nn = np.int64(n_vals[idx])
        s = np.int64(0)
        for j in range(kk + 1):
            # C(k, j) * (-1)^{k-j} * j^n
            c_kj = fact[kk] * inv_fact[j] % mod * inv_fact[kk - j] % mod
            # j^n mod p
            jn = np.int64(1)
            base = np.int64(j % mod)
            exp = nn
            while exp > 0:
                if exp & 1:
                    jn = jn * base % mod
                base = base * base % mod
                exp >>= 1
            term = c_kj * jn % mod
            if (kk - j) % 2 == 0:
                s = (s + term) % mod
            else:
                s = (s - term + mod) % mod
        # Multiply by inv(k!)
        s = s * inv_fact[kk] % mod
        result[idx] = s
    return result


# ── Derangements ──────────────────────────────────────────────────────────────
@njit(cache=True)
def derangements_jit(n: int, mod: int,
                     fact: np.ndarray, inv_fact: np.ndarray) -> int:
    """
    D(n) = n! * sum_{k=0}^{n} (-1)^k / k!  mod p.
    O(n) time.
    """
    if n == 0:
        return np.int64(1)
    s = np.int64(0)
    for k in range(n + 1):
        term = inv_fact[k]
        if k % 2 == 0:
            s = (s + term) % mod
        else:
            s = (s - term + mod) % mod
    return fact[n] * s % mod


# ── Integer partitions ────────────────────────────────────────────────────────
@njit(cache=True)
def partition_jit(n: int, mod: int) -> int:
    """
    Compute p(n) using the pentagonal number theorem DP.
    p(n) = sum_{k != 0} (-1)^{k+1} * p(n - k(3k-1)/2)
    O(n * sqrt(n)) time.
    """
    p = np.zeros(n + 1, dtype=np.int64)
    p[0] = np.int64(1)
    for i in range(1, n + 1):
        s = np.int64(0)
        k = np.int64(1)
        while True:
            # Generalized pentagonal: g_k = k*(3k-1)//2, g_{-k} = k*(3k+1)//2
            g1 = k * (np.int64(3) * k - np.int64(1)) // np.int64(2)
            g2 = k * (np.int64(3) * k + np.int64(1)) // np.int64(2)
            if g1 > i:
                break
            sign = np.int64(1) if k % 2 == 1 else np.int64(-1)
            s = (s + sign * p[i - g1]) % mod
            if g2 <= i:
                s = (s + sign * p[i - g2]) % mod
            k += 1
        p[i] = (s + mod) % mod
    return p[n]


@njit(parallel=True, cache=True)
def partition_batch_jit(n_vals: np.ndarray, mod: int) -> np.ndarray:
    """
    Compute p(n) for all n in n_vals.
    Builds table to max(n_vals) once, reads off all values.
    """
    max_n = np.int64(0)
    for i in range(len(n_vals)):
        if n_vals[i] > max_n:
            max_n = n_vals[i]
    # Build full table
    p = np.zeros(max_n + 1, dtype=np.int64)
    p[0] = np.int64(1)
    for i in range(1, max_n + 1):
        s = np.int64(0)
        k = np.int64(1)
        while True:
            g1 = k * (np.int64(3) * k - np.int64(1)) // np.int64(2)
            g2 = k * (np.int64(3) * k + np.int64(1)) // np.int64(2)
            if g1 > i:
                break
            sign = np.int64(1) if k % 2 == 1 else np.int64(-1)
            s = (s + sign * p[i - g1]) % mod
            if g2 <= i:
                s = (s + sign * p[i - g2]) % mod
            k += 1
        p[i] = (s + mod) % mod
    # Read off results in parallel
    result = np.empty(len(n_vals), dtype=np.int64)
    for idx in prange(len(n_vals)):
        result[idx] = p[n_vals[idx]]
    return result


# ── Bell numbers ──────────────────────────────────────────────────────────────
@njit(cache=True)
def bell_jit(n: int, mod: int) -> int:
    """
    Bell number B(n) using the Bell triangle method. O(n²).
    B(0)=1, B(n) = sum_{k=0}^{n-1} C(n-1, k) * B(k).
    """
    if n == 0:
        return np.int64(1)
    # Use Bell triangle: triangle[i][0] = B(i), triangle[i][j] = triangle[i-1][j-1] + triangle[i][j-1]
    row = np.zeros(n + 1, dtype=np.int64)
    row[0] = np.int64(1)  # B(0) = 1
    for i in range(1, n + 1):
        new_row = np.zeros(n + 1, dtype=np.int64)
        new_row[0] = row[i - 1]  # B(i) = last element of previous row
        for j in range(1, i + 1):
            new_row[j] = (new_row[j - 1] + row[j - 1]) % mod
        row = new_row
    return row[0]


# ── Inclusion-exclusion ───────────────────────────────────────────────────────
@njit(cache=True)
def inclusion_exclusion_jit(set_sizes: np.ndarray,
                             intersection_sizes: np.ndarray,
                             mod: int) -> int:
    """
    Inclusion-exclusion for the union of n <= 20 sets.
    intersection_sizes is a flat array of length 2^n indexed by bitmask.
    intersection_sizes[mask] = |intersection of sets indicated by mask|.
    Returns |union| = sum_{mask != 0} (-1)^{|mask|+1} * intersection_sizes[mask].
    """
    n = len(set_sizes)
    result = np.int64(0)
    total_masks = np.int64(1) << n
    for mask in range(1, total_masks):
        # popcount
        bits = np.int64(0)
        m = np.int64(mask)
        while m:
            bits += m & np.int64(1)
            m >>= 1
        sign = np.int64(1) if bits % 2 == 1 else np.int64(-1)
        result = (result + sign * intersection_sizes[mask]) % mod
    return (result + mod) % mod


# ── GPU batch binomial ────────────────────────────────────────────────────────
def binom_batch_gpu(n_vals: np.ndarray, k_vals: np.ndarray, mod: int) -> np.ndarray:
    """
    Batch binomial coefficients. Uses CuPy GPU if available, else CPU loop.
    """
    if _CUPY_AVAILABLE:
        try:
            ensure_tables(mod)
            n_gpu = cp.asarray(n_vals)
            k_gpu = cp.asarray(k_vals)
            fact_gpu = cp.asarray(_FACT)
            inv_fact_gpu = cp.asarray(_INV_FACT)
            mask = (k_gpu >= 0) & (k_gpu <= n_gpu) & (n_gpu <= COMB_MAX_N)
            result_gpu = cp.zeros(len(n_vals), dtype=cp.int64)
            result_gpu[mask] = (
                fact_gpu[n_gpu[mask]] *
                inv_fact_gpu[k_gpu[mask]] % mod *
                inv_fact_gpu[n_gpu[mask] - k_gpu[mask]] % mod
            )
            return cp.asnumpy(result_gpu)
        except Exception:
            pass
    # CPU fallback
    ensure_tables(mod)
    result = np.zeros(len(n_vals), dtype=np.int64)
    for i in range(len(n_vals)):
        result[i] = binom_fast(
            np.int64(n_vals[i]), np.int64(k_vals[i]),
            np.int64(mod), _FACT, _INV_FACT,
        )
    return result


# ── CombinatoricsSolver ───────────────────────────────────────────────────────
class CombinatoricsSolver:
    """
    Dispatches combinatorics problems to the appropriate JIT function
    based on params["sub_type"].
    """

    @staticmethod
    def solve(params: dict, mod: int) -> "TransformResult":
        """
        Solve a combinatorics problem given params dict and modulus.

        params["sub_type"]: binomial, catalan, stirling2, derangement,
                            partition, bell
        """
        actual_mod = int(params.get("modulus", mod))
        ensure_tables(actual_mod)
        sub = params.get("sub_type", "")

        if sub == "binomial":
            n = int(params["n"])
            k = int(params["k"])
            answer = int(binom_fast(np.int64(n), np.int64(k),
                                    np.int64(actual_mod), _FACT, _INV_FACT))
            return TransformResult(
                solved=True, answer=answer, reduced_state=None,
                certificate={"sub_type": "binomial", "n": n, "k": k},
                transform_name="combinatorics_v4",
            )

        elif sub == "catalan":
            n = int(params["n"])
            answer = int(catalan_jit(np.int64(n), np.int64(actual_mod),
                                     _FACT, _INV_FACT))
            return TransformResult(
                solved=True, answer=answer, reduced_state=None,
                certificate={"sub_type": "catalan", "n": n},
                transform_name="combinatorics_v4",
            )

        elif sub == "stirling2":
            n = int(params["n"])
            k = int(params["k"])
            n_vals = np.array([n], dtype=np.int64)
            arr = stirling2_batch(n_vals, np.int64(k), np.int64(actual_mod),
                                  _FACT, _INV_FACT)
            answer = int(arr[0])
            return TransformResult(
                solved=True, answer=answer, reduced_state=None,
                certificate={"sub_type": "stirling2", "n": n, "k": k},
                transform_name="combinatorics_v4",
            )

        elif sub == "derangement":
            n = int(params["n"])
            answer = int(derangements_jit(np.int64(n), np.int64(actual_mod),
                                          _FACT, _INV_FACT))
            return TransformResult(
                solved=True, answer=answer, reduced_state=None,
                certificate={"sub_type": "derangement", "n": n},
                transform_name="combinatorics_v4",
            )

        elif sub == "partition":
            n = int(params["n"])
            answer = int(partition_jit(np.int64(n), np.int64(actual_mod)))
            return TransformResult(
                solved=True, answer=answer, reduced_state=None,
                certificate={"sub_type": "partition", "n": n},
                transform_name="combinatorics_v4",
            )

        elif sub == "bell":
            n = int(params["n"])
            answer = int(bell_jit(np.int64(n), np.int64(actual_mod)))
            return TransformResult(
                solved=True, answer=answer, reduced_state=None,
                certificate={"sub_type": "bell", "n": n},
                transform_name="combinatorics_v4",
            )

        return TransformResult(
            solved=False, answer=None, reduced_state=None,
            certificate={"error": f"unknown sub_type: {sub}"},
            transform_name="combinatorics_v4",
        )
