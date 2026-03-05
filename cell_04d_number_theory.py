# cell_04d_number_theory.py
"""
CTRL-MATH v4 — Number Theory Solver
(after cell_04c_combinatorics.py)

Implements linear sieve for phi/mu/tau/sigma1, CRT, baby-step giant-step
discrete log, and related multiplicative function summation — all JIT-compiled.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Tuple

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


# ── Linear sieve (O(N) — phi, mu, tau, sigma1, primes) ───────────────────────
@njit(cache=True)
def linear_sieve_all(N: int) -> tuple:
    """
    O(N) linear sieve computing phi, mu, tau, sigma1, and the primes array
    simultaneously using multiplicativity.

    Returns (phi, mu, tau, sigma1, primes_array) — all int64 arrays.

    Multiplicativity rules for prime p:
      If p ∤ i:
        phi[i*p]    = phi[i]    * (p - 1)
        mu[i*p]     = -mu[i]
        tau[i*p]    = tau[i]    * 2
        sigma1[i*p] = sigma1[i] * (p + 1)
      If p | i (i = p^e * q where gcd(q,p)=1):
        phi[i*p]    = phi[i]    * p
        mu[i*p]     = 0
        tau[i*p]    = tau[i]    * (e+2) / (e+1)   (track e separately)
        sigma1[i*p] = sigma1[i] + sigma1[i/p] * p^{e+1}   (complex)
    """
    phi    = np.zeros(N + 1, dtype=np.int64)
    mu     = np.zeros(N + 1, dtype=np.int64)
    tau    = np.zeros(N + 1, dtype=np.int64)
    sigma1 = np.zeros(N + 1, dtype=np.int64)
    # smallest_prime[i] = smallest prime factor of i (0 for 1 and primes)
    sp     = np.zeros(N + 1, dtype=np.int64)
    # prime_exp[i] = exponent of sp[i] in i (used for tau/sigma1)
    exp_sp = np.zeros(N + 1, dtype=np.int64)

    primes = np.zeros(N + 1, dtype=np.int64)
    prime_cnt = np.int64(0)

    phi[1]    = np.int64(1)
    mu[1]     = np.int64(1)
    tau[1]    = np.int64(1)
    sigma1[1] = np.int64(1)

    for i in range(2, N + 1):
        if sp[i] == 0:
            # i is prime
            sp[i]     = np.int64(i)
            exp_sp[i] = np.int64(1)
            phi[i]    = np.int64(i - 1)
            mu[i]     = np.int64(-1)
            tau[i]    = np.int64(2)
            sigma1[i] = np.int64(i + 1)
            primes[prime_cnt] = np.int64(i)
            prime_cnt += 1

        # Mark composites: for each prime p <= sp[i]
        for j in range(prime_cnt):
            p = primes[j]
            if p > sp[i] or np.int64(i) * p > N:
                break
            ip = np.int64(i) * p
            sp[ip] = p
            if i % p == 0:
                # p | i
                exp_sp[ip] = exp_sp[i] + np.int64(1)
                e = exp_sp[ip]
                phi[ip]    = phi[i] * p
                mu[ip]     = np.int64(0)
                # tau[i*p] = tau[i] * (e+1) / e, since tau[p^e * m] = (e+1)*tau[m]
                tau[ip]    = tau[i] // e * (e + np.int64(1))
                i_div_p = np.int64(i) // p
                # p^{e-1}: track incrementally using sigma1[i//p] chain
                # sigma1[i*p] = sigma1[i] + sigma1[i/p] * p^{exp_sp[i]}
                # (since exp_sp[ip] = exp_sp[i]+1, so p^{e-1} = p^{exp_sp[i]})
                # Compute p^{exp_sp[i]} = p * p^{exp_sp[i]-1} incrementally
                p_e = _powmod_nt(p, exp_sp[i], np.int64(10**18))  # p^{exp_sp[i]}
                sigma1[ip] = sigma1[i] + sigma1[i_div_p] * p_e
            else:
                # p ∤ i
                exp_sp[ip] = np.int64(1)
                phi[ip]    = phi[i] * (p - np.int64(1))
                mu[ip]     = -mu[i]
                tau[ip]    = tau[i] * np.int64(2)
                sigma1[ip] = sigma1[i] * (p + np.int64(1))

    return phi, mu, tau, sigma1, primes[:prime_cnt]


# ── Sum of Euler phi up to N ──────────────────────────────────────────────────
@njit(cache=True)
def sum_phi_upto(N: int, mod: int) -> int:
    """Sum phi(1) + phi(2) + ... + phi(N) mod p via linear sieve."""
    phi, _, _, _, _ = linear_sieve_all(N)
    s = np.int64(0)
    for i in range(1, N + 1):
        s = (s + phi[i]) % mod
    return s


# ── Sum of multiplicative functions ──────────────────────────────────────────
@njit(cache=True)
def sum_mult_func_upto(N: int, func_type: int, k_param: int, mod: int) -> int:
    """
    Sum multiplicative function values from 1 to N mod p.
    func_type: 0=phi, 1=mu, 2=tau, 3=sigma1.
    k_param: unused (reserved for sigma_k generalization).
    """
    phi, mu, tau, sigma1, _ = linear_sieve_all(N)
    s = np.int64(0)
    if func_type == 0:
        for i in range(1, N + 1):
            s = (s + phi[i]) % mod
    elif func_type == 1:
        for i in range(1, N + 1):
            s = (s + mu[i] + mod) % mod
    elif func_type == 2:
        for i in range(1, N + 1):
            s = (s + tau[i]) % mod
    else:
        for i in range(1, N + 1):
            s = (s + sigma1[i]) % mod
    return s


# ── Chinese Remainder Theorem ─────────────────────────────────────────────────
@njit(cache=True)
def crt_jit(remainders: np.ndarray, moduli: np.ndarray) -> tuple:
    """
    Iterative CRT via extended GCD. Handles non-coprime moduli.
    Returns (x, lcm) or (-1, -1) if no solution exists.
    """
    if len(remainders) == 0:
        return np.int64(0), np.int64(1)

    x   = remainders[0] % moduli[0]
    lcm = moduli[0]

    for i in range(1, len(remainders)):
        r = remainders[i] % moduli[i]
        m = moduli[i]
        # Solve: x + lcm*t ≡ r (mod m)
        # gcd(lcm, m) | (r - x) required
        g = _gcd_jit(lcm, m)
        if (r - x) % g != 0:
            return np.int64(-1), np.int64(-1)
        # Extended GCD: find u, v s.t. lcm*u + m*v = g
        u = _modinv_ext_jit(lcm // g, m // g)
        t = ((r - x) // g * u) % (m // g)
        x = x + lcm * t
        lcm = lcm // g * m
        x = x % lcm

    return x, lcm


@njit(cache=True)
def _gcd_jit(a: int, b: int) -> int:
    """Euclidean GCD."""
    a, b = np.int64(abs(a)), np.int64(abs(b))
    while b:
        a, b = b, a % b
    return a


@njit(cache=True)
def _modinv_ext_jit(a: int, m: int) -> int:
    """Modular inverse of a mod m (m > 1, gcd(a,m)=1) via extended Euclidean."""
    a = np.int64(a % m)
    if a < 0:
        a += m
    old_r, r = a, m
    old_s, s = np.int64(1), np.int64(0)
    while r != 0:
        q = old_r // r
        old_r, r = r, old_r - q * r
        old_s, s = s, old_s - q * s
    return (old_s % m + m) % m


# ── Discrete logarithm (Baby-step Giant-step) ─────────────────────────────────
@njit(cache=True)
def discrete_log_bsgs(g: int, h: int, p: int) -> int:
    """
    Baby-step giant-step: find x in [0, p-1] such that g^x ≡ h (mod p).
    Returns x or -1 if no solution.

    Uses two parallel int64 arrays as an open-addressing hash table (size =
    2*ceil(sqrt(p)) + 10) since Numba @njit doesn't support Python dicts.
    """
    # Integer square root via Newton's method (no math.isqrt in @njit)
    sq = np.int64(1)
    while sq * sq < p:
        sq += np.int64(1)
    m = sq
    table_size = np.int64(2 * m + 10)

    # Hash table: keys[slot] = baby_val, vals[slot] = step_index
    keys = np.full(table_size, np.int64(-1), dtype=np.int64)
    vals = np.zeros(table_size, dtype=np.int64)

    # Baby steps: store g^j -> j for j = 0..m-1
    gj = np.int64(1)
    for j in range(m):
        baby = np.int64(gj % p)
        slot = baby % table_size
        # Linear probing
        while keys[slot] != np.int64(-1) and keys[slot] != baby:
            slot = (slot + np.int64(1)) % table_size
        keys[slot] = baby
        vals[slot] = np.int64(j)
        gj = gj * g % p

    # Giant steps: g^{-m} = (g^m)^{-1}
    # Compute g^m
    gm = _powmod_nt(np.int64(g), np.int64(m), np.int64(p))
    gm_inv = _powmod_nt(gm, np.int64(p - 2), np.int64(p))

    # Giant step: h * (g^{-m})^i for i = 0..m
    hh = np.int64(h % p)
    for i in range(m + 1):
        # Look up hh in hash table
        slot = hh % table_size
        while keys[slot] != np.int64(-1):
            if keys[slot] == hh:
                return np.int64(i) * m + vals[slot]
            slot = (slot + np.int64(1)) % table_size
        hh = hh * gm_inv % p

    return np.int64(-1)


@njit(cache=True)
def _powmod_nt(base: int, exp: int, mod: int) -> int:
    """Modular exponentiation inside @njit context."""
    result = np.int64(1)
    b = np.int64(base % mod)
    e = np.int64(exp)
    while e > 0:
        if e & 1:
            result = result * b % mod
        b = b * b % mod
        e >>= 1
    return result


# ── Euler product over primes ─────────────────────────────────────────────────
@njit(parallel=True, cache=True)
def euler_product_jit(primes: np.ndarray, N_bound: int,
                      factor_type: int, param: int, mod: int) -> int:
    """
    Compute product_{p <= N_bound, p prime} f(p) mod p in parallel.
    factor_type: 0 = (1 - 1/p), 1 = (1 + 1/p), 2 = (p-1)/p, 3 = user-defined
    """
    # Filter primes up to N_bound
    count = np.int64(0)
    for i in range(len(primes)):
        if primes[i] <= N_bound:
            count += 1
        else:
            break

    # Compute each factor in parallel
    factors = np.ones(count, dtype=np.int64)
    for i in prange(count):
        p = primes[i]
        if factor_type == 0:
            # (1 - p^{-1}) = (p-1)/p -> (p-1) * modinv(p)
            inv_p = _powmod_nt(p, mod - np.int64(2), mod)
            factors[i] = (p - np.int64(1)) * inv_p % mod
        elif factor_type == 1:
            # (1 + p^{-1}) = (p+1)/p
            inv_p = _powmod_nt(p, mod - np.int64(2), mod)
            factors[i] = (p + np.int64(1)) * inv_p % mod
        elif factor_type == 2:
            # (p-1)/p same as type 0
            inv_p = _powmod_nt(p, mod - np.int64(2), mod)
            factors[i] = (p - np.int64(1)) * inv_p % mod
        else:
            # param-th power of p
            factors[i] = _powmod_nt(p, np.int64(param), mod)

    # Reduce product
    result = np.int64(1)
    for i in range(count):
        result = result * factors[i] % mod
    return result


# ── NumberTheorySolver ────────────────────────────────────────────────────────
class NumberTheorySolver:
    """Dispatches number theory problems to JIT solvers."""

    @staticmethod
    def solve(params: dict, mod: int) -> "TransformResult":
        """
        Solve a number theory problem.
        params["sub_type"]: phi_sum, mobius_sum, mult_sum, crt, discrete_log.
        """
        actual_mod = int(params.get("modulus", mod))
        sub = params.get("sub_type", "")

        if sub == "phi_sum":
            N = int(params["N"])
            answer = int(sum_phi_upto(np.int64(N), np.int64(actual_mod)))
            return TransformResult(
                solved=True, answer=answer, reduced_state=None,
                certificate={"sub_type": "phi_sum", "N": N},
                transform_name="number_theory_v4",
            )

        elif sub == "mobius_sum":
            N = int(params["N"])
            answer = int(sum_mult_func_upto(
                np.int64(N), np.int64(1), np.int64(0), np.int64(actual_mod)))
            return TransformResult(
                solved=True, answer=answer, reduced_state=None,
                certificate={"sub_type": "mobius_sum", "N": N},
                transform_name="number_theory_v4",
            )

        elif sub == "mult_sum":
            N = int(params["N"])
            func_type = int(params.get("func_type", 0))
            k_param = int(params.get("k_param", 0))
            answer = int(sum_mult_func_upto(
                np.int64(N), np.int64(func_type),
                np.int64(k_param), np.int64(actual_mod)))
            return TransformResult(
                solved=True, answer=answer, reduced_state=None,
                certificate={"sub_type": "mult_sum", "N": N, "func_type": func_type},
                transform_name="number_theory_v4",
            )

        elif sub == "crt":
            rem = np.array(params["remainders"], dtype=np.int64)
            mods = np.array(params["moduli"], dtype=np.int64)
            x, lcm = crt_jit(rem, mods)
            return TransformResult(
                solved=True, answer=(int(x), int(lcm)), reduced_state=None,
                certificate={"sub_type": "crt", "x": int(x), "lcm": int(lcm)},
                transform_name="number_theory_v4",
            )

        elif sub == "discrete_log":
            g = int(params["g"])
            h = int(params["h"])
            p = int(params["p"])
            x = discrete_log_bsgs(np.int64(g), np.int64(h), np.int64(p))
            return TransformResult(
                solved=True, answer=int(x), reduced_state=None,
                certificate={"sub_type": "discrete_log", "g": g, "h": h, "p": p},
                transform_name="number_theory_v4",
            )

        return TransformResult(
            solved=False, answer=None, reduced_state=None,
            certificate={"error": f"unknown sub_type: {sub}"},
            transform_name="number_theory_v4",
        )
