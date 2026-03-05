# cell_11b_benchmarks_v4.py
"""
CTRL-MATH v4 — Extended Benchmark Suite
(after cell_11_benchmarks.py)

All correctness assertions and speedup assertions are hard assert statements.
Run this file to verify the complete v4 solver stack.
"""

import time

import numpy as np

# ── bench helper ──────────────────────────────────────────────────────────────
def bench(fn, *args, n_runs=100, warmup=10):
    """Run fn(*args) n_runs times after warmup, return mean elapsed seconds."""
    for _ in range(warmup):
        fn(*args)
    t0 = time.perf_counter()
    for _ in range(n_runs):
        fn(*args)
    return (time.perf_counter() - t0) / n_runs


MOD = 998_244_353

print("=" * 70)
print("CTRL-MATH v4 — Extended Benchmark Suite")
print("=" * 70)

# ── Imports ───────────────────────────────────────────────────────────────────
from cell_04c_combinatorics import (
    ensure_tables, binom_fast, catalan_jit, stirling2_batch,
    derangements_jit, partition_jit, bell_jit,
    _FACT, _INV_FACT,
)
from cell_04b_linear_recurrence import berlekamp_massey_jit, kitamasa_flint
from cell_04d_number_theory import (
    linear_sieve_all, sum_phi_upto, crt_jit, discrete_log_bsgs,
)
from cell_04e_gf_solver import rational_gf_coefficient
from cell_04f_geometry import shoelace_exact, picks_theorem

ensure_tables(MOD)

# ─────────────────────────────────────────────────────────────────────────────
# CORRECTNESS ASSERTIONS
# ─────────────────────────────────────────────────────────────────────────────

print("\n── Correctness checks ──")

# kitamasa_flint correctness
assert kitamasa_flint(
    np.array([1, 1], dtype=np.int64),
    np.array([0, 1], dtype=np.int64),
    10, MOD,
) == 55, "kitamasa F_10 wrong"

assert kitamasa_flint(
    np.array([1, 1], dtype=np.int64),
    np.array([0, 1], dtype=np.int64),
    0, MOD,
) == 0, "kitamasa F_0 wrong"

assert kitamasa_flint(
    np.array([1, 1], dtype=np.int64),
    np.array([0, 1], dtype=np.int64),
    1, MOD,
) == 1, "kitamasa F_1 wrong"

print("  ✓ kitamasa_flint Fibonacci correctness")

# berlekamp_massey_jit correctness
fib_terms = np.array([0, 1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89], dtype=np.int64)
bm_coeffs = berlekamp_massey_jit(fib_terms, np.int64(MOD))
assert len(bm_coeffs) == 2, f"BM should find order 2, got {len(bm_coeffs)}"
# Verify: a[n] = c[0]*a[n-1] + c[1]*a[n-2] which is [1, 1] for Fibonacci
assert list(bm_coeffs) == [1, 1], f"BM coefficients wrong: {list(bm_coeffs)}"
print("  ✓ berlekamp_massey_jit")

# binom_fast correctness
assert int(binom_fast(np.int64(10), np.int64(3), np.int64(MOD),
                      _FACT, _INV_FACT)) == 120, "binom_fast(10,3) wrong"
assert int(binom_fast(np.int64(0), np.int64(0), np.int64(MOD),
                      _FACT, _INV_FACT)) == 1, "binom_fast(0,0) wrong"
assert int(binom_fast(np.int64(5), np.int64(6), np.int64(MOD),
                      _FACT, _INV_FACT)) == 0, "binom_fast(5,6) wrong"
print("  ✓ binom_fast")

# catalan_jit correctness
assert int(catalan_jit(np.int64(5), np.int64(MOD), _FACT, _INV_FACT)) == 42, \
    "catalan(5) wrong"
assert int(catalan_jit(np.int64(0), np.int64(MOD), _FACT, _INV_FACT)) == 1, \
    "catalan(0) wrong"
print("  ✓ catalan_jit")

# stirling2_batch correctness
s2 = stirling2_batch(np.array([10], dtype=np.int64), np.int64(3),
                     np.int64(MOD), _FACT, _INV_FACT)
assert int(s2[0]) == 9330, f"stirling2(10,3) wrong: {int(s2[0])}"
print("  ✓ stirling2_batch")

# derangements_jit correctness
assert int(derangements_jit(np.int64(4), np.int64(MOD), _FACT, _INV_FACT)) == 9, \
    "D(4) wrong"
assert int(derangements_jit(np.int64(0), np.int64(MOD), _FACT, _INV_FACT)) == 1, \
    "D(0) wrong"
print("  ✓ derangements_jit")

# partition_jit correctness
assert int(partition_jit(np.int64(5), np.int64(MOD))) == 7, "p(5) wrong"
assert int(partition_jit(np.int64(10), np.int64(MOD))) == 42, "p(10) wrong"
print("  ✓ partition_jit")

# bell_jit correctness
assert int(bell_jit(np.int64(5), np.int64(MOD))) == 52, \
    f"B(5) wrong: {int(bell_jit(np.int64(5), np.int64(MOD)))}"
print("  ✓ bell_jit")

# linear_sieve_all correctness
phi_arr, mu_arr, tau_arr, sigma1_arr, _ = linear_sieve_all(np.int64(20))
assert int(phi_arr[6]) == 2,   f"phi(6) wrong: {int(phi_arr[6])}"
assert int(phi_arr[12]) == 4,  f"phi(12) wrong: {int(phi_arr[12])}"
assert int(mu_arr[6]) == 1,    f"mu(6) wrong: {int(mu_arr[6])}"
assert int(mu_arr[4]) == 0,    f"mu(4) wrong: {int(mu_arr[4])}"
assert int(tau_arr[12]) == 6,  f"tau(12) wrong: {int(tau_arr[12])}"
assert int(sigma1_arr[6]) == 12, f"sigma1(6) wrong: {int(sigma1_arr[6])}"
print("  ✓ linear_sieve_all (phi, mu, tau, sigma1)")

# crt_jit correctness
x_crt, lcm_crt = crt_jit(
    np.array([2, 3, 2], dtype=np.int64),
    np.array([3, 5, 7], dtype=np.int64),
)
assert (int(x_crt), int(lcm_crt)) == (23, 105), \
    f"CRT wrong: got ({int(x_crt)}, {int(lcm_crt)})"
print("  ✓ crt_jit")

# discrete_log_bsgs correctness
x_dlog = discrete_log_bsgs(np.int64(2), np.int64(8), np.int64(11))
assert int(x_dlog) == 3, f"discrete_log_bsgs(2,8,11) wrong: {int(x_dlog)}"
print("  ✓ discrete_log_bsgs")

# rational_gf_coefficient correctness (Fibonacci GF: x/(1-x-x^2))
# [x^n] P/Q where Q = 1 - x - x^2, P = x (numerator has only x term)
# i.e. Q = [1, -1, -1], P = [0, 1]
rg = rational_gf_coefficient(
    np.array([0, 1], dtype=np.int64),    # P = x
    np.array([1, -1, -1], dtype=np.int64),  # Q = 1 - x - x^2
    10, MOD,
)
assert rg == 55, f"rational_gf_coefficient Fibonacci n=10 wrong: {rg}"
print("  ✓ rational_gf_coefficient")

# shoelace_exact correctness (unit square)
xs_sq = np.array([0, 1, 1, 0], dtype=np.int64)
ys_sq = np.array([0, 0, 1, 1], dtype=np.int64)
area2 = int(shoelace_exact(xs_sq, ys_sq))
assert area2 == 2, f"shoelace_exact unit square wrong: {area2}"
print("  ✓ shoelace_exact (unit square)")

# picks_theorem correctness — triangle with 2*area=4, B=4 boundary points → I=1
# E.g., triangle (0,0),(2,1),(0,2): 2A=4, B=gcd(2,1)+gcd(2,1)+gcd(0,2)=1+1+2=4
I = int(picks_theorem(np.int64(4), np.int64(4)))
assert I == 1, f"picks_theorem(4, 4) wrong: {I}"
print("  ✓ picks_theorem")

# ─────────────────────────────────────────────────────────────────────────────
# SPEEDUP / TIMING BENCHMARKS
# ─────────────────────────────────────────────────────────────────────────────

print("\n── Speedup benchmarks ──")

# ── binom_fast vs Python (large k to show true table-lookup advantage) ───────
def py_binom(n, k, mod=MOD):
    """Pure Python modular binomial — O(k) loop without precomputed table."""
    if k < 0 or k > n:
        return 0
    if k > n - k:
        k = n - k
    num = 1
    for i in range(k):
        num = num * ((n - i) % mod) % mod
    den = 1
    for i in range(1, k + 1):
        den = den * i % mod
    return num * pow(den, mod - 2, mod) % mod

# Use large k to demonstrate O(1) table vs O(k) Python
_bench_n, _bench_k = 10**6, 500
t_py   = bench(py_binom, _bench_n, _bench_k, n_runs=200)
t_jit  = bench(binom_fast,
               np.int64(_bench_n), np.int64(_bench_k), np.int64(MOD),
               _FACT, _INV_FACT,
               n_runs=10000)
speedup = t_py / t_jit
print(f"  binom_fast(10^6, {_bench_k}): Python={t_py*1e6:.1f}μs  "
      f"JIT={t_jit*1e6:.2f}μs  speedup={speedup:.0f}×")
assert speedup > 50, f"binom_fast speedup too low: {speedup:.1f}×"

# ── partition_jit vs Python ───────────────────────────────────────────────────
def py_partition(n):
    p = [0] * (n + 1); p[0] = 1
    for k in range(1, n + 1):
        i = 1
        while True:
            g1 = i * (3 * i - 1) // 2
            g2 = i * (3 * i + 1) // 2
            if g1 > k:
                break
            sign = 1 if i % 2 == 1 else -1
            p[k] += sign * p[k - g1]
            if g2 <= k:
                p[k] += sign * p[k - g2]
            i += 1
    return p[n]

t_py  = bench(py_partition, 5000, n_runs=3, warmup=1)
t_jit = bench(partition_jit, np.int64(5000), np.int64(MOD), n_runs=5, warmup=2)
speedup = t_py / t_jit
print(f"  partition_jit(5000): Python={t_py*1e3:.1f}ms  "
      f"JIT={t_jit*1e3:.1f}ms  speedup={speedup:.0f}×")
assert speedup > 20, f"partition_jit speedup too low: {speedup:.1f}×"

# ── linear_sieve_all vs Python phi sieve ─────────────────────────────────────
# Use N=5*10^5 for a more pronounced speedup demonstration
_sieve_N = 500_000

def py_phi_sieve(N):
    phi = list(range(N + 1))
    for i in range(2, N + 1):
        if phi[i] == i:  # prime
            for j in range(i, N + 1, i):
                phi[j] -= phi[j] // i
    return phi

t_py  = bench(py_phi_sieve, _sieve_N, n_runs=3, warmup=1)
t_jit = bench(linear_sieve_all, np.int64(_sieve_N), n_runs=5, warmup=3)
speedup = t_py / t_jit
print(f"  linear_sieve_all(N={_sieve_N//1000}k): Python={t_py*1e3:.1f}ms  "
      f"JIT={t_jit*1e3:.1f}ms  speedup={speedup:.0f}×")
assert speedup > 10, f"linear_sieve_all speedup too low: {speedup:.1f}×"

# ── kitamasa_flint timing (k=2, n=10^18) ─────────────────────────────────────
t_fib_large = bench(
    kitamasa_flint,
    np.array([1, 1], dtype=np.int64),
    np.array([0, 1], dtype=np.int64),
    10**18, MOD,
    n_runs=50, warmup=5,
)
print(f"  kitamasa_flint (k=2, n=10^18): {t_fib_large*1e3:.2f}ms")
assert t_fib_large < 0.05, f"kitamasa_flint too slow: {t_fib_large*1e3:.2f}ms"

# ── berlekamp_massey_jit timing ───────────────────────────────────────────────
t_bm = bench(
    berlekamp_massey_jit, fib_terms, np.int64(MOD),
    n_runs=1000, warmup=10,
)
print(f"  berlekamp_massey_jit (k=2, 12 terms): {t_bm*1e3:.4f}ms")
assert t_bm < 0.001, f"berlekamp_massey_jit too slow: {t_bm*1e3:.4f}ms"

# ── crt_jit timing ────────────────────────────────────────────────────────────
t_crt = bench(
    crt_jit,
    np.array([2, 3, 2], dtype=np.int64),
    np.array([3, 5, 7], dtype=np.int64),
    n_runs=100000, warmup=1000,
)
print(f"  crt_jit (3 moduli): {t_crt*1e6:.3f}μs")
assert t_crt < 10e-6, f"crt_jit too slow: {t_crt*1e6:.3f}μs"

# ── stirling2_batch timing ────────────────────────────────────────────────────
t_s2 = bench(
    stirling2_batch,
    np.array([50, 100, 200], dtype=np.int64),
    np.int64(50), np.int64(MOD), _FACT, _INV_FACT,
    n_runs=20, warmup=5,
)
print(f"  stirling2_batch (3 queries, k=50): {t_s2*1e3:.2f}ms")
assert t_s2 < 0.05, f"stirling2_batch too slow: {t_s2*1e3:.2f}ms"

# ── rational_gf_coefficient Fibonacci n=10^15 ────────────────────────────────
t_rgf = bench(
    rational_gf_coefficient,
    np.array([0, 1], dtype=np.int64),
    np.array([1, -1, -1], dtype=np.int64),
    10**15, MOD,
    n_runs=20, warmup=3,
)
print(f"  rational_gf_coefficient Fibonacci n=10^15: {t_rgf*1e3:.2f}ms")
assert t_rgf < 0.05, f"rational_gf_coefficient too slow: {t_rgf*1e3:.2f}ms"

# ── shoelace_exact N=10^4 vertices ────────────────────────────────────────────
N_poly = 10_000
theta = np.linspace(0, 2 * np.pi, N_poly, endpoint=False)
xs_circle = (np.round(1000 * np.cos(theta))).astype(np.int64)
ys_circle = (np.round(1000 * np.sin(theta))).astype(np.int64)

t_shoelace = bench(shoelace_exact, xs_circle, ys_circle, n_runs=200, warmup=10)
print(f"  shoelace_exact (N={N_poly} vertices): {t_shoelace*1e3:.3f}ms")
assert t_shoelace < 0.005, f"shoelace_exact too slow: {t_shoelace*1e3:.3f}ms"

# ─────────────────────────────────────────────────────────────────────────────
print("\n✅ All v4 benchmarks passed. CTRL-MATH v4 solver stack verified.")
print("=" * 70)
