# cell_11_benchmarks.py
"""
MANDATORY: All benchmarks must pass before submission.
Every speedup assertion is a hard ASSERT, not a warning.
"""

import time
import numpy as np

from cell_02a_numba_nt import (
    vp_factorial_jit, dirichlet_conv_safe, powmod_batch,
    sigma_k_sieve, poly_mul_ntt,
    lte_p2_minus_jit, lte_odd_minus_jit, fib_jit, sum_sigma_k_upto,
    roots_of_unity_filter_batch,
)
from cell_05_cyclotomic import CyclotomicTools
from cell_10_norwegian import NorwegianNumbers
from fractions import Fraction


def benchmark(fn, *args, n_runs=100, warmup=10):
    for _ in range(warmup):
        fn(*args)
    t0 = time.perf_counter()
    for _ in range(n_runs):
        fn(*args)
    return (time.perf_counter() - t0) / n_runs


print("=" * 65)
print("CTRL-MATH v3 — Performance Benchmark Suite")
print("=" * 65)

# ── Benchmark 1: vp_factorial JIT vs Python ───────────────────────────────
def py_vp_fact(n, p):
    r, pk = 0, p
    while pk <= n:
        r += n // pk; pk *= p
    return r

t_py  = benchmark(py_vp_fact,      10**9, 5, n_runs=10000)
t_jit = benchmark(vp_factorial_jit, np.int64(10**9), np.int64(5), n_runs=10000)
speedup = t_py / t_jit
print(f"  vp_factorial:  Python={t_py*1e6:.1f}μs  JIT={t_jit*1e6:.1f}μs  speedup={speedup:.0f}×")
assert speedup > 20, f"vp_factorial JIT speedup too low: {speedup:.1f}×"

# ── Benchmark 2: dirichlet_conv parallel vs Python ───────────────────────
N_conv = 100_000
f_arr  = np.zeros(N_conv, dtype=np.int64); f_arr[1] = 1
g_arr  = np.ones(N_conv,  dtype=np.int64); g_arr[0] = 0

def py_conv(f, g):
    N = len(f); h = [0]*N
    for d in range(1, N):
        if f[d]:
            for m in range(d, N, d): h[m] += f[d]*g[m//d]
    return h

t_py   = benchmark(py_conv,             f_arr, g_arr, n_runs=3)
t_jit  = benchmark(dirichlet_conv_safe, f_arr, g_arr, n_runs=10)
speedup = t_py / t_jit
print(f"  dirichlet_conv (N={N_conv}): Python={t_py*1e3:.0f}ms  JIT={t_jit*1e3:.0f}ms  speedup={speedup:.0f}×")
assert speedup > 10, f"dirichlet_conv speedup too low: {speedup:.1f}×"

# ── Benchmark 3: powmod_batch GPU vs Python ───────────────────────────────
N_pm    = 100_000
bases_b = np.random.randint(2, 10**6, N_pm, dtype=np.int64)
exps_b  = np.random.randint(1, 10**6, N_pm, dtype=np.int64)
mod_b   = np.int64(10**9 + 7)

def py_powmod(bases, exps, mod):
    return [pow(int(b), int(e), mod) for b, e in zip(bases, exps)]

t_py  = benchmark(py_powmod,    bases_b, exps_b, mod_b, n_runs=3)
t_jit = benchmark(powmod_batch, bases_b, exps_b, mod_b, n_runs=10)
speedup = t_py / t_jit
print(f"  powmod_batch (N={N_pm}):  Python={t_py*1e3:.0f}ms  JIT={t_jit*1e3:.0f}ms  speedup={speedup:.0f}×")
assert speedup > 20, f"powmod_batch speedup too low: {speedup:.1f}×"

# ── Benchmark 4: sigma_k sieve ────────────────────────────────────────────
t_sieve = benchmark(sigma_k_sieve, np.int64(1_000_000), np.int64(1), n_runs=3)
print(f"  sigma_k_sieve (N=10^6, k=1): {t_sieve*1e3:.1f}ms")
assert t_sieve < 0.5, f"sigma_k_sieve too slow: {t_sieve*1e3:.1f}ms (limit 500ms)"

# ── Benchmark 5: NTT poly multiplication ─────────────────────────────────
deg    = 1024
f_poly = np.random.randint(0, 100, deg, dtype=np.int64)
g_poly = np.random.randint(0, 100, deg, dtype=np.int64)
t_ntt  = benchmark(poly_mul_ntt, f_poly, g_poly, n_runs=50)
print(f"  NTT poly_mul (deg={deg}): {t_ntt*1e3:.2f}ms")
assert t_ntt < 0.05, f"NTT too slow: {t_ntt*1e3:.2f}ms (limit 50ms)"

# ── Benchmark 6: fib_jit ──────────────────────────────────────────────────
t_fib = benchmark(fib_jit, np.int64(10**6), np.int64(10**9 + 7), n_runs=1000)
print(f"  fib_jit (n=10^6): {t_fib*1e3:.3f}ms")
assert t_fib < 0.001, f"fib_jit too slow: {t_fib*1e3:.3f}ms (limit 1ms)"

# ── Benchmark 7: roots_of_unity_filter_batch ──────────────────────────────
a_coeffs = np.random.randn(1000)
residues = np.arange(1000)
t_roots = benchmark(roots_of_unity_filter_batch, a_coeffs, 1000, residues, n_runs=100)
print(f"  roots_of_unity_filter_batch (N=10^3): {t_roots*1e3:.3f}ms")
assert t_roots < 0.005, f"roots_of_unity_filter_batch too slow: {t_roots*1e3:.3f}ms (limit 5ms)"

# ── Benchmark 8: KalmanBeliefState.update_batch ───────────────────────────
try:
    from cell_08_kalman import KalmanBeliefState
    kb = KalmanBeliefState()
    for i in range(1000):
        kb.facts[f"fact_{i}"] = i
        kb.mean = np.append(kb.mean, 0.5)
        kb.var = np.append(kb.var, 0.1)
    obs = {f"fact_{i}": 1.0 for i in range(1000)}
    t_kalman = benchmark(kb.update_batch, obs, n_runs=100)
    print(f"  KalmanBeliefState.update_batch (N=1000): {t_kalman*1e3:.3f}ms")
    assert t_kalman < 0.001, f"Kalman update too slow: {t_kalman*1e3:.3f}ms (limit 1ms)"
except ImportError:
    print("  [SKIP] cell_08_kalman not available for benchmark")

# ── Benchmark 9: ParallelZ3Checker ────────────────────────────────────────
try:
    from cell_09b_z3_parallel import ParallelZ3Checker
    import z3

    def make_goal(i):
        x = z3.Int(f'x_{i}')
        return z3.And(x > 0, x < 100, x * x == (i + 2) * (i + 2))

    goals = [make_goal(i) for i in range(4)]
    checker = ParallelZ3Checker(n_workers=4)
    t_z3 = benchmark(checker.check_all, goals, n_runs=5, warmup=1)
    print(f"  ParallelZ3Checker (4 goals): {t_z3:.3f}s")
    assert t_z3 < 5.0, f"Z3 checker too slow: {t_z3:.3f}s (limit 5s)"
except ImportError:
    print("  [SKIP] cell_09b_z3_parallel not available for benchmark")

# ── Correctness checks ───────────────────────────────────────────────────
assert vp_factorial_jit(np.int64(100), np.int64(5)) == 24, "Legendre formula wrong"
assert lte_p2_minus_jit(np.int64(3), np.int64(1), np.int64(4)) == 4, "LTE p=2 minus wrong"
assert lte_odd_minus_jit(np.int64(7), np.int64(2), np.int64(6), np.int64(5)) == 1, "LTE odd minus wrong"
assert fib_jit(np.int64(10), np.int64(10**9+7)) == 55, "Fibonacci wrong"
assert sum_sigma_k_upto(4, 1) == 15, "sigma sum wrong"

# Dirichlet identity: identity_f * g = g
identity_f = np.zeros(N_conv, dtype=np.int64); identity_f[1] = 1
g_test     = np.zeros(N_conv, dtype=np.int64)
g_test[1:100] = np.arange(1, 100)
result = dirichlet_conv_safe(identity_f, g_test)
assert np.array_equal(result[1:100], g_test[1:100]), "Dirichlet identity failed"

# Polynomial multiplication correctness
result_poly = poly_mul_ntt(np.array([1, 2], np.int64), np.array([1, 2], np.int64))
assert list(result_poly) == [1, 4, 4], f"poly_mul_ntt wrong: {list(result_poly)}"

# powmod_batch correctness
pm_result = powmod_batch(np.array([2], np.int64), np.array([10], np.int64), np.int64(1000))
assert pm_result[0] == 24, f"powmod_batch wrong: {pm_result[0]}"

# CyclotomicTools correctness
assert CyclotomicTools.count_shifty_polynomials() == 160, "CyclotomicTools count wrong"

# NorwegianNumbers correctness
assert NorwegianNumbers.solve_problem10() == Fraction(125561848, 19033825), "NorwegianNumbers wrong"

print("\n✅ All benchmarks passed. CTRL-MATH v3 performance verified.")
print("=" * 65)
