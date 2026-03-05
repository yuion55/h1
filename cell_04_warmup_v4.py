# cell_04_warmup_v4.py
"""
CTRL-MATH v4 — Pre-compile all @njit functions
(after cell_04_engine_v4.py)

warmup_v4_solvers() triggers JIT compilation for every @njit function
in the v4 solver stack (cells 04b, 04c, 04d, 04f) with small dummy inputs.
Called at module level (executed on import).
"""

import numpy as np

# ── Import all JIT functions ──────────────────────────────────────────────────
try:
    from cell_04b_linear_recurrence import berlekamp_massey_jit, kitamasa_flint
    _HAS_LR = True
except ImportError:
    _HAS_LR = False

try:
    from cell_04c_combinatorics import (
        ensure_tables, _build_fact_table, binom_fast, catalan_jit,
        stirling2_batch, derangements_jit, partition_jit,
        partition_batch_jit, bell_jit, inclusion_exclusion_jit,
        _FACT, _INV_FACT, COMB_MOD,
    )
    _HAS_COMB = True
except ImportError:
    _HAS_COMB = False
    COMB_MOD = 998_244_353

try:
    from cell_04d_number_theory import (
        linear_sieve_all, sum_phi_upto, sum_mult_func_upto,
        crt_jit, discrete_log_bsgs,
    )
    _HAS_NT = True
except ImportError:
    _HAS_NT = False

try:
    from cell_04f_geometry import (
        shoelace_exact, picks_theorem, boundary_lattice_points,
        convex_hull_area,
    )
    _HAS_GEOM = True
except ImportError:
    _HAS_GEOM = False

MOD = 998_244_353


def warmup_v4_solvers() -> None:
    """
    Pre-compile all @njit functions from the v4 solver stack.
    Prints progress and "✅ done." upon completion.
    """
    print("Warming up CTRL-MATH v4 JIT solvers...", end="", flush=True)

    # ── cell_04b: Linear Recurrence ───────────────────────────────────────────
    if _HAS_LR:
        fib_terms = np.array([0, 1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89],
                             dtype=np.int64)
        berlekamp_massey_jit(fib_terms, np.int64(MOD))
        kitamasa_flint(
            np.array([1, 1], dtype=np.int64),
            np.array([0, 1], dtype=np.int64),
            10, MOD,
        )

    # ── cell_04c: Combinatorics ───────────────────────────────────────────────
    if _HAS_COMB:
        ensure_tables(MOD)
        binom_fast(np.int64(10), np.int64(3), np.int64(MOD), _FACT, _INV_FACT)
        catalan_jit(np.int64(5), np.int64(MOD), _FACT, _INV_FACT)
        stirling2_batch(
            np.array([5], dtype=np.int64),
            np.int64(3), np.int64(MOD), _FACT, _INV_FACT,
        )
        derangements_jit(np.int64(4), np.int64(MOD), _FACT, _INV_FACT)
        partition_jit(np.int64(5), np.int64(MOD))
        partition_batch_jit(np.array([3, 5], dtype=np.int64), np.int64(MOD))
        bell_jit(np.int64(4), np.int64(MOD))
        set_sizes = np.array([3, 4], dtype=np.int64)
        inter = np.array([0, 3, 4, 1], dtype=np.int64)  # mask 0,1,2,3
        inclusion_exclusion_jit(set_sizes, inter, np.int64(MOD))

    # ── cell_04d: Number Theory ───────────────────────────────────────────────
    if _HAS_NT:
        linear_sieve_all(np.int64(100))
        sum_phi_upto(np.int64(10), np.int64(MOD))
        sum_mult_func_upto(np.int64(10), np.int64(0), np.int64(0), np.int64(MOD))
        crt_jit(
            np.array([2, 3], dtype=np.int64),
            np.array([3, 5], dtype=np.int64),
        )
        discrete_log_bsgs(np.int64(2), np.int64(4), np.int64(7))

    # ── cell_04f: Geometry ────────────────────────────────────────────────────
    if _HAS_GEOM:
        xs = np.array([0, 1, 1, 0], dtype=np.int64)
        ys = np.array([0, 0, 1, 1], dtype=np.int64)
        shoelace_exact(xs, ys)
        picks_theorem(np.int64(2), np.int64(4))
        boundary_lattice_points(xs, ys)
        convex_hull_area(xs, ys)

    print(" ✅ done.")


# ── Execute at import time ────────────────────────────────────────────────────
warmup_v4_solvers()
