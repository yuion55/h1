# cell_04f_geometry.py
"""
CTRL-MATH v4 — Coordinate Geometry Solver
(after cell_04e_gf_solver.py)

Implements Shoelace formula, Pick's theorem, boundary lattice points,
and convex hull area — all with @njit(cache=True) for JIT compilation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict

import numpy as np
from numba import njit

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


# ── Shoelace formula (exact int64) ────────────────────────────────────────────
@njit(cache=True)
def shoelace_exact(x: np.ndarray, y: np.ndarray) -> int:
    """
    Compute twice the signed area of a polygon using the Shoelace formula.
    Returns |sum_i (x_i * y_{i+1} - x_{i+1} * y_i)| (exact int64).
    O(N).
    """
    n = len(x)
    s = np.int64(0)
    for i in range(n):
        j = (i + 1) % n
        s += x[i] * y[j] - x[j] * y[i]
    if s < 0:
        s = -s
    return s


# ── Pick's theorem ────────────────────────────────────────────────────────────
@njit(cache=True)
def picks_theorem(area_times_2: int, boundary_points: int) -> int:
    """
    Compute the number of interior lattice points via Pick's theorem.
    I = (2A - B + 2) / 2 where area_times_2 = 2*A, B = boundary_points.
    Equivalently: I = A - B/2 + 1.
    area_times_2 = 2 * (polygon area), as returned by shoelace_exact.
    """
    return (area_times_2 - boundary_points + 2) // 2


# ── Boundary lattice points ───────────────────────────────────────────────────
@njit(cache=True)
def boundary_lattice_points(x: np.ndarray, y: np.ndarray) -> int:
    """
    Count lattice points on the boundary of a polygon.
    For each edge from (x_i, y_i) to (x_{i+1}, y_{i+1}),
    the number of boundary points is gcd(|dx|, |dy|).
    """
    n = len(x)
    total = np.int64(0)
    for i in range(n):
        j = (i + 1) % n
        dx = abs(x[j] - x[i])
        dy = abs(y[j] - y[i])
        # Inline Euclidean GCD
        a, b = np.int64(dx), np.int64(dy)
        while b:
            a, b = b, a % b
        total += a
    return total


# ── Convex hull area ──────────────────────────────────────────────────────────
@njit(cache=True)
def convex_hull_area(x: np.ndarray, y: np.ndarray) -> int:
    """
    Compute 2 * (area of convex hull of points) using Graham scan + Shoelace.
    Uses insertion sort for polar angle since argsort is not available in @njit.
    Returns 2 * area (integer).
    """
    n = len(x)
    if n < 3:
        return np.int64(0)

    # Find pivot: lowest y, then leftmost x
    pivot = 0
    for i in range(1, n):
        if y[i] < y[pivot] or (y[i] == y[pivot] and x[i] < x[pivot]):
            pivot = i

    # Copy arrays for sorting
    xs = x.copy()
    ys = y.copy()
    # Swap pivot to position 0
    xs[0], xs[pivot] = xs[pivot], xs[0]
    ys[0], ys[pivot] = ys[pivot], ys[0]
    px, py = xs[0], ys[0]

    # Insertion sort by polar angle relative to pivot (CCW)
    for i in range(1, n):
        j = i
        while j > 1:
            # Compare polar angles of xs[j-1] and xs[j] relative to pivot
            # via the cross product: (a - pivot) × (b - pivot)
            ax, ay = xs[j - 1] - px, ys[j - 1] - py
            bx, vy = xs[j] - px, ys[j] - py
            cross = ax * vy - ay * bx
            if cross < 0:  # xs[j] is CCW of xs[j-1], swap
                xs[j - 1], xs[j] = xs[j], xs[j - 1]
                ys[j - 1], ys[j] = ys[j], ys[j - 1]
                j -= 1
            elif cross == 0:
                # Collinear: keep closer point first
                d1 = ax * ax + ay * ay
                d2 = bx * bx + vy * vy
                if d1 > d2:
                    xs[j - 1], xs[j] = xs[j], xs[j - 1]
                    ys[j - 1], ys[j] = ys[j], ys[j - 1]
                j -= 1  # break inner while loop after handling collinear case
                break
            else:
                break

    # Graham scan: build convex hull stack
    hull_x = np.zeros(n, dtype=np.int64)
    hull_y = np.zeros(n, dtype=np.int64)
    hull_size = np.int64(0)

    for i in range(n):
        # Pop while the last 3 points make a non-left turn
        while hull_size >= 2:
            ax = hull_x[hull_size - 1] - hull_x[hull_size - 2]
            ay = hull_y[hull_size - 1] - hull_y[hull_size - 2]
            bx = xs[i] - hull_x[hull_size - 2]
            vy = ys[i] - hull_y[hull_size - 2]
            cross = ax * vy - ay * bx
            if cross <= 0:
                hull_size -= 1
            else:
                break
        hull_x[hull_size] = xs[i]
        hull_y[hull_size] = ys[i]
        hull_size += 1

    return shoelace_exact(hull_x[:hull_size], hull_y[:hull_size])


# ── GeometrySolver ────────────────────────────────────────────────────────────
class GeometrySolver:
    """Dispatches geometry problems to the appropriate JIT solver."""

    @staticmethod
    def solve(params: dict, mod: int) -> "TransformResult":
        """
        Solve a geometry problem.
        params["sub_type"]: polygon_area, lattice_interior, convex_hull_area.
        """
        sub = params.get("sub_type", "polygon_area")
        x = np.array(params["x"], dtype=np.int64)
        y = np.array(params["y"], dtype=np.int64)

        if sub == "polygon_area":
            area2 = int(shoelace_exact(x, y))
            answer = area2  # return 2*area (exact integer)
            return TransformResult(
                solved=True, answer=answer, reduced_state=None,
                certificate={"sub_type": "polygon_area", "area_times_2": area2},
                transform_name="geometry_v4",
            )

        elif sub == "lattice_interior":
            area2 = int(shoelace_exact(x, y))
            B = int(boundary_lattice_points(x, y))
            I = int(picks_theorem(np.int64(area2), np.int64(B)))
            return TransformResult(
                solved=True, answer=I, reduced_state=None,
                certificate={"sub_type": "lattice_interior",
                             "area_times_2": area2, "boundary": B, "interior": I},
                transform_name="geometry_v4",
            )

        elif sub == "convex_hull_area":
            area2 = int(convex_hull_area(x, y))
            return TransformResult(
                solved=True, answer=area2, reduced_state=None,
                certificate={"sub_type": "convex_hull_area", "area_times_2": area2},
                transform_name="geometry_v4",
            )

        return TransformResult(
            solved=False, answer=None, reduced_state=None,
            certificate={"error": f"unknown sub_type: {sub}"},
            transform_name="geometry_v4",
        )
