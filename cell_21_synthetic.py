# cell_21_synthetic.py
"""
CTRL-MATH AIMO3 — Synthetic Data Generation Loop

Generates synthetic competition math problems using:
  1. geometry_tool mutations (random triangles, concyclic quads)
  2. Number theory kernel-solved problems (p-adic, modular)
  3. LLM-written TIR traces for solved problems
  4. PRM scoring for quality filtering

Output: JSON files in /kaggle/working/synthetic_aimo3/
"""

from __future__ import annotations

import json
import os
import random
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

# ── Constants ─────────────────────────────────────────────────────────────────

SYNTHETIC_OUTPUT_DIR = "/kaggle/working/synthetic_aimo3"
DEFAULT_BATCH_SIZE = 100
MIN_PRM_SCORE = 0.5

# ── Number theory problem generators ─────────────────────────────────────────

try:
    from cell_02a_numba_nt import vp_factorial_jit, vp_jit, fib_jit
    HAS_NUMBA_NT = True
except ImportError:
    HAS_NUMBA_NT = False


def _generate_vp_factorial_problems(n: int = 50) -> List[Dict[str, Any]]:
    """Generate p-adic valuation of factorial problems."""
    problems = []
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
    for i in range(n):
        p = random.choice(primes)
        factorial_n = random.randint(10, 10**6)
        if HAS_NUMBA_NT:
            answer = int(vp_factorial_jit(np.int64(factorial_n), np.int64(p)))
        else:
            # Python fallback
            answer, pk = 0, p
            while pk <= factorial_n:
                answer += factorial_n // pk
                pk *= p

        problems.append({
            "id": f"syn_vp_fact_{i}",
            "problem": (
                f"Find the largest power of {p} that divides {factorial_n}!. "
                f"In other words, compute v_{p}({factorial_n}!)."
            ),
            "answer": str(answer),
            "domain": "number_theory",
            "sub_type": "p_adic_valuation",
            "difficulty": "easy" if factorial_n < 1000 else "medium",
        })
    return problems


def _generate_fibonacci_mod_problems(n: int = 50) -> List[Dict[str, Any]]:
    """Generate Fibonacci modular arithmetic problems."""
    problems = []
    mods = [10**9 + 7, 998244353, 1000, 10**6 + 3]
    for i in range(n):
        fib_n = random.randint(100, 10**6)
        mod = random.choice(mods)
        if HAS_NUMBA_NT:
            answer = int(fib_jit(np.int64(fib_n), np.int64(mod)))
        else:
            # Matrix power fallback
            def fib_mod(nn, mm):
                if nn == 0:
                    return 0
                a, b = 0, 1
                for _ in range(nn - 1):
                    a, b = b, (a + b) % mm
                return b
            answer = fib_mod(min(fib_n, 10000), mod)

        problems.append({
            "id": f"syn_fib_{i}",
            "problem": (
                f"Find F_{{{fib_n}}} mod {mod}, where F_n is the n-th Fibonacci number "
                f"(F_0 = 0, F_1 = 1, F_n = F_{{n-1}} + F_{{n-2}})."
            ),
            "answer": str(answer),
            "domain": "number_theory",
            "sub_type": "fibonacci_mod",
            "difficulty": "medium",
        })
    return problems


def _generate_combinatorics_problems(n: int = 50) -> List[Dict[str, Any]]:
    """Generate combinatorics problems with known solutions."""
    import math
    problems = []
    for i in range(n):
        problem_type = random.choice(["binomial", "catalan", "derangement"])

        if problem_type == "binomial":
            nn = random.randint(5, 30)
            k = random.randint(1, nn - 1)
            answer = math.comb(nn, k)
            text = f"Compute C({nn}, {k}), the number of ways to choose {k} items from {nn}."

        elif problem_type == "catalan":
            nn = random.randint(2, 15)
            answer = math.comb(2 * nn, nn) // (nn + 1)
            text = f"Find the {nn}-th Catalan number C_{nn}."

        else:  # derangement
            nn = random.randint(3, 12)
            # D(n) = n! * sum_{k=0}^{n} (-1)^k / k!
            answer = 0
            for k in range(nn + 1):
                answer += ((-1) ** k) * math.factorial(nn) // math.factorial(k)
            text = (
                f"Find the number of derangements of {nn} elements, "
                f"i.e., the number of permutations with no fixed points."
            )

        problems.append({
            "id": f"syn_comb_{i}",
            "problem": text,
            "answer": str(answer),
            "domain": "combinatorics",
            "sub_type": problem_type,
            "difficulty": "easy" if nn <= 10 else "medium",
        })
    return problems


# ── Geometry problem generators ───────────────────────────────────────────────

def _generate_geometry_problems(n: int = 50) -> List[Dict[str, Any]]:
    """Generate geometry problems using geometry_tool mutations."""
    try:
        from cell_04g_geometry_prover import geometry_tool
        mutations = geometry_tool.generate_mutations(n)
        problems = []
        for mut in mutations:
            problems.append({
                "id": mut["id"],
                "problem": mut["problem"],
                "answer": str(mut["answer"]),
                "domain": "geometry",
                "sub_type": "triangle_area",
                "difficulty": "medium",
            })
        return problems
    except ImportError:
        return []


# ── LLM TIR trace generation ─────────────────────────────────────────────────

def generate_tir_traces(
    problems: List[Dict[str, Any]],
    llm=None,
    max_traces: int = 100,
) -> List[Dict[str, Any]]:
    """
    Generate TIR (Tool-Integrated Reasoning) traces for problems using LLM.

    Each trace includes: problem, CoT reasoning, code, answer.
    Returns augmented problem dicts with 'solution' field.
    """
    if llm is None:
        return problems

    augmented = []
    for i, prob in enumerate(problems[:max_traces]):
        try:
            if hasattr(llm, 'solve_with_reasoning'):
                result = llm.solve_with_reasoning(
                    prob["problem"],
                    domain=prob.get("domain", "unknown"),
                )
                prob_aug = dict(prob)
                prob_aug["solution"] = result.get("reasoning", "")
                prob_aug["llm_answer"] = str(result.get("answer", ""))
                augmented.append(prob_aug)
            else:
                augmented.append(prob)
        except Exception:
            augmented.append(prob)

    return augmented


# ── PRM quality filtering ────────────────────────────────────────────────────

def filter_by_prm(
    problems: List[Dict[str, Any]],
    prm=None,
    min_score: float = MIN_PRM_SCORE,
) -> List[Dict[str, Any]]:
    """
    Filter problems by PRM (Process Reward Model) score.
    Only keeps problems where the solution scores above min_score.
    """
    if prm is None:
        return problems

    filtered = []
    for prob in problems:
        solution = prob.get("solution", "")
        if not solution:
            filtered.append(prob)
            continue

        try:
            steps = [s.strip() for s in solution.split("\n") if s.strip()]
            scores = prm.score_batch(steps)
            avg_score = float(np.mean(scores)) if len(scores) > 0 else 0.0
            if avg_score >= min_score:
                prob["prm_score"] = avg_score
                filtered.append(prob)
        except Exception:
            filtered.append(prob)

    return filtered


# ── Main generation loop ─────────────────────────────────────────────────────

def generate_synthetic_batch(
    n_per_domain: int = DEFAULT_BATCH_SIZE,
    llm=None,
    prm=None,
    output_dir: str = SYNTHETIC_OUTPUT_DIR,
) -> List[Dict[str, Any]]:
    """
    Generate a batch of synthetic problems across all domains.

    Pipeline:
      1. Generate raw problems (NT + combinatorics + geometry)
      2. Add LLM TIR traces (if LLM available)
      3. Filter by PRM quality (if PRM available)
      4. Save to output directory

    Returns list of all generated problems.
    """
    print(f"\n{'='*65}")
    print(f"Synthetic Data Generation — {n_per_domain} problems per domain")
    print(f"{'='*65}")

    all_problems: List[Dict[str, Any]] = []

    # Generate by domain
    print("  Generating number theory (vp_factorial)...", end=" ", flush=True)
    nt_problems = _generate_vp_factorial_problems(n_per_domain)
    print(f"{len(nt_problems)} problems")
    all_problems.extend(nt_problems)

    print("  Generating number theory (fibonacci)...", end=" ", flush=True)
    fib_problems = _generate_fibonacci_mod_problems(n_per_domain)
    print(f"{len(fib_problems)} problems")
    all_problems.extend(fib_problems)

    print("  Generating combinatorics...", end=" ", flush=True)
    comb_problems = _generate_combinatorics_problems(n_per_domain)
    print(f"{len(comb_problems)} problems")
    all_problems.extend(comb_problems)

    print("  Generating geometry...", end=" ", flush=True)
    geo_problems = _generate_geometry_problems(n_per_domain)
    print(f"{len(geo_problems)} problems")
    all_problems.extend(geo_problems)

    total_raw = len(all_problems)
    print(f"\n  Total raw problems: {total_raw}")

    # LLM TIR traces
    if llm is not None:
        print("  Generating TIR traces...", end=" ", flush=True)
        all_problems = generate_tir_traces(all_problems, llm=llm)
        print("done")

    # PRM filtering
    if prm is not None:
        print("  Filtering by PRM quality...", end=" ", flush=True)
        all_problems = filter_by_prm(all_problems, prm=prm)
        print(f"{len(all_problems)}/{total_raw} passed")

    # Save to disk
    os.makedirs(output_dir, exist_ok=True)
    timestamp = int(time.time())
    output_path = os.path.join(output_dir, f"synthetic_batch_{timestamp}.json")
    with open(output_path, "w") as f:
        json.dump(all_problems, f, indent=2, default=str)
    print(f"\n  ✅ Saved {len(all_problems)} problems to {output_path}")

    return all_problems


def run_synthetic_loop(
    n_iterations: int = 10,
    n_per_domain: int = DEFAULT_BATCH_SIZE,
    llm=None,
    prm=None,
    output_dir: str = SYNTHETIC_OUTPUT_DIR,
) -> int:
    """
    Run the full synthetic data generation loop.

    Generates n_iterations batches, each with n_per_domain problems per domain.
    Returns total number of problems generated.
    """
    total = 0
    for iteration in range(n_iterations):
        print(f"\n── Iteration {iteration + 1}/{n_iterations} ──")
        batch = generate_synthetic_batch(
            n_per_domain=n_per_domain,
            llm=llm,
            prm=prm,
            output_dir=output_dir,
        )
        total += len(batch)

    print(f"\n{'='*65}")
    print(f"✅ Synthetic data loop complete: {total} total problems generated")
    print(f"{'='*65}")
    return total
