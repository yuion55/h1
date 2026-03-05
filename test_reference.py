# test_reference.py
"""
CTRL-MATH AIMO3 — Reference Test Validation

Loads reference.csv, runs the orchestrator on each problem,
and reports accuracy against ground truth answers.

Usage:
    python test_reference.py [--csv reference.csv] [--limit N]
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional


def _normalize_answer(val) -> Optional[int]:
    """Normalize a predicted or expected answer value to int for comparison."""
    try:
        return int(float(str(val).strip())) % 100_000
    except (ValueError, TypeError):
        return None


def load_reference_csv(path: str) -> List[Dict]:
    """Load reference.csv — expects columns: id, problem, answer."""
    problems = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            problems.append({
                "id":      row.get("id", ""),
                "problem": row.get("problem", row.get("question", "")),
                "answer":  str(row.get("answer", "")),
            })
    return problems


def run_reference_validation(
    csv_path: str = "reference.csv",
    limit: Optional[int] = None,
) -> Dict:
    """
    Run the CTRL-MATH orchestrator on reference problems and report accuracy.

    Args:
        csv_path: Path to reference CSV file with columns id, problem, answer.
        limit:    Maximum number of problems to evaluate (None = all).

    Returns:
        Dict with accuracy, correct, total, and per-problem results.
    """
    if not os.path.exists(csv_path):
        print(f"[ERROR] Reference CSV not found: {csv_path}")
        print("  Download with: kaggle competitions download -c ai-mathematical-olympiad-progress-prize-3")
        return {"accuracy": 0.0, "correct": 0, "total": 0, "results": []}

    problems = load_reference_csv(csv_path)
    if limit is not None:
        problems = problems[:limit]

    print(f"[Reference Test] Loaded {len(problems)} problems from {csv_path}")

    # ── Load CTRL-MATH modules ────────────────────────────────────────────────
    print("[Reference Test] Initializing CTRL-MATH modules...")
    try:
        from cell_15_orchestrator_v5 import SolveOrchestrator
    except ImportError as e:
        print(f"[ERROR] Failed to import SolveOrchestrator: {e}")
        print("  Make sure you are running from the h1-main directory.")
        return {"accuracy": 0.0, "correct": 0, "total": 0, "results": []}

    # Try to load LLM executor
    llm = None
    try:
        from cell_07_llm_executor_v5 import LLMExecutorV5
        PRIMARY_MODEL = os.environ.get("CTRLMATH_MODEL", "Qwen/Qwen2.5-Math-14B-Instruct")
        LORA_ADAPTER = os.environ.get("CTRLMATH_LORA", "/kaggle/working/ctrlmath_aimo3_lora")
        if not os.path.isdir(LORA_ADAPTER):
            LORA_ADAPTER = None
        llm = LLMExecutorV5(
            primary_model=PRIMARY_MODEL,
            lora_adapter=LORA_ADAPTER,
            load_in_4bit=True,
        )
        print(f"[Reference Test] LLM loaded: {PRIMARY_MODEL}")
    except Exception as e:
        print(f"[WARN] LLM not available: {e} — running without LLM inference.")

    # Try to load Z3 checker
    z3_checker = None
    try:
        from cell_09b_z3_parallel import ParallelZ3Checker
        if ParallelZ3Checker is not None:
            z3_checker = ParallelZ3Checker(n_workers=4)
    except Exception:
        pass

    orchestrator = SolveOrchestrator(
        llm=llm,
        z3_checker=z3_checker,
        total_seconds=len(problems) * 180.0,  # 3 min per problem budget
        n_problems=len(problems),
        mcts_sims=64,
    )

    # ── Evaluate ──────────────────────────────────────────────────────────────
    correct = 0
    results = []
    t_start = time.time()

    print(f"\n{'─'*65}")
    print(f"{'#':>4}  {'ID':<12}  {'Expected':>8}  {'Predicted':>9}  {'OK':>4}  {'Time':>6}")
    print(f"{'─'*65}")

    for i, prob in enumerate(problems):
        pid     = prob["id"]
        problem = prob["problem"]
        expected_raw = prob["answer"]
        expected = _normalize_answer(expected_raw)

        t0 = time.time()
        try:
            predicted_raw = orchestrator.solve_problem(pid, problem)
            predicted = _normalize_answer(predicted_raw)
        except Exception as e:
            predicted = 0
            print(f"  [WARN] solve_problem raised: {e}")
        elapsed = time.time() - t0

        is_correct = (predicted is not None and expected is not None
                      and predicted == expected)
        if is_correct:
            correct += 1

        status = "✅" if is_correct else "❌"
        print(f"{i+1:>4}  {pid:<12}  {expected_raw:>8}  {str(predicted):>9}  {status:>4}  {elapsed:>5.1f}s")

        results.append({
            "id":        pid,
            "expected":  expected_raw,
            "predicted": predicted,
            "correct":   is_correct,
            "elapsed_s": elapsed,
        })

    total_elapsed = time.time() - t_start
    accuracy = correct / max(len(problems), 1)

    print(f"{'─'*65}")
    print(f"\n[Reference Test] Results:")
    print(f"  Correct:  {correct}/{len(problems)}")
    print(f"  Accuracy: {accuracy:.1%}")
    print(f"  Total:    {total_elapsed/60:.1f} min  (avg {total_elapsed/max(len(problems),1):.1f}s/problem)")

    return {
        "accuracy": accuracy,
        "correct":  correct,
        "total":    len(problems),
        "results":  results,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CTRL-MATH AIMO3 reference validation")
    parser.add_argument("--csv",   default="reference.csv", help="Path to reference CSV")
    parser.add_argument("--limit", type=int, default=None,  help="Max problems to evaluate")
    args = parser.parse_args()

    # Add parent directory to path so cell_*.py modules are importable
    sys.path.insert(0, str(Path(__file__).parent))

    result = run_reference_validation(csv_path=args.csv, limit=args.limit)
    sys.exit(0 if result["total"] > 0 else 1)
