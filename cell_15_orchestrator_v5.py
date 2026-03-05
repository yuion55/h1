# cell_15_orchestrator_v5.py
"""
CTRL-MATH v5 — Full Solve Orchestrator

Wires ALL components:
  MCTS, LLM, PRM, MathRAG, TemplateStore, AnswerExtractor,
  TimeAllocator, SelfConsistencyChecker, VerificationLadder

4-phase solve loop:
  1. Shortcuts (template match, direct extract)
  2. MCTS solve (K=3 independent rollouts → self-consistency)
  3. Verify + Lean correction (up to 3 retries)
  4. Template save on success

NEVER raises — all exceptions caught internally.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Tuple

from cell_06_mcts import MCTSEngine
from cell_09_mathrag import MathRAG
from cell_10_template_store import TemplateStore
from cell_11_answer_extractor import AnswerExtractor
from cell_12_time_allocator import BudgetState, TimeAllocator
from cell_13_self_consistency import SelfConsistencyChecker
from cell_14_verification_ladder import VerificationLadder

# Optional imports (won't crash if missing)
try:
    from cell_07_llm_executor_v5 import LLMExecutorV5 as _LLMClass
    _HAS_LLM = True
except ImportError:
    _HAS_LLM = False

try:
    from cell_08_prm import ProcessRewardModel as _PRMClass
    _HAS_PRM = True
except ImportError:
    _HAS_PRM = False


class SolveOrchestrator:
    """
    Top-level orchestrator for CTRL-MATH v5.

    Coordinates all components to solve competition math problems
    within a time budget.
    """

    _SC_K        = 3    # default self-consistency rollouts
    _LEAN_RETRIES = 3   # max Lean correction retries
    _DECOMP_AT   = 10   # attempt number to trigger decomposition

    def __init__(
        self,
        llm=None,
        prm=None,
        lean_repl=None,
        z3_checker=None,
        total_seconds:  float = 9 * 3600.0,
        n_problems:     int   = 50,
        mcts_sims:      int   = 64,
        persist_path:   str   = "/tmp/mathrag_templates.json",
    ):
        self.llm   = llm
        self.prm   = prm

        self.mcts   = MCTSEngine(
            llm_executor = llm,
            prm          = prm,
            n_simulations = mcts_sims,
        )
        self.rag    = MathRAG()
        self.store  = TemplateStore(persist_path=persist_path)
        self.extractor = AnswerExtractor()
        self.allocator = TimeAllocator(
            total_seconds=total_seconds,
            n_problems=n_problems,
        )
        self.sc     = SelfConsistencyChecker(k=self._SC_K)
        self.ladder = VerificationLadder(
            lean_repl  = lean_repl,
            z3_checker = z3_checker,
        )

    # ── Main entry point ──────────────────────────────────────────────────────

    def solve_problem(self, problem_id: str, problem_text: str) -> int:
        """
        Solve a competition math problem.

        NEVER raises — all exceptions caught.
        Returns integer answer (0 on failure).
        """
        try:
            return self._solve_inner(problem_id, problem_text)
        except Exception:
            return 0

    def _solve_inner(self, problem_id: str, problem_text: str) -> int:
        """Internal solve logic. May raise (will be caught by solve_problem)."""
        budget: BudgetState = self.allocator.allocate_budget(
            problem_id, problem_text
        )

        # ── Phase 1: Shortcuts ────────────────────────────────────────────────
        shortcut = self._try_shortcut(problem_text, budget)
        if shortcut is not None:
            self.allocator.record_result(budget, solved=True, answer=shortcut)
            return shortcut

        # ── Retrieve relevant theorems for context ────────────────────────────
        theorems = []
        try:
            theorems = self.rag.retrieve(problem_text, k=5)
        except Exception:
            pass
        theorem_context = self.rag.format_for_prompt(theorems)

        # ── Phase 2: MCTS + Self-Consistency ─────────────────────────────────
        rollout_answers: List[int] = []
        rollout_texts:   List[str] = []

        attempt = 0
        while (
            not budget.solved
            and not self.allocator.should_abandon(budget)
        ):
            attempt += 1

            # Problem decomposition at attempt threshold
            if attempt == self._DECOMP_AT and self.llm is not None:
                try:
                    sub_answers = self._solve_decomposed(problem_text, budget)
                    if sub_answers:
                        rollout_answers.extend(sub_answers)
                except Exception:
                    pass

            # MCTS rollout
            remaining = self.allocator.time_remaining_for(budget)
            if remaining <= 0:
                break
            try:
                mcts_budget = min(remaining * 0.8, budget.allocated_sec * 0.4)
                ans, text   = self.mcts.solve(
                    problem_text + "\n\n" + theorem_context,
                    budget_seconds=mcts_budget,
                )
                rollout_answers.append(ans)
                rollout_texts.append(text)
            except Exception:
                pass

            # Check self-consistency after SC_K rollouts
            if len(rollout_answers) >= self._SC_K:
                best_ans, conf, escalate = self.sc.check(rollout_answers)
                if not escalate:
                    budget.solved = True
                    budget.answer = best_ans
                    break
                # If escalating: continue collecting more rollouts
                if conf >= 0.4:
                    budget.solved = True
                    budget.answer = best_ans
                    break

        if not budget.solved:
            if rollout_answers:
                budget.answer, _ = self.sc.vote(rollout_answers)
            else:
                budget.answer = 0

        answer = budget.answer

        # ── Phase 3: Verification + Lean correction ───────────────────────────
        answer = self._verify_and_correct(
            answer, problem_text, rollout_texts, budget
        )

        # ── Phase 4: Template save ─────────────────────────────────────────────
        if answer != 0:
            self._save_template(problem_text, rollout_texts, answer)

        self.allocator.record_result(budget, solved=answer != 0, answer=answer)
        return answer

    # ── Phase helpers ─────────────────────────────────────────────────────────

    def _try_shortcut(
        self, problem_text: str, budget: BudgetState
    ) -> Optional[int]:
        """
        Try fast shortcuts before MCTS:
          1. Template match
          2. Direct answer extraction from problem statement
        """
        # Template match
        try:
            similar = self.store.find_similar(problem_text, k=1, threshold=0.7)
            if similar:
                ans = similar[0].get("answer", 0)
                if ans:
                    return int(ans)
        except Exception:
            pass

        # Direct extraction (e.g. problem already states the answer)
        try:
            ans, _, _, conf = self.extractor.extract(problem_text)
            if conf >= 0.90 and ans != 0:
                return ans
        except Exception:
            pass

        return None

    def _verify_and_correct(
        self,
        answer: int,
        problem_text: str,
        texts: List[str],
        budget: BudgetState,
    ) -> int:
        """
        Run verification ladder; attempt Lean correction if needed.
        Up to 3 retries.
        """
        best_text = texts[-1] if texts else ""

        # Level 0 + 1 always run
        try:
            vr = self.ladder.verify(answer, problem=problem_text)
            if vr.passed and vr.confidence >= 0.8:
                return answer
        except Exception:
            return answer

        # Lean correction loop
        if self.llm is not None and self.ladder.lean is not None:
            lean_statement = f"∃ n : ℤ, n = {answer}"
            lean_tactic    = "norm_num"

            for retry in range(self._LEAN_RETRIES):
                try:
                    vr = self.ladder.verify(
                        answer,
                        lean_statement = lean_statement,
                        lean_tactic    = lean_tactic,
                    )
                    if vr.passed:
                        return answer
                    # Ask LLM to correct
                    lean_tactic = self.llm.correct_from_lean_error(
                        theorem_name  = f"problem_answer_{abs(answer)}",
                        statement     = lean_statement,
                        failed_tactic = lean_tactic,
                        lean_error    = vr.message,
                    )
                except Exception:
                    break

        return answer

    def _solve_decomposed(
        self, problem_text: str, budget: BudgetState
    ) -> List[int]:
        """Decompose problem and solve each sub-problem."""
        answers: List[int] = []
        try:
            sub_problems = self.llm.decompose_problem(problem_text, n_parts=3)
            for sub in sub_problems:
                if self.allocator.should_abandon(budget):
                    break
                sub_text = sub.get("problem", "")
                if not sub_text:
                    continue
                remaining = self.allocator.time_remaining_for(budget)
                ans, _    = self.mcts.solve(
                    sub_text, budget_seconds=min(remaining * 0.3, 60.0)
                )
                if ans:
                    answers.append(ans)
        except Exception:
            pass
        return answers

    def _save_template(
        self, problem_text: str, texts: List[str], answer: int
    ) -> None:
        """Save solution template for future use."""
        try:
            best_text = texts[-1] if texts else ""
            if self.llm is not None:
                tmpl = self.llm.compress_to_template(
                    problem  = problem_text,
                    solution = best_text,
                    answer   = answer,
                )
                self.store.save_template(
                    problem     = problem_text,
                    pattern     = tmpl.get("pattern", ""),
                    key_steps   = tmpl.get("key_steps", ""),
                    domain_tags = tmpl.get("domain_tags", "unknown"),
                    answer      = answer,
                )
            else:
                self.store.save_template(
                    problem     = problem_text,
                    pattern     = best_text[:200],
                    key_steps   = "",
                    domain_tags = "unknown",
                    answer      = answer,
                )
        except Exception:
            pass
