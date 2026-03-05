# cell_07_llm_executor_v5.py
"""
CTRL-MATH v5 — LLM Executor with Full Prompt System
Flash Attention 2 + 4-bit NF4 + speculative decoding.

All 7 prompt templates are module-level string constants.
Draft model: Qwen2.5-Math-1.5B
Main model:  Qwen2.5-Math-7B-Instruct
"""

from __future__ import annotations

import re
import time
from typing import Any, Dict, List, Optional, Tuple

# ── Prompt templates (module-level constants) ──────────────────────────────────

PROMPT_SOLVE = """\
You are an expert competition mathematician.

Problem: {problem}
Domain: {domain}
Context: {context}

Solve this problem step by step. Show all reasoning.
At the end, output: ANSWER: <integer>
"""

PROMPT_LEAN_TACTIC = """\
You are a Lean 4 proof expert.

Mathematical statement to prove: {statement}
Known facts: {facts}
Previous attempt (if any): {prev_attempt}

Generate a Lean 4 tactic proof. Output only the tactic block, no imports.
"""

PROMPT_VERIFY_SECOND = """\
You are verifying a mathematical solution using an alternative method.

Problem: {problem}
Proposed answer: {answer}
First method used: {first_method}

Verify this answer using a completely different approach.
Output: VERIFIED: yes/no
CONFIDENCE: <0-100>
REASONING: <your verification>
"""

PROMPT_CORRECT_LEAN = """\
You are correcting a failed Lean 4 proof.

Theorem: {theorem_name}
Statement: {statement}
Failed tactic:
  {failed_tactic}
Lean error:
  {lean_error}

Provide a corrected tactic proof. Output only the corrected tactic.
"""

PROMPT_DECOMPOSE = """\
You are decomposing a complex competition math problem.

Problem: {problem}

Break this into {n_parts} independent sub-problems.
For each sub-problem, output:
SUB_PROBLEM_1: <description>
SUB_ANSWER_1: <expected form>
...
"""

PROMPT_COMPRESS = """\
You are extracting a reusable solution template from a solved problem.

Problem: {problem}
Full solution: {solution}
Answer: {answer}
Domain: {domain}

Extract the key mathematical technique as a template.
Output:
TEMPLATE_PATTERN: <abstract pattern>
KEY_STEPS: <numbered list>
DOMAIN_TAGS: <comma-separated tags>
"""

PROMPT_PROPOSE_STEPS = """\
You are a competition math solver.

Problem context so far:
{context}

Propose {k} distinct next solution steps.
For each step output:
STEP: <mathematical operation or insight>
"""

# ── HAS_TRANSFORMERS guard ─────────────────────────────────────────────────────
try:
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
    )
    import torch
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False


class LLMExecutorV5:
    """
    LLM executor using:
      - Main model:  Qwen2.5-Math-7B-Instruct
      - Draft model: Qwen2.5-Math-1.5B  (speculative decoding)
      - Flash Attention 2 + 4-bit NF4 quantization
    """

    MAIN_MODEL  = "Qwen/Qwen2.5-Math-7B-Instruct"
    DRAFT_MODEL = "Qwen/Qwen2.5-Math-1.5B"

    def __init__(self, device: str = "cuda", load_models: bool = True):
        self.device = device
        self.model       = None
        self.draft_model = None
        self.tokenizer   = None

        if load_models and HAS_TRANSFORMERS:
            self._load()

    def _load(self) -> None:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.MAIN_MODEL, trust_remote_code=True
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            self.MAIN_MODEL,
            device_map="auto",
            quantization_config=bnb_config,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
        )
        self.model.eval()

        # Draft model for speculative decoding (smaller, faster)
        self.draft_model = AutoModelForCausalLM.from_pretrained(
            self.DRAFT_MODEL,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )
        self.draft_model.eval()
        print("✅ LLMExecutorV5 loaded: main=7B + draft=1.5B, FlashAttn2 + NF4")

    # ── Core generation ────────────────────────────────────────────────────────

    def _generate_structured(
        self,
        prompt: str,
        max_new_tokens: int = 512,
        temperature: float  = 0.3,
    ) -> str:
        """
        Generate with speculative decoding via assistant_model kwarg.
        Falls back to greedy if no draft model or no transformers.
        """
        if not HAS_TRANSFORMERS or self.model is None:
            return ""

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=2048,
        ).to(self.device)

        kwargs: Dict[str, Any] = dict(
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=temperature > 0.0,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        if self.draft_model is not None:
            kwargs["assistant_model"] = self.draft_model

        with torch.no_grad():
            output = self.model.generate(**inputs, **kwargs)

        return self.tokenizer.decode(
            output[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        )

    def _parse_structured(self, raw: str) -> Dict[str, str]:
        """
        Parse 'FIELD: value' lines from raw LLM output.
        < 5ms. Returns dict of {field: value}.
        """
        result: Dict[str, str] = {}
        for line in raw.splitlines():
            m = re.match(r"^([A-Z_][A-Z0-9_]*):\s*(.*)", line.strip())
            if m:
                result[m.group(1)] = m.group(2).strip()
        return result

    # ── Public API ─────────────────────────────────────────────────────────────

    def solve_with_reasoning(
        self,
        problem: str,
        domain: str    = "unknown",
        context: str   = "",
    ) -> Dict[str, Any]:
        """Full CoT solve. Returns dict with 'answer', 'reasoning', 'confidence'."""
        prompt = PROMPT_SOLVE.format(
            problem=problem, domain=domain, context=context
        )
        raw    = self._generate_structured(prompt, max_new_tokens=1024)
        parsed = self._parse_structured(raw)
        answer_str = parsed.get("ANSWER", "0")
        try:
            answer = int(re.search(r"-?\d+", answer_str).group())
        except Exception:
            answer = 0
        return {"answer": answer, "reasoning": raw, "confidence": 0.7}

    def generate_lean_tactic(
        self,
        statement: str,
        facts: str       = "",
        prev_attempt: str = "",
    ) -> str:
        """Generate a Lean 4 tactic for the given statement."""
        prompt = PROMPT_LEAN_TACTIC.format(
            statement=statement, facts=facts, prev_attempt=prev_attempt
        )
        return self._generate_structured(prompt, max_new_tokens=256)

    def verify_by_second_method(
        self,
        problem: str,
        answer: int,
        first_method: str = "",
    ) -> Dict[str, Any]:
        """Verify answer using a different approach."""
        prompt = PROMPT_VERIFY_SECOND.format(
            problem=problem, answer=answer, first_method=first_method
        )
        raw    = self._generate_structured(prompt, max_new_tokens=512)
        parsed = self._parse_structured(raw)
        verified = parsed.get("VERIFIED", "no").lower() == "yes"
        try:
            confidence = int(parsed.get("CONFIDENCE", "50")) / 100.0
        except ValueError:
            confidence = 0.5
        return {
            "verified":   verified,
            "confidence": confidence,
            "reasoning":  parsed.get("REASONING", raw),
        }

    def correct_from_lean_error(
        self,
        theorem_name: str,
        statement: str,
        failed_tactic: str,
        lean_error: str,
    ) -> str:
        """Return corrected tactic given a Lean error."""
        prompt = PROMPT_CORRECT_LEAN.format(
            theorem_name=theorem_name,
            statement=statement,
            failed_tactic=failed_tactic,
            lean_error=lean_error,
        )
        return self._generate_structured(prompt, max_new_tokens=256)

    def decompose_problem(
        self,
        problem: str,
        n_parts: int = 3,
    ) -> List[Dict[str, str]]:
        """Decompose a hard problem into n_parts sub-problems."""
        prompt = PROMPT_DECOMPOSE.format(problem=problem, n_parts=n_parts)
        raw    = self._generate_structured(prompt, max_new_tokens=512)
        parts: List[Dict[str, str]] = []
        for i in range(1, n_parts + 1):
            m_prob = re.search(
                rf"SUB_PROBLEM_{i}:\s*(.+?)(?=SUB_PROBLEM_|\Z)", raw, re.S
            )
            m_ans  = re.search(rf"SUB_ANSWER_{i}:\s*(.+?)(?=SUB_|\Z)", raw, re.S)
            if m_prob:
                parts.append({
                    "problem": m_prob.group(1).strip(),
                    "answer_form": m_ans.group(1).strip() if m_ans else "",
                })
        return parts

    def compress_to_template(
        self,
        problem: str,
        solution: str,
        answer: int,
        domain: str = "unknown",
    ) -> Dict[str, str]:
        """Extract reusable template from a solved problem."""
        prompt = PROMPT_COMPRESS.format(
            problem=problem, solution=solution, answer=answer, domain=domain
        )
        raw    = self._generate_structured(prompt, max_new_tokens=256)
        parsed = self._parse_structured(raw)
        return {
            "pattern": parsed.get("TEMPLATE_PATTERN", ""),
            "key_steps": parsed.get("KEY_STEPS", ""),
            "domain_tags": parsed.get("DOMAIN_TAGS", domain),
        }

    def propose_steps_batched(
        self,
        context: str,
        k: int = 8,
    ) -> List[str]:
        """Propose k next solution steps from current context."""
        prompt = PROMPT_PROPOSE_STEPS.format(context=context, k=k)
        raw    = self._generate_structured(prompt, max_new_tokens=512)
        steps: List[str] = []
        for m in re.finditer(r"STEP:\s*(.+?)(?=STEP:|\Z)", raw, re.S):
            steps.append(m.group(1).strip())
        if not steps:
            lines = [l.strip() for l in raw.split("\n") if l.strip()]
            steps = lines[:k]
        return steps[:k]

    # Legacy compatibility
    def propose_operations(self, state, k: int = 8) -> List[Dict]:
        """Legacy API used by older cells."""
        problem = getattr(state, "problem_text", str(state))
        steps   = self.propose_steps_batched(problem, k=k)
        return [{"type": "step", "params": s, "raw": s} for s in steps]

    def propose_operations_batched(
        self, states: list, k: int = 8
    ) -> List[List[Dict]]:
        """Legacy batched API."""
        return [self.propose_operations(s, k) for s in states]


# ── Module-level constant for EXPANSION_K (needed by propose_steps_batched) ───
_EXPANSION_K = 8
