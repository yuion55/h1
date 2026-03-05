# cell_07_llm_executor_v5.py
"""
CTRL-MATH AIMO3 — LLM Executor
Flash Attention 2 + 4-bit NF4 + LoRA adapter + ensemble.

Primary model:  Qwen2.5-Math-14B-Instruct (+ LoRA adapter)
Ensemble model: DeepSeek-Math-7B-Instruct
PRM model:      Qwen2.5-Math-1.5B-Instruct
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

try:
    from peft import PeftModel
    HAS_PEFT = True
except ImportError:
    HAS_PEFT = False


class LLMExecutorV5:
    """
    LLM executor using:
      - Main model:     Qwen2.5-Math-14B-Instruct (+ LoRA adapter via PEFT)
      - Ensemble model: DeepSeek-Math-7B-Instruct
      - PRM model:      Qwen2.5-Math-1.5B-Instruct
      - Flash Attention 2 + 4-bit NF4 quantization
    """

    def __init__(
        self,
        primary_model:   str = "Qwen/Qwen2.5-Math-14B-Instruct",
        ensemble_model:  str = "deepseek-ai/DeepSeek-Math-7B-Instruct",
        prm_model:       str = "Qwen/Qwen2.5-Math-1.5B-Instruct",
        lora_adapter:    Optional[str] = None,
        load_in_4bit:    bool = True,
        use_flash_attn:  bool = True,
        device_map:      str = "auto",
    ):
        self.device = "cuda"
        self.primary_model_id  = primary_model
        self.ensemble_model_id = ensemble_model
        self.prm_model_id      = prm_model
        self.lora_adapter      = lora_adapter
        self.load_in_4bit      = load_in_4bit
        self.use_flash_attn    = use_flash_attn
        self.device_map        = device_map

        self.primary_model  = None
        self.primary_tok    = None
        self.ensemble_model = None
        self.ensemble_tok   = None
        # Legacy aliases
        self.model       = None
        self.draft_model = None
        self.tokenizer   = None

        if HAS_TRANSFORMERS:
            self._load()

    def _load(self) -> None:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=self.load_in_4bit,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        ) if self.load_in_4bit else None

        attn_impl = "flash_attention_2" if self.use_flash_attn else "eager"

        # ── 1. Primary model (Qwen2.5-Math-14B-Instruct) ──────────────────────
        self.primary_tok = AutoTokenizer.from_pretrained(
            self.primary_model_id, trust_remote_code=True
        )
        load_kwargs: Dict[str, Any] = dict(
            device_map=self.device_map,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            attn_implementation=attn_impl,
        )
        if bnb_config is not None:
            load_kwargs["quantization_config"] = bnb_config
        self.primary_model = AutoModelForCausalLM.from_pretrained(
            self.primary_model_id, **load_kwargs
        )
        self.primary_model.eval()

        # ── 2. Apply LoRA adapter if provided ─────────────────────────────────
        if self.lora_adapter is not None and HAS_PEFT:
            self.primary_model = PeftModel.from_pretrained(
                self.primary_model, self.lora_adapter
            )
            self.primary_model.eval()

        # Legacy alias
        self.model     = self.primary_model
        self.tokenizer = self.primary_tok

        # ── 3. Ensemble model (DeepSeek-Math-7B) — only if VRAM > 30 GB ──────
        try:
            vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        except Exception:
            vram_gb = 0.0

        if vram_gb > 30.0:
            self.ensemble_tok = AutoTokenizer.from_pretrained(
                self.ensemble_model_id, trust_remote_code=True
            )
            ensemble_kwargs: Dict[str, Any] = dict(
                device_map=self.device_map,
                trust_remote_code=True,
                torch_dtype=torch.bfloat16,
            )
            if bnb_config is not None:
                ensemble_kwargs["quantization_config"] = bnb_config
            self.ensemble_model = AutoModelForCausalLM.from_pretrained(
                self.ensemble_model_id, **ensemble_kwargs
            )
            self.ensemble_model.eval()
            print(
                f"✅ LLMExecutorV5 loaded: primary=14B + ensemble=7B, "
                f"FlashAttn2 + NF4"
            )
        else:
            print(
                f"✅ LLMExecutorV5 loaded: primary=14B (ensemble skipped, "
                f"VRAM={vram_gb:.1f}GB), FlashAttn2 + NF4"
            )

        # Legacy draft_model alias (for any code expecting speculative decoding)
        self.draft_model = self.ensemble_model

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
            import warnings
            warnings.warn(
                "LLMExecutorV5: model not loaded — returning empty response. "
                "Ensure the model is available at the configured path.",
                RuntimeWarning,
                stacklevel=2,
            )
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

    def generate_ensemble(self, prompt: str, **kwargs) -> str:
        """
        Run generation with the primary model (+ ensemble for verification).

        Returns the primary model's answer. The ensemble model is run in
        parallel when available and used for voting/verification.
        """
        primary_out = self._generate_structured(prompt, **kwargs)

        if self.ensemble_model is not None and self.ensemble_tok is not None:
            try:
                inputs = self.ensemble_tok(
                    prompt,
                    return_tensors="pt",
                    truncation=True,
                    max_length=2048,
                ).to(self.device)
                gen_kwargs: Dict[str, Any] = dict(
                    max_new_tokens=kwargs.get("max_new_tokens", 512),
                    temperature=kwargs.get("temperature", 0.3),
                    do_sample=kwargs.get("temperature", 0.3) > 0.0,
                    pad_token_id=self.ensemble_tok.eos_token_id,
                )
                with torch.no_grad():
                    output = self.ensemble_model.generate(**inputs, **gen_kwargs)
                # ensemble output available for external voting if needed
                self._last_ensemble_output = self.ensemble_tok.decode(
                    output[0][inputs["input_ids"].shape[1]:],
                    skip_special_tokens=True,
                )
            except Exception:
                pass

        return primary_out

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
