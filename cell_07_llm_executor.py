# cell_07_llm_executor.py
"""
IMPLEMENTATION REQUIREMENT:
Use Flash Attention 2 if available (reduces VRAM by 30%, increases throughput 2×).
Batch multiple short prompts together when proposing candidates.
Use KV-cache sharing across rollout calls for the same problem.

TIR (Tool-Integrated Reasoning) sandbox:
  - Exposes geometry_tool, vp_factorial_jit, sympy.solve, z3
  - Format: CoT → Python code → verify → <answer>12345</answer>
"""

import re
from typing import List, Dict, Any, Optional

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    import torch
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False

# ── TIR System Prompt ─────────────────────────────────────────────────────────

TIR_SYSTEM_PROMPT = """\
You are a competition mathematics solver using Tool-Integrated Reasoning (TIR).

Available tools in the Python sandbox:
  - geometry_tool.prove_concyclic(A, B, C, D)
  - geometry_tool.add_point(name, x, y)
  - geometry_tool.triangle_area(A, B, C)
  - geometry_tool.triangle_properties(A, B, C)
  - geometry_tool.distance(A, B)
  - geometry_tool.angle(vertex, A, B)
  - geometry_tool.circumcircle(A, B, C)
  - vp_factorial_jit(n, p)  # p-adic valuation of n!
  - sympy.solve(), sympy.simplify(), sympy.factor()
  - z3.Solver(), z3.Int(), z3.And(), z3.Or()
  - math, numpy, itertools, functools

Format your answer as:
1. Chain-of-thought reasoning
2. Python code block to compute/verify
3. Final answer: <answer>INTEGER</answer>

Always verify your answer numerically before outputting.
The answer MUST be a non-negative integer.
"""

# ── Sandbox for TIR code execution ────────────────────────────────────────────

def _build_tir_sandbox() -> Dict[str, Any]:
    """
    Build a restricted sandbox namespace for TIR code execution.
    Includes math tools but blocks dangerous builtins.
    """
    import math
    import numpy as np

    sandbox: Dict[str, Any] = {
        "__builtins__": {
            "abs": abs, "all": all, "any": any, "bin": bin,
            "bool": bool, "chr": chr, "dict": dict, "divmod": divmod,
            "enumerate": enumerate, "filter": filter, "float": float,
            "frozenset": frozenset, "hasattr": hasattr, "hash": hash,
            "hex": hex, "int": int, "isinstance": isinstance,
            "issubclass": issubclass, "iter": iter, "len": len,
            "list": list, "map": map, "max": max, "min": min,
            "next": next, "oct": oct, "ord": ord, "pow": pow,
            "print": print, "range": range, "repr": repr,
            "reversed": reversed, "round": round, "set": set,
            "slice": slice, "sorted": sorted, "str": str,
            "sum": sum, "tuple": tuple, "type": type, "zip": zip,
        },
        "math": math,
        "np": np,
        "numpy": np,
    }

    # Add sympy if available
    try:
        import sympy
        sandbox["sympy"] = sympy
        sandbox["sp"] = sympy
    except ImportError:
        pass

    # Add z3 if available
    try:
        import z3
        sandbox["z3"] = z3
    except ImportError:
        pass

    # Add itertools, functools
    try:
        import itertools
        import functools
        sandbox["itertools"] = itertools
        sandbox["functools"] = functools
    except ImportError:
        pass

    # Add Numba JIT kernels
    try:
        from cell_02a_numba_nt import vp_factorial_jit, vp_jit, fib_jit, powmod_batch
        sandbox["vp_factorial_jit"] = vp_factorial_jit
        sandbox["vp_jit"] = vp_jit
        sandbox["fib_jit"] = fib_jit
        sandbox["powmod_batch"] = powmod_batch
    except ImportError:
        pass

    # Add geometry tool
    try:
        from cell_04g_geometry_prover import geometry_tool
        sandbox["geometry_tool"] = geometry_tool
    except ImportError:
        pass

    return sandbox


class LLMExecutor:
    """
    LLM executor using Qwen2.5-Math-14B-Instruct with Flash Attention 2,
    4-bit quantization, TIR sandbox, and batched forward pass.
    """

    def __init__(self, device: str = "cuda",
                 model_id: str = "Qwen/Qwen2.5-Math-14B-Instruct"):
        if not HAS_TRANSFORMERS:
            raise ImportError("transformers package required for LLMExecutor")

        self.model_id = model_id
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id, trust_remote_code=True
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map="auto",
            quantization_config=bnb_config,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
        )
        self.model.eval()
        self._kv_cache = None
        self._sandbox = _build_tir_sandbox()
        print(f"✅ LLM loaded ({model_id}) with Flash Attention 2 + bfloat16 + NF4 + TIR sandbox")

    def execute_tir_code(self, code: str) -> Dict[str, Any]:
        """
        Execute Python code in the TIR sandbox.
        Returns dict with 'output', 'error', and 'variables'.
        """
        sandbox = dict(self._sandbox)  # fresh copy
        result = {"output": None, "error": None, "variables": {}}

        try:
            import io
            import contextlib

            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                exec(code, sandbox)  # noqa: S102

            result["output"] = stdout.getvalue()
            # Extract any 'answer' variable from the sandbox
            if "answer" in sandbox:
                result["variables"]["answer"] = sandbox["answer"]
            if "result" in sandbox:
                result["variables"]["result"] = sandbox["result"]

        except Exception as e:
            result["error"] = str(e)

        return result

    def propose_operations_batched(self, states: list, k: int = 16) -> List[List[Dict]]:
        """
        Batch multiple states into a single LLM forward pass.
        Returns list of operation lists, one per state.
        """
        prompts = [self._build_prompt(s, k) for s in states]
        inputs  = self.tokenizer(
            prompts, return_tensors="pt", padding=True, truncation=True,
            max_length=2048
        ).to("cuda")
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.3,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        results = []
        for i, out in enumerate(outputs):
            raw = self.tokenizer.decode(
                out[inputs["input_ids"].shape[1]:], skip_special_tokens=True
            )
            results.append(self._parse_operations(raw, k))
        return results

    def propose_operations(self, state, k: int = 16) -> List[Dict]:
        """Single-state wrapper around batched version."""
        return self.propose_operations_batched([state], k)[0]

    def _build_prompt(self, state, k: int) -> str:
        """Build a TIR math problem solving prompt from the state."""
        problem = getattr(state, 'problem_text', str(state))
        domain  = getattr(state, 'domain', 'unknown')
        facts   = getattr(state, 'facts', {})

        facts_str = ""
        if facts:
            facts_str = "\nKnown facts:\n" + "\n".join(f"  - {k}: {v}" for k, v in facts.items())

        prompt = (
            f"{TIR_SYSTEM_PROMPT}\n\n"
            f"Problem: {problem}\n"
            f"Domain: {domain}{facts_str}\n\n"
            f"Propose {k} mathematical operations or transformations to solve this problem.\n"
            f"Format each operation as: OPERATION: <type> | PARAMS: <parameters>\n"
            f"Think step by step.\n"
        )
        return prompt

    def _parse_operations(self, raw: str, k: int) -> List[Dict]:
        """Parse raw LLM output into a list of operation dicts."""
        ops = []
        pattern = re.compile(r'OPERATION:\s*(\w+)\s*\|\s*PARAMS:\s*(.+)', re.IGNORECASE)
        for match in pattern.finditer(raw):
            op_type = match.group(1).strip()
            params  = match.group(2).strip()
            ops.append({"type": op_type, "params": params, "raw": match.group(0)})

        # Fallback: extract numbered steps if no OPERATION format found
        if not ops:
            lines = [l.strip() for l in raw.split('\n') if l.strip()]
            for i, line in enumerate(lines[:k]):
                if line:
                    ops.append({"type": "step", "params": line, "raw": line})

        return ops[:k]
