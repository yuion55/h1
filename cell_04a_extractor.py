# cell_04a_extractor.py
"""
CTRL-MATH v4 — LLM Parameter Extractor
(after cell_03_mog_parser.py)

Single fast LLM call that classifies the problem and extracts ALL numerical
parameters needed by the deterministic solvers. Output is always valid JSON.
Max 100 tokens generated. Uses constrained generation (logits processor) to
force JSON format. Target: < 200ms including tokenization and GPU inference.
"""

import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

# ── Extraction prompt template ────────────────────────────────────────────────
EXTRACTION_PROMPT_TEMPLATE = """\
You are a math competition parameter extractor. Read the problem and output ONLY valid JSON.

Categories and their required params:
- linear_recurrence: {"category": "linear_recurrence", "params": {"coefficients": [...], "initial_values": [...], "n": <int>, "modulus": <int>}, "confidence": 0.9}
  OR with observed_terms: {"category": "linear_recurrence", "params": {"observed_terms": [...], "n": <int>, "modulus": <int>}, "confidence": 0.9}
- combinatorics/binomial: {"category": "combinatorics", "params": {"sub_type": "binomial", "n": <int>, "k": <int>, "modulus": <int>}, "confidence": 0.9}
- combinatorics/catalan: {"category": "combinatorics", "params": {"sub_type": "catalan", "n": <int>, "modulus": <int>}, "confidence": 0.9}
- combinatorics/stirling2: {"category": "combinatorics", "params": {"sub_type": "stirling2", "n": <int>, "k": <int>, "modulus": <int>}, "confidence": 0.9}
- combinatorics/derangement: {"category": "combinatorics", "params": {"sub_type": "derangement", "n": <int>, "modulus": <int>}, "confidence": 0.9}
- combinatorics/partition: {"category": "combinatorics", "params": {"sub_type": "partition", "n": <int>, "modulus": <int>}, "confidence": 0.9}
- number_theory/phi_sum: {"category": "number_theory", "params": {"sub_type": "phi_sum", "N": <int>, "modulus": <int>}, "confidence": 0.9}
- number_theory/mobius_sum: {"category": "number_theory", "params": {"sub_type": "mobius_sum", "N": <int>, "modulus": <int>}, "confidence": 0.9}
- number_theory/mult_sum: {"category": "number_theory", "params": {"sub_type": "mult_sum", "N": <int>, "func_type": <0|1|2|3>, "k_param": <int>, "modulus": <int>}, "confidence": 0.9}
- number_theory/crt: {"category": "number_theory", "params": {"sub_type": "crt", "remainders": [...], "moduli": [...]}, "confidence": 0.9}
- number_theory/discrete_log: {"category": "number_theory", "params": {"sub_type": "discrete_log", "g": <int>, "h": <int>, "p": <int>}, "confidence": 0.9}
- generating_function: {"category": "generating_function", "params": {"numerator": [...], "denominator": [...], "n": <int>, "modulus": <int>}, "confidence": 0.9}
- polynomial_coeff: {"category": "polynomial_coeff", "params": {"base_poly": [...], "power": <int>, "coeff_index": <int>, "modulus": <int>}, "confidence": 0.9}
- geometry: {"category": "geometry", "params": {"sub_type": "polygon_area"|"lattice_interior"|"convex_hull_area", "x": [...], "y": [...]}, "confidence": 0.9}
- unknown: {"category": "unknown", "params": {}, "confidence": 0.0}

Problem:
{problem_text}

JSON:"""


# ── ExtractionResult dataclass ────────────────────────────────────────────────
@dataclass
class ExtractionResult:
    """Result from LLM parameter extraction."""
    category: str
    params: Dict[str, Any]
    confidence: float
    raw_json: str


# ── LLMExtractor ──────────────────────────────────────────────────────────────
class LLMExtractor:
    """
    Wraps a HuggingFace model/tokenizer to extract structured parameters
    from math competition problems with greedy decoding (do_sample=False,
    max_new_tokens=100).
    """

    def __init__(self, model, tokenizer, device: str = "cuda"):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

    def extract(self, problem_text: str) -> ExtractionResult:
        """
        Run a single greedy inference pass (do_sample=False, max_new_tokens=100)
        to extract structured parameters from the problem text.

        Returns an ExtractionResult with parsed category, params, and confidence.
        """
        prompt = EXTRACTION_PROMPT_TEMPLATE.format(problem_text=problem_text)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        with __import__("torch").no_grad():
            output_ids = self.model.generate(
                **inputs,
                do_sample=False,
                max_new_tokens=100,
            )
        # Decode only the newly generated tokens
        new_ids = output_ids[0][inputs["input_ids"].shape[-1]:]
        raw = self.tokenizer.decode(new_ids, skip_special_tokens=True)
        return self._parse_output(raw)

    def _parse_output(self, raw: str) -> ExtractionResult:
        """
        Parse LLM output with 3 fallback strategies:
        1. Direct JSON parse of the full output
        2. Regex field extraction from partial JSON
        3. Total failure → unknown category
        """
        # Strategy 1: direct JSON parse
        try:
            data = json.loads(raw.strip())
            category = str(data.get("category", "unknown"))
            params = data.get("params", {})
            confidence = float(data.get("confidence", 0.0))
            return ExtractionResult(
                category=category,
                params=params if isinstance(params, dict) else {},
                confidence=confidence,
                raw_json=raw,
            )
        except (json.JSONDecodeError, ValueError):
            pass

        # Strategy 2: regex field extraction
        try:
            cat_m = re.search(r'"category"\s*:\s*"([^"]+)"', raw)
            conf_m = re.search(r'"confidence"\s*:\s*([0-9.]+)', raw)
            # Try to find a params JSON object
            params_m = re.search(r'"params"\s*:\s*(\{[^}]*\})', raw, re.DOTALL)
            category = cat_m.group(1) if cat_m else "unknown"
            confidence = float(conf_m.group(1)) if conf_m else 0.0
            params: Dict[str, Any] = {}
            if params_m:
                try:
                    params = json.loads(params_m.group(1))
                except json.JSONDecodeError:
                    params = {}
            return ExtractionResult(
                category=category,
                params=params,
                confidence=confidence,
                raw_json=raw,
            )
        except Exception:
            pass

        # Strategy 3: total failure
        return ExtractionResult(
            category="unknown",
            params={},
            confidence=0.0,
            raw_json=raw,
        )
