# cell_10_template_store.py
"""
CTRL-MATH v5 — Cross-Problem Learning (TemplateStore)

Stores solution templates and error patterns across problems.
JSON persistence on every save.
Jaccard similarity for retrieval.
Max 200 templates with LRU eviction.
"""

from __future__ import annotations

import json
import os
import re
import time
from typing import Any, Dict, List, Optional, Tuple

_MAX_TEMPLATES   = 200
_MAX_ERROR_PATS  = 500
_PERSIST_PATH    = "/tmp/mathrag_templates.json"


def _tokenize(text: str) -> set:
    """Simple word tokenizer returning a set of tokens."""
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return set(w for w in text.split() if len(w) > 1)


def _jaccard(a: set, b: set) -> float:
    """Jaccard similarity between two token sets."""
    if not a and not b:
        return 1.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0


class TemplateStore:
    """
    Persistent store of solution templates and error patterns.

    Templates: {problem_keywords, pattern, key_steps, domain_tags, answer, timestamp}
    Error patterns: {error_text, step, count, timestamp}
    """

    def __init__(self, persist_path: str = _PERSIST_PATH):
        self.persist_path  = persist_path
        self._templates:    List[Dict[str, Any]] = []
        self._error_pats:   List[Dict[str, Any]] = []
        self._load()

    # ── Persistence ───────────────────────────────────────────────────────────

    def _load(self) -> None:
        if os.path.exists(self.persist_path):
            try:
                with open(self.persist_path, "r") as f:
                    data = json.load(f)
                self._templates  = data.get("templates", [])
                self._error_pats = data.get("error_patterns", [])
            except Exception:
                self._templates  = []
                self._error_pats = []

    def _persist(self) -> None:
        """Write current state to JSON file."""
        try:
            with open(self.persist_path, "w") as f:
                json.dump(
                    {
                        "templates":      self._templates,
                        "error_patterns": self._error_pats,
                    },
                    f,
                    indent=2,
                )
        except Exception:
            pass

    # ── Template API ──────────────────────────────────────────────────────────

    def save_template(
        self,
        problem: str,
        pattern: str,
        key_steps: str,
        domain_tags: str,
        answer: int,
    ) -> None:
        """
        Save a new solution template.
        Evicts oldest entry when MAX_TEMPLATES reached.
        Persists to JSON immediately.
        """
        entry: Dict[str, Any] = {
            "problem":     problem,
            "keywords":    list(_tokenize(problem)),
            "pattern":     pattern,
            "key_steps":   key_steps,
            "domain_tags": domain_tags,
            "answer":      answer,
            "timestamp":   time.time(),
        }
        if len(self._templates) >= _MAX_TEMPLATES:
            # Evict oldest (index 0)
            self._templates.pop(0)
        self._templates.append(entry)
        self._persist()

    def save_error_pattern(
        self,
        error_text: str,
        step: str,
    ) -> None:
        """
        Record an error pattern.
        Deduplicates by incrementing count if identical error_text exists.
        """
        for ep in self._error_pats:
            if ep["error_text"] == error_text:
                ep["count"] += 1
                ep["timestamp"] = time.time()
                self._persist()
                return

        if len(self._error_pats) >= _MAX_ERROR_PATS:
            self._error_pats.pop(0)

        self._error_pats.append(
            {
                "error_text": error_text,
                "step":       step,
                "count":      1,
                "timestamp":  time.time(),
            }
        )
        self._persist()

    def find_similar(
        self,
        problem: str,
        k: int         = 5,
        threshold: float = 0.1,
    ) -> List[Dict[str, Any]]:
        """
        Return top-k templates most similar to `problem` by Jaccard similarity.
        Only returns entries with similarity >= threshold.
        """
        query_tokens = _tokenize(problem)
        scored: List[Tuple[float, Dict[str, Any]]] = []
        for tmpl in self._templates:
            tmpl_tokens = set(tmpl.get("keywords", []))
            sim = _jaccard(query_tokens, tmpl_tokens)
            if sim >= threshold:
                scored.append((sim, tmpl))
        scored.sort(key=lambda x: -x[0])
        return [t for _, t in scored[:k]]

    def format_few_shot(
        self,
        problem: str,
        k: int = 3,
    ) -> str:
        """
        Format top-k similar templates as few-shot examples for LLM prompt.
        """
        similar = self.find_similar(problem, k=k)
        if not similar:
            return ""
        parts = ["Similar solved problems:"]
        for i, tmpl in enumerate(similar, 1):
            parts.append(
                f"\nExample {i}:\n"
                f"  Problem: {tmpl['problem'][:200]}\n"
                f"  Pattern: {tmpl['pattern']}\n"
                f"  Key steps: {tmpl['key_steps']}\n"
                f"  Answer: {tmpl['answer']}"
            )
        return "\n".join(parts)

    def format_error_patterns(
        self,
        k: int = 5,
    ) -> str:
        """
        Format top-k most frequent error patterns as a warning block.
        """
        if not self._error_pats:
            return ""
        sorted_eps = sorted(self._error_pats, key=lambda e: -e["count"])
        parts = ["Known error patterns to avoid:"]
        for ep in sorted_eps[:k]:
            parts.append(f"  - [{ep['count']}×] {ep['error_text'][:150]}")
        return "\n".join(parts)

    # ── Introspection ─────────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self._templates)

    def num_error_patterns(self) -> int:
        return len(self._error_pats)
