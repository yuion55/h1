# cell_09_mathrag.py
"""
CTRL-MATH v5 — Mathematical Knowledge Retrieval (MathRAG)

25+ hand-curated competition math theorems.
TF-IDF vectorization with NumPy.
Cosine similarity via @njit(parallel=True, cache=True).
Cache to .npz after first build.
Retrieval < 20ms.
"""

from __future__ import annotations

import math
import os
import re
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from numba import njit, prange

# ── Theorem Database (25+ entries) ────────────────────────────────────────────

THEOREM_DATABASE: List[Dict[str, Any]] = [
    {
        "name": "Fermat's Little Theorem",
        "statement": "If p is prime and gcd(a, p) = 1, then a^(p-1) ≡ 1 (mod p).",
        "keywords": ["prime", "modular", "exponent", "congruence", "fermat"],
        "domain": "number_theory",
        "lean4_hint": "Nat.ModEq, Finset.card_units_zmod_prime",
    },
    {
        "name": "Euler's Totient Theorem",
        "statement": "If gcd(a, n) = 1, then a^φ(n) ≡ 1 (mod n).",
        "keywords": ["euler", "totient", "phi", "coprime", "modular"],
        "domain": "number_theory",
        "lean4_hint": "ZMod.units_pow_card_sub_one",
    },
    {
        "name": "Chinese Remainder Theorem",
        "statement": "If m_1,...,m_k are pairwise coprime, the system x ≡ a_i (mod m_i) has a unique solution mod M = m_1...m_k.",
        "keywords": ["chinese", "remainder", "crt", "coprime", "system", "congruence"],
        "domain": "number_theory",
        "lean4_hint": "ChineseRemainder, ZMod.chineseRemainder",
    },
    {
        "name": "Bézout's Identity",
        "statement": "For integers a, b with gcd d = gcd(a, b), there exist integers x, y such that ax + by = d.",
        "keywords": ["bezout", "gcd", "linear combination", "integer", "diophantine"],
        "domain": "number_theory",
        "lean4_hint": "Int.gcd_eq_gcd_ab",
    },
    {
        "name": "Wilson's Theorem",
        "statement": "p is prime iff (p-1)! ≡ -1 (mod p).",
        "keywords": ["wilson", "prime", "factorial", "congruence"],
        "domain": "number_theory",
        "lean4_hint": "ZMod.wilsons_lemma",
    },
    {
        "name": "Quadratic Reciprocity",
        "statement": "For odd primes p ≠ q, (p/q)(q/p) = (-1)^((p-1)(q-1)/4).",
        "keywords": ["quadratic", "reciprocity", "legendre", "prime", "symbol"],
        "domain": "number_theory",
        "lean4_hint": "ZMod.quadraticReciprocity",
    },
    {
        "name": "Cauchy-Schwarz Inequality",
        "statement": "(Σa_ib_i)^2 ≤ (Σa_i^2)(Σb_i^2).",
        "keywords": ["cauchy", "schwarz", "inequality", "sum", "product"],
        "domain": "algebra",
        "lean4_hint": "inner_mul_le_norm_mul_norm",
    },
    {
        "name": "AM-GM Inequality",
        "statement": "For nonneg reals, (a_1+...+a_n)/n ≥ (a_1...a_n)^(1/n).",
        "keywords": ["am", "gm", "arithmetic", "geometric", "mean", "inequality"],
        "domain": "algebra",
        "lean4_hint": "Real.inner_le_iff",
    },
    {
        "name": "Pigeonhole Principle",
        "statement": "If n+1 objects are placed in n boxes, some box contains at least 2 objects.",
        "keywords": ["pigeonhole", "box", "object", "counting", "average"],
        "domain": "combinatorics",
        "lean4_hint": "Finset.exists_lt_card_fiber_of_nsmul_lt_card",
    },
    {
        "name": "Inclusion-Exclusion Principle",
        "statement": "|A₁∪...∪Aₙ| = Σ|Aᵢ| - Σ|Aᵢ∩Aⱼ| + ... ± |A₁∩...∩Aₙ|.",
        "keywords": ["inclusion", "exclusion", "union", "intersection", "counting"],
        "domain": "combinatorics",
        "lean4_hint": "Finset.card_union_add_card_inter",
    },
    {
        "name": "Stars and Bars",
        "statement": "The number of ways to put n identical items into k distinct bins is C(n+k-1, k-1).",
        "keywords": ["stars", "bars", "combinations", "distribution", "bins"],
        "domain": "combinatorics",
        "lean4_hint": "Nat.multichoose",
    },
    {
        "name": "Vieta's Formulas",
        "statement": "For polynomial x^n + a_{n-1}x^{n-1}+...+a_0 with roots r_1,...,r_n: Σr_i = -a_{n-1}, Πr_i = (-1)^n a_0.",
        "keywords": ["vieta", "polynomial", "roots", "symmetric", "coefficients"],
        "domain": "algebra",
        "lean4_hint": "Polynomial.roots, Polynomial.Vieta",
    },
    {
        "name": "Newton's Identities",
        "statement": "Power sums p_k and elementary symmetric polynomials e_k satisfy a recurrence.",
        "keywords": ["newton", "power", "sum", "symmetric", "polynomial", "recurrence"],
        "domain": "algebra",
        "lean4_hint": "MvPolynomial.Newton",
    },
    {
        "name": "Ptolemy's Theorem",
        "statement": "For a cyclic quadrilateral ABCD: AC·BD = AB·CD + AD·BC.",
        "keywords": ["ptolemy", "cyclic", "quadrilateral", "diagonal", "geometry"],
        "domain": "geometry",
        "lean4_hint": "EuclideanGeometry.ptolemy",
    },
    {
        "name": "Power Mean Inequality",
        "statement": "M_r(x) ≤ M_s(x) when r ≤ s, where M_r is the r-th power mean.",
        "keywords": ["power", "mean", "inequality", "monotone"],
        "domain": "algebra",
        "lean4_hint": "NNReal.pow_arith_mean_le_arith_mean_pow",
    },
    {
        "name": "Jensen's Inequality",
        "statement": "For convex f: f(E[X]) ≤ E[f(X)].",
        "keywords": ["jensen", "convex", "inequality", "expectation", "function"],
        "domain": "algebra",
        "lean4_hint": "ConvexOn.smul_le_sum",
    },
    {
        "name": "Burnside's Lemma",
        "statement": "|X/G| = (1/|G|) Σ_{g∈G} |X^g| (number of orbits via fixed points).",
        "keywords": ["burnside", "orbit", "group", "symmetry", "counting"],
        "domain": "combinatorics",
        "lean4_hint": "MulAction.card_orbit_eq",
    },
    {
        "name": "Cayley's Formula",
        "statement": "The number of labeled trees on n vertices is n^(n-2).",
        "keywords": ["cayley", "tree", "labeled", "prufer", "counting"],
        "domain": "combinatorics",
        "lean4_hint": "Finset.card_labeled_trees",
    },
    {
        "name": "Lucas' Theorem",
        "statement": "C(m, n) ≡ Π C(m_i, n_i) (mod p) where m_i, n_i are base-p digits.",
        "keywords": ["lucas", "binomial", "prime", "modular", "digits"],
        "domain": "number_theory",
        "lean4_hint": "Nat.lucas_theorem",
    },
    {
        "name": "Bertrand's Postulate",
        "statement": "For every integer n > 1, there exists a prime p with n < p < 2n.",
        "keywords": ["bertrand", "prime", "gap", "interval"],
        "domain": "number_theory",
        "lean4_hint": "Nat.bertrand",
    },
    {
        "name": "Möbius Inversion",
        "statement": "If g(n) = Σ_{d|n} f(d), then f(n) = Σ_{d|n} μ(d)g(n/d).",
        "keywords": ["mobius", "inversion", "multiplicative", "divisor", "sum"],
        "domain": "number_theory",
        "lean4_hint": "ArithmeticFunction.moebius_inversion",
    },
    {
        "name": "Dirichlet's Theorem on Primes",
        "statement": "If gcd(a, d) = 1, there are infinitely many primes p ≡ a (mod d).",
        "keywords": ["dirichlet", "primes", "arithmetic progression", "infinite"],
        "domain": "number_theory",
        "lean4_hint": "Nat.setOf_prime_and_eq_mod_infinite",
    },
    {
        "name": "Shoelace Formula",
        "statement": "Area = (1/2)|Σ(x_i·y_{i+1} - x_{i+1}·y_i)| for polygon vertices.",
        "keywords": ["shoelace", "area", "polygon", "lattice", "coordinate", "geometry"],
        "domain": "geometry",
        "lean4_hint": "MeasureTheory.area_eq_shoelace",
    },
    {
        "name": "Pick's Theorem",
        "statement": "Area = I + B/2 - 1 for lattice polygon with I interior and B boundary points.",
        "keywords": ["pick", "lattice", "polygon", "interior", "boundary", "area"],
        "domain": "geometry",
        "lean4_hint": "Nat.picks_theorem",
    },
    {
        "name": "Lagrange's Four-Square Theorem",
        "statement": "Every positive integer can be expressed as the sum of four integer squares.",
        "keywords": ["lagrange", "four square", "sum of squares", "integer"],
        "domain": "number_theory",
        "lean4_hint": "Nat.sum_four_squares",
    },
    {
        "name": "Catalan Number Formula",
        "statement": "C_n = C(2n, n) / (n+1) counts many combinatorial structures.",
        "keywords": ["catalan", "ballot", "path", "parentheses", "counting"],
        "domain": "combinatorics",
        "lean4_hint": "Nat.catalan",
    },
    {
        "name": "Lifting the Exponent Lemma",
        "statement": "For odd prime p with p|a-b but p∤a,b: v_p(a^n - b^n) = v_p(a-b) + v_p(n).",
        "keywords": ["lte", "lifting", "exponent", "valuation", "prime"],
        "domain": "number_theory",
        "lean4_hint": "multiplicity.Finset.pow_prime_of_prime_pow_dvd",
    },
]

_N_DOCS = len(THEOREM_DATABASE)


# ── Vectorized cosine similarity ──────────────────────────────────────────────

@njit(parallel=True, cache=True)
def cosine_similarity_batch(
    query_vec: np.ndarray,    # shape (D,) float64
    doc_matrix: np.ndarray,   # shape (N, D) float64
) -> np.ndarray:
    """
    Compute cosine similarity between query_vec and each row of doc_matrix.
    Returns shape (N,) float64.
    < 1ms for 600 docs.
    """
    N = doc_matrix.shape[0]
    scores = np.zeros(N, dtype=np.float64)
    q_norm = 0.0
    for d in range(query_vec.shape[0]):
        q_norm += query_vec[d] * query_vec[d]
    q_norm = math.sqrt(q_norm) + 1e-10

    for i in prange(N):
        dot   = 0.0
        d_norm = 0.0
        for d in range(doc_matrix.shape[1]):
            dot    += query_vec[d] * doc_matrix[i, d]
            d_norm += doc_matrix[i, d] * doc_matrix[i, d]
        d_norm = math.sqrt(d_norm) + 1e-10
        scores[i] = dot / (q_norm * d_norm)
    return scores


# ── MathRAG ───────────────────────────────────────────────────────────────────

class MathRAG:
    """
    Retrieves relevant theorems for a math problem using TF-IDF + cosine similarity.
    Caches the TF-IDF matrix to .npz after first build.
    """

    _CACHE_PATH = "/tmp/mathrag_tfidf.npz"

    def __init__(
        self,
        database: List[Dict[str, Any]] = THEOREM_DATABASE,
        cache_path: str                = _CACHE_PATH,
    ):
        self.db         = database
        self.cache_path = cache_path
        self._vocab:  Optional[List[str]]  = None
        self._idf:    Optional[np.ndarray] = None
        self._tfidf:  Optional[np.ndarray] = None

        self.build_index()

    # ── Index building ─────────────────────────────────────────────────────────

    def build_index(self) -> None:
        """Build or load TF-IDF matrix."""
        if os.path.exists(self.cache_path):
            try:
                data = np.load(self.cache_path, allow_pickle=True)
                self._vocab  = list(data["vocab"])
                self._idf    = data["idf"].astype(np.float64)
                self._tfidf  = data["tfidf"].astype(np.float64)
                return
            except Exception:
                pass

        self._build_tfidf()
        np.savez(
            self.cache_path,
            vocab=np.array(self._vocab),
            idf=self._idf,
            tfidf=self._tfidf,
        )

    def _tokenize(self, text: str) -> List[str]:
        """Simple word tokenizer (lowercase, remove punctuation)."""
        text = text.lower()
        text = re.sub(r"[^a-z0-9\s]", " ", text)
        return [w for w in text.split() if len(w) > 1]

    def _build_tfidf(self) -> None:
        """Build TF-IDF from scratch."""
        docs: List[List[str]] = []
        for entry in self.db:
            tokens = (
                self._tokenize(entry["statement"])
                + self._tokenize(entry["name"])
                + entry["keywords"]
            )
            docs.append(tokens)

        # Build vocabulary
        from collections import Counter
        all_words: Counter = Counter()
        for doc in docs:
            all_words.update(set(doc))  # df counting

        # Keep words appearing in at most N-1 docs (remove stopwords)
        self._vocab = [
            w for w, df in all_words.items()
            if 1 <= df <= len(docs) - 1
        ]
        if not self._vocab:
            self._vocab = list(all_words.keys())

        V = len(self._vocab)
        N = len(docs)
        word2idx = {w: i for i, w in enumerate(self._vocab)}

        # TF matrix
        tf = np.zeros((N, V), dtype=np.float64)
        for i, doc in enumerate(docs):
            cnt = Counter(doc)
            total = sum(cnt.values()) or 1
            for w, c in cnt.items():
                if w in word2idx:
                    tf[i, word2idx[w]] = c / total

        # IDF
        df = np.zeros(V, dtype=np.float64)
        for i, doc in enumerate(docs):
            for w in set(doc):
                if w in word2idx:
                    df[word2idx[w]] += 1.0
        self._idf = np.log((N + 1.0) / (df + 1.0)) + 1.0

        # TF-IDF
        self._tfidf = tf * self._idf[np.newaxis, :]

    # ── Retrieval ──────────────────────────────────────────────────────────────

    def _vectorize_query(self, query: str) -> np.ndarray:
        """TF-IDF vectorize a query string."""
        from collections import Counter
        tokens = self._tokenize(query)
        V      = len(self._vocab)
        word2idx = {w: i for i, w in enumerate(self._vocab)}
        cnt   = Counter(tokens)
        total = sum(cnt.values()) or 1
        vec   = np.zeros(V, dtype=np.float64)
        for w, c in cnt.items():
            if w in word2idx:
                vec[word2idx[w]] = (c / total) * self._idf[word2idx[w]]
        return vec

    def retrieve(
        self,
        query: str,
        k: int      = 5,
        domain: str = "",
    ) -> List[Dict[str, Any]]:
        """
        Retrieve top-k theorems for query.
        Optionally filter by domain prefix.
        """
        if self._tfidf is None or len(self._vocab) == 0:
            return []

        q_vec  = self._vectorize_query(query)
        scores = cosine_similarity_batch(q_vec, self._tfidf)

        # Domain filter: boost matching domain
        if domain:
            for i, entry in enumerate(self.db):
                if entry["domain"] == domain:
                    scores[i] *= 1.2

        order = np.argsort(-scores)
        results = []
        for idx in order[:k]:
            results.append({**self.db[idx], "score": float(scores[idx])})
        return results

    def format_for_prompt(self, theorems: List[Dict[str, Any]]) -> str:
        """Format retrieved theorems for inclusion in LLM prompt."""
        if not theorems:
            return ""
        parts = ["Relevant theorems:"]
        for i, thm in enumerate(theorems, 1):
            parts.append(
                f"{i}. {thm['name']}: {thm['statement']}"
                + (f" [Lean4: {thm['lean4_hint']}]" if thm.get("lean4_hint") else "")
            )
        return "\n".join(parts)


# ── Warm-up JIT ───────────────────────────────────────────────────────────────
def _warmup_jit() -> None:
    q = np.ones(4, dtype=np.float64)
    m = np.ones((3, 4), dtype=np.float64)
    cosine_similarity_batch(q, m)


_warmup_jit()
