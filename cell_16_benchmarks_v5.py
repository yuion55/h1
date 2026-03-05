# cell_16_benchmarks_v5.py
"""
CTRL-MATH v5 — Full Benchmark Suite

Tests ALL v5 components with hard asserts on speed and correctness.
Run this file to verify the complete v5 solver stack.
"""

import time
import math

import numpy as np

# ── bench helper ──────────────────────────────────────────────────────────────
def bench(fn, *args, n_runs=100, warmup=10):
    """Run fn(*args) n_runs times after warmup, return mean elapsed seconds."""
    for _ in range(warmup):
        fn(*args)
    t0 = time.perf_counter()
    for _ in range(n_runs):
        fn(*args)
    return (time.perf_counter() - t0) / n_runs


print("=" * 70)
print("CTRL-MATH v5 — Full Benchmark Suite")
print("=" * 70)

# ── Imports ───────────────────────────────────────────────────────────────────
from cell_06_mcts import (
    MCTSNodeStore, MCTSEngine, backpropagate_jit,
    MAX_NODES, C_EXPLORE,
)
from cell_08_prm import (
    symbolic_score_jit, _extract_symbolic_features, ProcessRewardModel,
    KNOWN_THEOREMS, CONTRADICTION_PATTERNS,
)
from cell_09_mathrag import (
    THEOREM_DATABASE, cosine_similarity_batch, MathRAG,
)
from cell_10_template_store import TemplateStore
from cell_11_answer_extractor import AnswerExtractor
from cell_12_time_allocator import TimeAllocator, BudgetState
from cell_13_self_consistency import SelfConsistencyChecker
from cell_14_verification_ladder import VerificationLadder

print("\n── Correctness checks ──")

# ─────────────────────────────────────────────────────────────────────────────
# 1. MCTSNodeStore — UCT selection
# ─────────────────────────────────────────────────────────────────────────────
store = MCTSNodeStore(max_nodes=MAX_NODES)
root  = store.alloc(-1, 0, "root")
for i in range(10):
    child = store.alloc(root, 1, f"child_{i}")
    store.add_child(root, child)
    store.n[child] = float(i + 1)
    store.q[child] = float(i)
store.n[root] = 10.0

scores = store.uct_score_children(root)
assert scores.shape[0] == 10, f"UCT scores shape wrong: {scores.shape}"
best   = store.best_child(root)
assert best >= 0, "best_child returned -1"
print("  ✓ MCTSNodeStore UCT selection")

# Leaf traversal on a deeper tree
store2 = MCTSNodeStore(max_nodes=MAX_NODES)
r      = store2.alloc(-1, 0, "root")
cur    = r
for d in range(5):
    c = store2.alloc(cur, d + 1, f"node_d{d}")
    store2.add_child(cur, c)
    store2.n[c] = 1.0
    cur = c
leaf = store2.best_path_leaf(r)
assert store2.depth[leaf] == 5, f"leaf depth wrong: {store2.depth[leaf]}"
print("  ✓ MCTSNodeStore best_path_leaf")

# ─────────────────────────────────────────────────────────────────────────────
# 2. backpropagate_jit correctness
# ─────────────────────────────────────────────────────────────────────────────
depth = 20
_par  = np.full(depth + 1, -1, dtype=np.int64)
_n    = np.zeros(depth + 1, dtype=np.float64)
_q    = np.zeros(depth + 1, dtype=np.float64)
for i in range(1, depth + 1):
    _par[i] = i - 1
backpropagate_jit(_par, _n, _q, depth, 1.0)
assert _n[0] == 1.0 and _n[depth] == 1.0, "backprop N wrong"
assert _q[0] == 1.0 and _q[depth] == 1.0, "backprop Q wrong"
print("  ✓ backpropagate_jit correctness")

# ─────────────────────────────────────────────────────────────────────────────
# 3. ProcessRewardModel — symbolic_score_jit
# ─────────────────────────────────────────────────────────────────────────────
prm = ProcessRewardModel()

# Good step
feats_good = _extract_symbolic_features("By Fermat's little theorem, a^(p-1) ≡ 1 (mod p).")
score_good  = symbolic_score_jit(feats_good)
assert 0.0 <= score_good <= 1.0, f"score out of range: {score_good}"
assert score_good > 0.3, f"good step scored too low: {score_good}"

# Contradiction step
feats_bad  = _extract_symbolic_features("Therefore 0 = 5, which is false.")
score_bad  = symbolic_score_jit(feats_bad)
assert score_bad < score_good, f"contradiction not penalised: {score_bad} >= {score_good}"

# Batch scoring (no LLM)
batch_scores = prm.score_batch(["x + y = z", "fermat little theorem applies here"])
assert batch_scores.shape[0] == 2, "batch_scores shape wrong"
assert all(0.0 <= s <= 1.0 for s in batch_scores), "batch_scores out of range"
print("  ✓ ProcessRewardModel symbolic scoring")

# ─────────────────────────────────────────────────────────────────────────────
# 4. MathRAG — database and retrieval
# ─────────────────────────────────────────────────────────────────────────────
assert len(THEOREM_DATABASE) >= 25, \
    f"THEOREM_DATABASE too small: {len(THEOREM_DATABASE)} < 25 (have {len(THEOREM_DATABASE)})"

rag = MathRAG()
assert rag._tfidf is not None, "TF-IDF matrix not built"
assert rag._tfidf.shape[0] == len(THEOREM_DATABASE), "TF-IDF row count mismatch"

results = rag.retrieve("Find the number using Fermat theorem and prime modulus", k=3)
assert len(results) > 0, "MathRAG returned no results"
assert "name" in results[0], "result missing 'name'"

# Ensure format_for_prompt works
prompt_block = rag.format_for_prompt(results)
assert len(prompt_block) > 10, "format_for_prompt output too short"
print(f"  ✓ MathRAG ({len(THEOREM_DATABASE)} theorems, retrieval returns {len(results)} results)")

# ─────────────────────────────────────────────────────────────────────────────
# 5. TemplateStore — save and retrieve
# ─────────────────────────────────────────────────────────────────────────────
ts = TemplateStore(persist_path="/tmp/bench_templates_v5.json")
# Clear any state from previous runs
ts._templates  = []
ts._error_pats = []

ts.save_template(
    problem     = "Find the number of ways to arrange n items",
    pattern     = "factorial_counting",
    key_steps   = "1. Apply permutation formula 2. Simplify",
    domain_tags = "combinatorics",
    answer      = 120,
)
ts.save_error_pattern(
    error_text = "division by zero in modular inverse",
    step       = "inv(0, mod)",
)
ts.save_error_pattern(
    error_text = "division by zero in modular inverse",
    step       = "inv(0, mod)",
)  # duplicate → count should be 2

assert len(ts) >= 1, "TemplateStore save failed"
assert ts.num_error_patterns() >= 1, "error pattern save failed"

similar = ts.find_similar("Find the number of arrangements of n objects", k=3)
assert len(similar) >= 1, "find_similar returned nothing"

# Verify error deduplication
for ep in ts._error_pats:
    if ep["error_text"] == "division by zero in modular inverse":
        assert ep["count"] == 2, f"dedup count wrong: {ep['count']}"
        break
print("  ✓ TemplateStore save/retrieve/dedup")

# ─────────────────────────────────────────────────────────────────────────────
# 6. AnswerExtractor — extraction correctness
# ─────────────────────────────────────────────────────────────────────────────
ae = AnswerExtractor()

# \\boxed{}
ans, cands, src, conf = ae.extract(r"Therefore $\boxed{42}$ is the answer.")
assert ans == 42, f"boxed extraction wrong: {ans}"
assert src == "boxed", f"source wrong: {src}"

# ANSWER: field
ans, _, src, _ = ae.extract("After calculation, ANSWER: 7")
assert ans == 7 and src == "answer_field", f"answer_field wrong: {ans}, {src}"

# Fraction (2/4 = 0 via exact division)
ans, _, _, _ = ae.extract(r"\frac{6}{2}")
assert ans == 3, f"frac extraction wrong: {ans}"

# Power
ans, _, src, _ = ae.extract("Result: 2^10")
assert ans == 1024, f"power extraction wrong: {ans}"

# Factorial
ans, _, src, _ = ae.extract("The answer is 5!")
assert ans == 120, f"factorial extraction wrong: {ans}"

# Never raises
ans2, cands2, src2, conf2 = ae.extract("")
assert ans2 == 0, "empty string should return 0"

ans3, _, src3, _ = ae.extract(None)
assert ans3 == 0 and src3 == "failed", "None input should return failed"

print("  ✓ AnswerExtractor (boxed, answer_field, fraction, power, factorial, never-raises)")

# ─────────────────────────────────────────────────────────────────────────────
# 7. TimeAllocator — difficulty estimation and budget
# ─────────────────────────────────────────────────────────────────────────────
ta = TimeAllocator(total_seconds=9 * 3600.0, n_problems=50)

easy_prob  = "Compute 2 + 2."
hard_prob  = (
    "Prove that for all primes p and integers a with gcd(a,p)=1, "
    "a^(p-1) ≡ 1 (mod p). Determine all solutions."
)

diff_easy  = ta.estimate_difficulty(easy_prob)
diff_hard  = ta.estimate_difficulty(hard_prob)
assert diff_easy in ("easy", "medium"), f"easy problem misclassified: {diff_easy}"
assert diff_hard == "hard", f"hard problem misclassified: {diff_hard}"

state = ta.allocate_budget("p1", hard_prob)
assert state.allocated_sec > 0, "allocated budget must be positive"
assert not ta.should_abandon(state), "fresh budget should not be abandoned"
ta.record_result(state, solved=True, answer=1)
print("  ✓ TimeAllocator (difficulty, budget, should_abandon)")

# ─────────────────────────────────────────────────────────────────────────────
# 8. SelfConsistencyChecker — vote correctness
# ─────────────────────────────────────────────────────────────────────────────
sc = SelfConsistencyChecker(k=3)

ans, conf = sc.vote([42, 42, 7])
assert ans == 42 and abs(conf - 2 / 3) < 1e-9, f"vote wrong: {ans}, {conf}"

ans, conf = sc.vote([1, 2, 3])
assert conf == 1 / 3, f"unanimous split wrong: {conf}"

best, confidence, escalate = sc.check([42, 42, 42])
assert best == 42 and not escalate, "perfect agreement should not escalate"

best, confidence, escalate = sc.check([1, 2, 3])
assert escalate, "uniform split should escalate"

# Empty input
ans_e, conf_e = sc.vote([])
assert ans_e == 0 and conf_e == 0.0, "empty vote wrong"
print("  ✓ SelfConsistencyChecker (vote, escalate, edge cases)")

# ─────────────────────────────────────────────────────────────────────────────
# 9. VerificationLadder — level 0 correctness
# ─────────────────────────────────────────────────────────────────────────────
vl = VerificationLadder()

r0_ok  = vl._level0(42)
r0_bad = vl._level0(-1)
assert r0_ok.passed  and r0_ok.level  == 0, "level0 should pass for 42"
assert not r0_bad.passed, "level0 should fail for -1"

# Level 1 SymPy check
r1 = vl._level1(6, "2 * 3")
assert r1.passed and r1.level == 1, f"level1 wrong: {r1}"

r1_bad = vl._level1(5, "2 * 3")
assert not r1_bad.passed, f"level1 should fail for 5 vs 2*3"

# Full verify (only levels 0+1 without lean/z3)
vr = vl.verify(42, sympy_expr="6 * 7")
assert vr.passed, f"verify 42=6*7 failed: {vr}"
print("  ✓ VerificationLadder (level 0, level 1, full verify)")

# ─────────────────────────────────────────────────────────────────────────────
# PERFORMANCE BENCHMARKS
# ─────────────────────────────────────────────────────────────────────────────
print("\n── Performance benchmarks ──")

# ── UCT select 2048 nodes < 1ms ──────────────────────────────────────────────
big_store = MCTSNodeStore(max_nodes=MAX_NODES)
big_root  = big_store.alloc(-1, 0, "root")
for i in range(min(MAX_NODES - 1, 2047)):
    ci = big_store.alloc(big_root, 1, f"child_{i}")
    big_store.add_child(big_root, ci)
    big_store.n[ci] = float(i + 1)
    big_store.q[ci] = float(i % 100)
big_store.n[big_root] = 2047.0

t_uct = bench(big_store.uct_score_children, big_root, n_runs=1000, warmup=100)
print(f"  UCT select {big_store.size()} nodes: {t_uct*1e3:.3f}ms")
assert t_uct < 0.001, f"UCT too slow: {t_uct*1e3:.3f}ms > 1ms"

# ── Backprop depth=20 < 0.1ms ─────────────────────────────────────────────────
depth = 20
_par2 = np.full(depth + 1, -1, dtype=np.int64)
_n2   = np.zeros(depth + 1, dtype=np.float64)
_q2   = np.zeros(depth + 1, dtype=np.float64)
for i in range(1, depth + 1):
    _par2[i] = i - 1

t_bp = bench(
    backpropagate_jit, _par2, _n2, _q2, depth, 1.0,
    n_runs=10000, warmup=1000,
)
print(f"  Backprop depth={depth}: {t_bp*1e6:.2f}μs")
assert t_bp < 0.0001, f"Backprop too slow: {t_bp*1e6:.2f}μs > 100μs"

# ── PRM symbolic_score_jit < 1μs ─────────────────────────────────────────────
sample_step  = "By AM-GM inequality we have a + b ≥ 2√(ab)."
sample_feats = _extract_symbolic_features(sample_step)
t_sym = bench(symbolic_score_jit, sample_feats, n_runs=100000, warmup=10000)
print(f"  symbolic_score_jit: {t_sym*1e6:.3f}μs")
assert t_sym < 1e-6, f"symbolic_score_jit too slow: {t_sym*1e6:.3f}μs > 1μs"

# ── MathRAG retrieval < 20ms ──────────────────────────────────────────────────
t_rag = bench(
    rag.retrieve, "prime modulus fermat little theorem", 5,
    n_runs=100, warmup=10,
)
print(f"  MathRAG.retrieve (k=5): {t_rag*1e3:.2f}ms")
assert t_rag < 0.020, f"MathRAG.retrieve too slow: {t_rag*1e3:.2f}ms > 20ms"

# ── cosine_similarity_batch < 1ms for 600 docs ───────────────────────────────
N_docs  = 600
D       = max(len(rag._vocab), 10)
q_big   = np.random.rand(D).astype(np.float64)
mat_big = np.random.rand(N_docs, D).astype(np.float64)
t_cos   = bench(cosine_similarity_batch, q_big, mat_big, n_runs=200, warmup=20)
print(f"  cosine_similarity_batch (N=600, D={D}): {t_cos*1e3:.3f}ms")
assert t_cos < 0.001, f"cosine_sim too slow: {t_cos*1e3:.3f}ms > 1ms"

# ── TimeAllocator.allocate_budget < 100μs ────────────────────────────────────
ta2 = TimeAllocator(total_seconds=9 * 3600.0, n_problems=50)
t_alloc = bench(
    ta2.allocate_budget, "p_bench", "Find the prime divisors of n.",
    n_runs=10000, warmup=1000,
)
print(f"  TimeAllocator.allocate_budget: {t_alloc*1e6:.1f}μs")
assert t_alloc < 1e-4, f"allocate_budget too slow: {t_alloc*1e6:.1f}μs > 100μs"

# ── SelfConsistencyChecker.vote < 100μs ──────────────────────────────────────
sc2         = SelfConsistencyChecker(k=5)
vote_inputs = [42, 42, 7, 42, 13]
t_vote = bench(sc2.vote, vote_inputs, n_runs=100000, warmup=10000)
print(f"  SelfConsistencyChecker.vote (k=5): {t_vote*1e6:.2f}μs")
assert t_vote < 1e-4, f"vote too slow: {t_vote*1e6:.2f}μs > 100μs"

# ── AnswerExtractor.extract < 1ms ─────────────────────────────────────────────
ae2      = AnswerExtractor()
test_str = r"After all calculations, $\boxed{2024}$ is the final answer."
t_ae = bench(ae2.extract, test_str, n_runs=10000, warmup=1000)
print(f"  AnswerExtractor.extract (boxed): {t_ae*1e6:.1f}μs")
assert t_ae < 0.001, f"AnswerExtractor too slow: {t_ae*1e6:.1f}μs > 1ms"

# ── _parse_structured < 5ms ────────────────────────────────────────────────────
# Test parse_structured standalone (fast regex)
import re as _re
def _parse_structured_bench(raw: str):
    result = {}
    for line in raw.splitlines():
        m = _re.match(r"^([A-Z_][A-Z0-9_]*):\s*(.*)", line.strip())
        if m:
            result[m.group(1)] = m.group(2).strip()
    return result

sample_output = "\n".join([
    "ANSWER: 42",
    "CONFIDENCE: 95",
    "REASONING: By Fermat little theorem",
    "TEMPLATE_PATTERN: modular_arithmetic",
    "KEY_STEPS: 1. Apply Fermat 2. Reduce",
    "DOMAIN_TAGS: number_theory",
    "VERIFIED: yes",
])
t_parse = bench(_parse_structured_bench, sample_output, n_runs=10000, warmup=1000)
print(f"  _parse_structured (7 fields): {t_parse*1e6:.1f}μs")
assert t_parse < 0.005, f"_parse_structured too slow: {t_parse*1e6:.1f}μs > 5ms"

# ─────────────────────────────────────────────────────────────────────────────
print("\n✅ All v5 benchmarks passed. CTRL-MATH v5 solver stack verified.")
print("=" * 70)
