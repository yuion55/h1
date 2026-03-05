# CTRL-MATH AIMO3 — Competition Math Solver

**Competition**: [AI Mathematical Olympiad Progress Prize 3 (Kaggle)](https://www.kaggle.com/competitions/ai-mathematical-olympiad-progress-prize-3)

CTRL-MATH AIMO3 is a high-performance competition math solver that combines large language models, Monte Carlo Tree Search, process reward models, symbolic verification, and retrieval-augmented generation to solve Olympiad-style integer-answer problems under strict time constraints.

---

## Models

| Role | Model | Details |
|------|-------|---------|
| **Primary solver** | `Qwen/Qwen2.5-Math-14B-Instruct` | 4-bit NF4 quantized, Flash Attention 2 |
| **Ensemble model** | `deepseek-ai/DeepSeek-Math-7B-Instruct` | Secondary solver for cross-verification |
| **PRM model** | `Qwen/Qwen2.5-Math-1.5B-Instruct` | Process Reward Model for step-level scoring |
| **LoRA adapter** | `ctrlmath_aimo3_lora` | Custom fine-tuned adapter via SFT + GRPO training |

---

## Dependencies / Wheels

The following packages must be available as offline wheels for Kaggle submission (internet is disabled during competition rerun):

### Core scientific
- `numpy>=1.26`
- `scipy>=1.12`
- `numba>=0.59.0`

### GPU
- `cupy-cuda12x`

### Number theory
- `python-flint`

### LLM
- `torch` (with CUDA)
- `transformers>=4.40.0`
- `accelerate>=0.27.0`
- `bitsandbytes>=0.43.0`
- `sentencepiece`
- `peft>=0.10.0`
- `trl>=0.8.6`
- `datasets>=2.19.0`
- `flash-attn>=2.5.8`

### Symbolic / Verification
- `sympy>=1.12`
- `z3-solver`
- `networkx>=3.0`

### Retrieval
- `sentence-transformers>=2.7.0`

### Data
- `polars`
- `kaggle`

---

## Architecture

Every source file and its purpose, in pipeline order:

| File | Purpose |
|------|---------|
| `cell_00_install.py` | Package installation (pip installs all deps, sets up Lean 4 binary, configures Numba threads) |
| `cell_01_imports.py` | Global imports, hardware detection (GPU/VRAM), prime sieve initialization (primes < 10^7) |
| `cell_02a_numba_nt.py` | Numba JIT number theory: p-adic valuations, Fibonacci, NTT polynomial multiplication, Dirichlet convolution, modular arithmetic batches |
| `cell_02b_cupy_gpu.py` | CuPy GPU-accelerated arithmetic for H100: batch modular exponentiation, GPU sieve, polynomial ops |
| `cell_02c_flint_poly.py` | FLINT polynomial library bindings for fast polynomial operations |
| `cell_03_mog_parser.py` | MOG (Mathematics Object Graph) parser: classifies problems into domains (number theory, combinatorics, algebra, geometry) using keyword scoring |
| `cell_04_transform_engine.py` | Transform engine: applies mathematical transformations (substitutions, simplifications) to problem states |
| `cell_04a_extractor.py` | Extraction utilities for pulling structured data from problem text |
| `cell_04b_linear_recurrence.py` | Linear recurrence solver using Berlekamp-Massey + matrix exponentiation |
| `cell_04c_combinatorics.py` | Combinatorics solver: counting, partitions, generating functions |
| `cell_04d_number_theory.py` | Number theory solver: modular arithmetic, CRT, Euler's theorem |
| `cell_04e_gf_solver.py` | Generating function solver for sequence problems |
| `cell_04f_geometry.py` | Computational geometry solver |
| `cell_04g_geometry_prover.py` | Geometric theorem prover using coordinate geometry and algebraic methods |
| `cell_05_cyclotomic.py` | Cyclotomic polynomial utilities (**standalone**, not in main pipeline — see note in file) |
| `cell_06_mcts.py` | Monte Carlo Tree Search engine with flat NumPy node storage and @njit backpropagation |
| `cell_06_mpc_planner.py` | MPC (Model Predictive Control) planner: routes problems to geometry or LLM solver, majority voting |
| `cell_07_llm_executor_v5.py` | LLM executor: loads Qwen2.5-Math-14B with 4-bit NF4 quantization, Flash Attention 2, optional LoRA adapter, DeepSeek ensemble; provides solve/verify/decompose/compress prompt templates |
| `cell_08_prm.py` | Process Reward Model: combines symbolic scoring (Numba JIT, 5 features) with LLM scoring (Qwen-1.5B); combined score = 0.6×llm + 0.4×symbolic |
| `cell_08_kalman.py` | Kalman belief filter for tracking solve confidence across verification stages |
| `cell_09_mathrag.py` | MathRAG: retrieval-augmented generation using sentence-transformer embeddings for similar problem lookup |
| `cell_09a_lean4_repl.py` | Lean 4 REPL interface for formal proof verification |
| `cell_09b_z3_parallel.py` | Parallel Z3 SMT solver for constraint verification |
| `cell_10_template_store.py` | Template store: saves/retrieves reusable solution templates from successful solves |
| `cell_10_norwegian.py` | Problem-specific solver for Norwegian competition number theory problems (**standalone** — see note in file) |
| `cell_11_answer_extractor.py` | Answer normalization: extracts integer answers from LLM output via boxed/ANSWER/fraction/power/factorial/keyword/last-integer fallback chain |
| `cell_12_time_allocator.py` | Competition time budget management: difficulty estimation, per-problem allocation, 10% retry reserve, 3× hard cutoff |
| `cell_13_self_consistency.py` | Self-consistency checker: vectorized majority vote (np.unique + np.argmax), escalation from K=5 to more rollouts if confidence < 0.5 |
| `cell_14_verification_ladder.py` | 4-level hierarchical verification: L0 range check → L1 SymPy → L2 Z3 → L3 Lean 4 proof |
| `cell_15_orchestrator_v5.py` | Top-level orchestrator: 4-phase solve loop (shortcuts → MCTS → verify+Lean → template save), wires all components, never raises |
| `cell_16_benchmarks_v5.py` | Performance benchmarks and ablation tests |
| `cell_17_aimo3_training.py` | Training pipeline: SFT (Qwen14B + OpenMathReasoning + InternMath) → PRM → GRPO refinement → evaluation on AIMO3 Val (347 problems) |
| `cell_21_synthetic.py` | Synthetic data generation for GRPO training |
| `test_reference.py` | Validation against `reference.csv` (11 known problems with ground truth answers) |
| `ctrlmath_aimo3_submission.ipynb` | Kaggle gateway submission notebook (gRPC InferenceServer pattern) |
| `ctrlmath_aimo3_final.ipynb` | Alternative submission notebook (direct CSV generation pattern) |

---

## Submission Pipeline

The `predict()` function in `ctrlmath_aimo3_submission.ipynb` implements the following flow:

1. Gateway sends a single-row `polars.DataFrame` with `id` and `problem` columns.
2. `predict()` extracts `problem_id` and `problem_text` from the row.
3. `orchestrator.solve_problem()` runs the **4-phase solve loop**:
   - **Phase 1 — Shortcuts**: template match or direct extraction from problem text.
   - **Phase 2 — MCTS**: K=5 independent rollouts → self-consistency vote.
   - **Phase 3 — Verify + Lean**: up to 3 retries with Lean 4 correction.
   - **Phase 4 — Template save**: persist successful solution for future retrieval.
4. Raw answer is normalized: `answer = max(0, int(answer) % 100_000)`.
5. Returns `pl.DataFrame({"answer": [answer]})`.

---

## Kaggle Deployment

1. Upload this repository as a Kaggle dataset named **`h1-main`**.
2. Upload `Qwen/Qwen2.5-Math-14B-Instruct` model weights as a Kaggle dataset named **`qwen2.5-math-14b-instruct`**.
3. Upload `deepseek-ai/DeepSeek-Math-7B-Instruct` model weights as a Kaggle dataset named **`deepseek-math-7b-instruct`**.
4. *(Optional)* Upload pre-built wheel files as **`ctrlmath-wheels`**.
5. *(Optional)* Upload the trained LoRA adapter as **`ctrlmath-aimo3-lora`**.
6. Attach all datasets to the submission notebook in Kaggle.
7. Submit **`ctrlmath_aimo3_submission.ipynb`** as the competition notebook.

---

## Local Testing

Run validation against the 11 reference problems with known answers:

```bash
python test_reference.py --csv reference.csv --limit 3
```

---

## Training

Train the full pipeline (SFT → PRM → GRPO → Eval):

```bash
python cell_17_aimo3_training.py
```

Training phases:
1. **SFT** — Qwen2.5-Math-14B fine-tuned on OpenMathReasoning + InternMath datasets.
2. **PRM** — Process Reward Model trained on step-level annotations.
3. **GRPO** — Group Relative Policy Optimization for answer-correctness reward.
4. **Eval** — Benchmarked on AIMO3 validation set (347 problems).

---

## Project Structure

```
h1/
├── cell_00_install.py             # Dependency installation & setup
├── cell_01_imports.py             # Global imports & hardware detection
├── cell_02a_numba_nt.py           # Numba JIT number theory
├── cell_02b_cupy_gpu.py           # CuPy GPU arithmetic
├── cell_02c_flint_poly.py         # FLINT polynomial bindings
├── cell_03_mog_parser.py          # Problem domain classifier
├── cell_04_transform_engine.py    # Mathematical transform engine
├── cell_04a_extractor.py          # Structured data extraction
├── cell_04b_linear_recurrence.py  # Linear recurrence solver
├── cell_04c_combinatorics.py      # Combinatorics solver
├── cell_04d_number_theory.py      # Number theory solver
├── cell_04e_gf_solver.py          # Generating function solver
├── cell_04f_geometry.py           # Computational geometry
├── cell_04g_geometry_prover.py    # Geometric theorem prover
├── cell_05_cyclotomic.py          # Cyclotomic tools (standalone)
├── cell_06_mcts.py                # MCTS engine
├── cell_06_mpc_planner.py         # MPC problem router
├── cell_07_llm_executor_v5.py     # LLM executor (Qwen + DeepSeek)
├── cell_08_prm.py                 # Process Reward Model
├── cell_08_kalman.py              # Kalman belief filter
├── cell_09_mathrag.py             # MathRAG retrieval
├── cell_09a_lean4_repl.py         # Lean 4 REPL interface
├── cell_09b_z3_parallel.py        # Parallel Z3 SMT solver
├── cell_10_template_store.py      # Solution template store
├── cell_10_norwegian.py           # Norwegian competition solver (standalone)
├── cell_11_answer_extractor.py    # Answer normalization
├── cell_12_time_allocator.py      # Time budget manager
├── cell_13_self_consistency.py    # Self-consistency checker
├── cell_14_verification_ladder.py # 4-level verification ladder
├── cell_15_orchestrator_v5.py     # Top-level solve orchestrator
├── cell_16_benchmarks_v5.py       # Performance benchmarks
├── cell_17_aimo3_training.py      # Training pipeline
├── cell_21_synthetic.py           # Synthetic data generation
├── test_reference.py              # Reference validation tests
├── ctrlmath_aimo3_submission.ipynb # Kaggle submission notebook
├── ctrlmath_aimo3_final.ipynb     # Alternative submission notebook
└── README.md                      # This file
```
