# cell_01_imports.py
import os, sys, re, math, time, json, warnings, tempfile, subprocess
from pathlib import Path
from typing import Optional
from fractions import Fraction
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from functools import lru_cache

import numpy as np
from numpy.fft import fft, ifft
import sympy as sp
from sympy import factorint, isprime, totient, primerange
import networkx as nx

# ── Numba ────────────────────────────────────────────────────────────────────
from numba import njit, prange, cuda, vectorize, int64, float64, boolean
from numba import typed, types
import numba as nb

# ── CuPy ─────────────────────────────────────────────────────────────────────
try:
    import cupy as cp
    HAS_CUPY = True
except ImportError:
    HAS_CUPY = False
    print("[WARN] CuPy not available. GPU vectorization disabled.")

# ── FLINT ─────────────────────────────────────────────────────────────────────
try:
    import flint
    from flint import fmpz, fmpz_poly, fmpq_poly, fmpz_mod_poly
    HAS_FLINT = True
except ImportError:
    HAS_FLINT = False
    print("[WARN] python-flint not available. Falling back to SymPy for polynomials.")

# ── Torch ─────────────────────────────────────────────────────────────────────
import torch
warnings.filterwarnings("ignore")

# ── Hardware inventory ────────────────────────────────────────────────────────
print("=" * 65)
print("CTRL-MATH AIMO3 — H100 High Performance Hardware Inventory")
print("=" * 65)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
if DEVICE != "cuda":
    warnings.warn("H100 GPU not detected — running in CPU-only mode (degraded performance).", RuntimeWarning)
    VRAM_GB = 0.0
    props   = None
else:
    props   = torch.cuda.get_device_properties(0)
    VRAM_GB = props.total_memory / 1e9
N_CORES = os.cpu_count()
print(f"  GPU    : {props.name if props else 'CPU'}  ({VRAM_GB:.1f} GB VRAM)")
print(f"  CPU    : {N_CORES} cores")
print(f"  Numba  : {nb.__version__}  threads={os.environ.get('NUMBA_NUM_THREADS', str(N_CORES))}")
print(f"  CuPy   : {'✅' if HAS_CUPY  else '❌'}")
print(f"  FLINT  : {'✅' if HAS_FLINT else '❌'}")
print("=" * 65)

# ── Global prime sieve (shared memory, used by all JIT functions) ─────────────
SIEVE_LIMIT  = 10_000_000
_sieve       = np.ones(SIEVE_LIMIT + 1, dtype=np.bool_)
_sieve[0]    = _sieve[1] = False
for _i in range(2, int(SIEVE_LIMIT**0.5) + 1):
    if _sieve[_i]:
        _sieve[_i*_i::_i] = False
PRIMES_ARRAY = np.where(_sieve)[0].astype(np.int64)  # all primes < 10^7
print(f"  Prime sieve: {len(PRIMES_ARRAY):,} primes up to {SIEVE_LIMIT:,}")

# ── Precomputed Euler phi table ───────────────────────────────────────────────
PHI_TABLE = np.arange(100_001, dtype=np.int64)
for _p in PRIMES_ARRAY:
    if _p > 100_000: break
    PHI_TABLE[_p::_p] = PHI_TABLE[_p::_p] // _p * (_p - 1)
print(f"  phi(n) table: n ≤ 100,000 precomputed")

# ── AIMO3 Model Configuration ─────────────────────────────────────────────────
MODEL_PRIMARY     = "Qwen/Qwen2.5-Math-14B-Instruct"   # Primary TIR solver
MODEL_ENSEMBLE    = "deepseek-ai/DeepSeek-Math-7B-Instruct"  # Ensemble/backup
MODEL_PRM         = "Qwen/Qwen2.5-Math-1.5B-Instruct"  # Process Reward Model
LORA_ADAPTER_PATH = "/kaggle/working/ctrlmath_aimo3_lora.safetensors"

# ── AIMO3 Dataset Configuration ───────────────────────────────────────────────
DATASET_SFT_PRIMARY  = "nvidia/OpenMathReasoning"   # 540K TIR traces
DATASET_GEOMETRY     = "THUDM/InternMath"            # 20K geometry-heavy
DATASET_AIME_IMO     = "math-ai/imo-aime-problems"   # 5K classic problems
DATASET_AIMO3_VAL    = "/kaggle/input/ai-mathematical-olympiad-progress-prize-3"  # 347 verified
DATASET_SYNTHETIC    = "/kaggle/working/synthetic_aimo3"  # Generated synthetic data
