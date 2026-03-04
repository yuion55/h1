# cell_00_install.py
"""
Install all performance-critical packages.
Order matters: Numba before CuPy, FLINT before SymPy fallback.
"""
import subprocess, sys, os

def pip(pkg):
    result = subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q", pkg],
        stdout=subprocess.DEVNULL, stderr=subprocess.PIPE
    )
    if result.returncode != 0:
        print(f"[WARN] pip install {pkg!r} failed: {result.stderr.decode().strip()}")

# Core scientific
pip("numpy>=1.26")
pip("scipy>=1.12")
pip("numba>=0.59.0")           # JIT compiler for CPU

# GPU array computing
pip("cupy-cuda12x")            # CuPy for T4 (CUDA 12)

# Fast polynomial / number theory
pip("python-flint")            # Python bindings to FLINT (fast number theory)

# LLM
pip("transformers>=4.40.0")
pip("accelerate>=0.27.0")
pip("bitsandbytes>=0.43.0")
pip("sentencepiece")

# Symbolic / verification
pip("sympy>=1.12")
pip("z3-solver")
pip("networkx>=3.0")

# Lean 4 (from Kaggle dataset cache)
LEAN_BIN = "/kaggle/input/lean4-mathlib-cache/lean/bin/lean"
if not os.path.exists(LEAN_BIN):
    subprocess.run([
        "curl", "-sL",
        "https://github.com/leanprover/lean4/releases/download/v4.8.0/lean-4.8.0-linux.tar.gz",
        "-o", "/tmp/lean.tar.gz"], check=True)
    subprocess.run(["tar", "-xzf", "/tmp/lean.tar.gz", "-C", "/tmp/"], check=True)
    LEAN_BIN = "/tmp/lean-4.8.0/bin/lean"

os.environ["PATH"] = os.path.dirname(LEAN_BIN) + ":" + os.environ["PATH"]

# Numba threading — use all available cores
os.environ["NUMBA_NUM_THREADS"] = str(os.cpu_count())
os.environ["NUMBA_CACHE_DIR"]   = "/kaggle/working/.numba_cache"

print("✅ All packages installed.")
