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
pip("cupy-cuda12x")            # CuPy for H100 (CUDA 12)

# Fast polynomial / number theory
pip("python-flint")            # Python bindings to FLINT (fast number theory)

# LLM
pip("transformers>=4.40.0")
pip("accelerate>=0.27.0")
pip("bitsandbytes>=0.43.0")
pip("sentencepiece")
pip("peft>=0.10.0")            # LoRA adapter loading
pip("trl>=0.8.6")              # GRPO training
pip("datasets>=2.19.0")        # HuggingFace dataset loading
pip("flash-attn>=2.5.8")       # Flash Attention 2 for H100
pip("kaggle")                  # AIMO3 Val dataset download

# Symbolic / verification
pip("sympy>=1.12")
pip("z3-solver")
pip("networkx>=3.0")

# Sentence transformers / retrieval
pip("sentence-transformers>=2.7.0")
pip("lean4-client")

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


def download_models_and_datasets():
    """Download all AIMO3 models and datasets from HuggingFace."""
    import subprocess

    models = [
        "Qwen/Qwen2.5-Math-14B-Instruct",
        "Qwen/Qwen2.5-Math-1.5B-Instruct",
        "deepseek-ai/DeepSeek-Math-7B-Instruct",
    ]
    datasets = [
        "nvidia/OpenMathReasoning",
        "THUDM/InternMath",
        "math-ai/imo-aime-problems",
    ]

    for model in models:
        print(f"[Download] Model: {model}")
        result = subprocess.run(
            ["huggingface-cli", "download", model],
            capture_output=True, text=True
        )
        if result.returncode == 0:
            print(f"  ✅ {model}")
        else:
            print(f"  ⚠️  {model}: {result.stderr.strip()[:100]}")

    for ds in datasets:
        print(f"[Download] Dataset: {ds}")
        result = subprocess.run(
            ["huggingface-cli", "download", "--repo-type", "dataset", ds],
            capture_output=True, text=True
        )
        if result.returncode == 0:
            print(f"  ✅ {ds}")
        else:
            print(f"  ⚠️  {ds}: {result.stderr.strip()[:100]}")

    print("\n[AIMO3 Val] Download competition data with:")
    print("  kaggle competitions download -c ai-mathematical-olympiad-progress-prize-3")
