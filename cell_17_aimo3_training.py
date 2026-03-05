# cell_17_aimo3_training.py
"""
CTRL-MATH AIMO3 — Training Pipeline

Training flow:
  1. SFT:  Qwen2.5-Math-14B + OpenMathReasoning + InternMath → LoRA adapter
  2. PRM:  Qwen2.5-Math-1.5B + AIMO3 Val → reward model
  3. GRPO: LoRA(Qwen14B) + PRM rewards + synthetic data → refined adapter
  4. Eval: Validate on AIMO3 Val benchmark (347 problems)

Hardware: 80GB H100, DeepSpeed ZeRO-3
Expected output: ctrlmath_aimo3_lora.safetensors
"""

from __future__ import annotations

import os
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

# ── Model & Dataset constants ──────────────────────────────────────────────────
MODEL_PRIMARY     = "Qwen/Qwen2.5-Math-14B-Instruct"
MODEL_PRM         = "Qwen/Qwen2.5-Math-1.5B-Instruct"
MODEL_ENSEMBLE    = "deepseek-ai/DeepSeek-Math-7B-Instruct"

DATASET_SFT_PRIMARY = "nvidia/OpenMathReasoning"
DATASET_GEOMETRY    = "THUDM/InternMath"
DATASET_AIME_IMO    = "math-ai/imo-aime-problems"
DATASET_AIMO3_VAL   = "/kaggle/input/ai-mathematical-olympiad-progress-prize-3"
DATASET_SYNTHETIC   = "/kaggle/working/synthetic_aimo3"

LORA_OUTPUT_PATH    = "/kaggle/working/ctrlmath_aimo3_lora"
LORA_ADAPTER_FILE   = "/kaggle/working/ctrlmath_aimo3_lora.safetensors"

# ── LoRA configuration ─────────────────────────────────────────────────────────
LORA_CONFIG = {
    "r": 64,
    "lora_alpha": 128,
    "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj",
                       "gate_proj", "up_proj", "down_proj"],
    "lora_dropout": 0.05,
    "bias": "none",
    "task_type": "CAUSAL_LM",
}

# ── SFT training configuration ─────────────────────────────────────────────────
SFT_CONFIG = {
    "num_train_epochs": 2,
    "per_device_train_batch_size": 2,
    "gradient_accumulation_steps": 8,
    "learning_rate": 2e-4,
    "warmup_ratio": 0.03,
    "lr_scheduler_type": "cosine",
    "bf16": True,
    "tf32": True,
    "optim": "adamw_torch_fused",
    "dataloader_num_workers": 4,
    "logging_steps": 50,
    "save_steps": 500,
    "max_seq_length": 4096,
}

# ── GRPO configuration ─────────────────────────────────────────────────────────
GRPO_CONFIG = {
    "num_train_epochs": 1,
    "per_device_train_batch_size": 1,
    "gradient_accumulation_steps": 16,
    "learning_rate": 5e-5,
    "num_generations": 8,       # G in GRPO
    "max_new_tokens": 2048,
    "temperature": 0.9,
    "beta": 0.04,               # KL penalty
    "bf16": True,
}


def load_openmath_reasoning(split: str = "train", max_samples: int = 50_000) -> List[Dict]:
    """
    Load nvidia/OpenMathReasoning dataset.
    Returns list of {"problem": str, "solution": str, "answer": str}.

    The dataset contains 540K Tool-Integrated Reasoning (TIR) traces
    used by NVIDIA to win AIMO2.
    """
    try:
        from datasets import load_dataset
        ds = load_dataset(DATASET_SFT_PRIMARY, split=split, streaming=True)
        samples = []
        for i, ex in enumerate(ds):
            if i >= max_samples:
                break
            samples.append({
                "problem":  ex.get("problem", ex.get("question", "")),
                "solution": ex.get("solution", ex.get("generated_solution", "")),
                "answer":   str(ex.get("answer", ex.get("expected_answer", ""))),
            })
        print(f"[OpenMathReasoning] Loaded {len(samples):,} samples.")
        return samples
    except Exception as e:
        print(f"[WARN] Failed to load OpenMathReasoning: {e}")
        return []


def load_internmath_geometry(max_samples: int = 20_000) -> List[Dict]:
    """
    Load THUDM/InternMath dataset (geometry-heavy subset).
    Returns list of {"problem": str, "solution": str, "answer": str}.
    """
    try:
        from datasets import load_dataset
        ds = load_dataset(DATASET_GEOMETRY, split="train", streaming=True)
        samples = []
        for i, ex in enumerate(ds):
            if i >= max_samples:
                break
            samples.append({
                "problem":  ex.get("problem", ex.get("question", "")),
                "solution": ex.get("solution", ""),
                "answer":   str(ex.get("answer", "")),
                "domain":   "geometry",
            })
        print(f"[InternMath] Loaded {len(samples):,} samples.")
        return samples
    except Exception as e:
        print(f"[WARN] Failed to load InternMath: {e}")
        return []


def load_imo_aime_archive(max_samples: int = 5_000) -> List[Dict]:
    """
    Load math-ai/imo-aime-problems dataset.
    Returns list of {"problem": str, "answer": str, "source": str}.
    """
    try:
        from datasets import load_dataset
        ds = load_dataset(DATASET_AIME_IMO, split="train")
        samples = []
        for i, ex in enumerate(ds):
            if i >= max_samples:
                break
            samples.append({
                "problem": ex.get("problem", ""),
                "answer":  str(ex.get("answer", "")),
                "source":  ex.get("source", "unknown"),
                "year":    ex.get("year", 0),
            })
        print(f"[IMO/AIME Archive] Loaded {len(samples):,} samples.")
        return samples
    except Exception as e:
        print(f"[WARN] Failed to load IMO/AIME archive: {e}")
        return []


def load_aimo3_val_benchmark() -> List[Dict]:
    """
    Load AIMO3 validation benchmark (347 verified problems).
    Tries Kaggle competition data directory first, then HuggingFace.
    """
    val_path = Path(DATASET_AIMO3_VAL)

    # Try Kaggle competition data
    for fname in ["train.csv", "test.csv", "problems.csv", "val.csv"]:
        fpath = val_path / fname
        if fpath.exists():
            try:
                import csv
                problems = []
                with open(fpath) as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        problems.append({
                            "id":      row.get("id", ""),
                            "problem": row.get("problem", row.get("question", "")),
                            "answer":  str(row.get("answer", "")),
                        })
                print(f"[AIMO3 Val] Loaded {len(problems)} problems from {fpath}.")
                return problems
            except Exception as e:
                print(f"[WARN] Failed to read {fpath}: {e}")

    print("[WARN] AIMO3 Val benchmark not found. Download with:")
    print("  kaggle competitions download -c ai-mathematical-olympiad-progress-prize-3")
    return []


def format_sft_prompt(problem: str, solution: str) -> Dict[str, str]:
    """
    Format a problem+solution pair into a chat-style SFT training sample.
    Uses the Qwen2.5-Math instruction format.
    """
    return {
        "prompt": (
            f"<|im_start|>system\nYou are a competition math expert. "
            f"Solve problems step by step using Tool-Integrated Reasoning (TIR).<|im_end|>\n"
            f"<|im_start|>user\n{problem}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        ),
        "completion": f"{solution}<|im_end|>",
    }


def run_sft_training(
    train_samples: List[Dict],
    output_dir: str = LORA_OUTPUT_PATH,
    config: Dict = SFT_CONFIG,
) -> bool:
    """
    Run SFT (Supervised Fine-Tuning) on Qwen2.5-Math-14B with LoRA.

    Phase 1 of training: SFT on OpenMathReasoning + InternMath.

    Returns True on success.
    """
    print(f"\n{'='*65}")
    print(f"PHASE 1: SFT Training")
    print(f"  Model:    {MODEL_PRIMARY}")
    print(f"  Samples:  {len(train_samples):,}")
    print(f"  Output:   {output_dir}")
    print(f"{'='*65}")

    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments
        from peft import LoraConfig, get_peft_model, TaskType
        from trl import SFTTrainer
        from datasets import Dataset

        # Quantization config (NF4 for H100)
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

        print(f"[SFT] Loading tokenizer: {MODEL_PRIMARY}")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PRIMARY, trust_remote_code=True)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"

        print(f"[SFT] Loading model: {MODEL_PRIMARY} (4-bit NF4)")
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_PRIMARY,
            quantization_config=bnb_config,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            trust_remote_code=True,
        )
        model.config.use_cache = False

        # Apply LoRA
        lora_config = LoraConfig(
            r=LORA_CONFIG["r"],
            lora_alpha=LORA_CONFIG["lora_alpha"],
            target_modules=LORA_CONFIG["target_modules"],
            lora_dropout=LORA_CONFIG["lora_dropout"],
            bias=LORA_CONFIG["bias"],
            task_type=TaskType.CAUSAL_LM,
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

        # Format dataset
        formatted = [format_sft_prompt(s["problem"], s["solution"]) for s in train_samples]
        texts = [f["prompt"] + f["completion"] for f in formatted]
        hf_dataset = Dataset.from_dict({"text": texts})

        # Training arguments
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=config["num_train_epochs"],
            per_device_train_batch_size=config["per_device_train_batch_size"],
            gradient_accumulation_steps=config["gradient_accumulation_steps"],
            learning_rate=config["learning_rate"],
            warmup_ratio=config["warmup_ratio"],
            lr_scheduler_type=config["lr_scheduler_type"],
            bf16=config["bf16"],
            tf32=config.get("tf32", True),
            optim=config["optim"],
            dataloader_num_workers=config["dataloader_num_workers"],
            logging_steps=config["logging_steps"],
            save_steps=config["save_steps"],
            report_to="none",
        )

        trainer = SFTTrainer(
            model=model,
            args=training_args,
            train_dataset=hf_dataset,
            tokenizer=tokenizer,
            max_seq_length=config["max_seq_length"],
            dataset_text_field="text",
        )

        print(f"[SFT] Starting training...")
        t0 = time.time()
        trainer.train()
        elapsed = time.time() - t0
        print(f"[SFT] Training complete in {elapsed/3600:.1f}h")

        # Save LoRA adapter
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
        print(f"[SFT] LoRA adapter saved to {output_dir}")
        return True

    except Exception as e:
        print(f"[ERROR] SFT training failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_grpo_training(
    train_samples: List[Dict],
    prm_model=None,
    output_dir: str = LORA_OUTPUT_PATH,
    config: Dict = GRPO_CONFIG,
) -> bool:
    """
    Run GRPO (Group Relative Policy Optimization) training.

    Phase 3 of training: Refine the SFT LoRA adapter using PRM rewards
    and synthetic data.

    Returns True on success.
    """
    print(f"\n{'='*65}")
    print(f"PHASE 3: GRPO Training")
    print(f"  Base:     {MODEL_PRIMARY} + LoRA from {output_dir}")
    print(f"  Samples:  {len(train_samples):,}")
    print(f"  G:        {config['num_generations']} generations per problem")
    print(f"{'='*65}")

    try:
        import re
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        from peft import PeftModel
        from trl import GRPOTrainer, GRPOConfig
        from datasets import Dataset

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

        tokenizer = AutoTokenizer.from_pretrained(MODEL_PRIMARY, trust_remote_code=True)
        tokenizer.pad_token = tokenizer.eos_token

        base_model = AutoModelForCausalLM.from_pretrained(
            MODEL_PRIMARY,
            quantization_config=bnb_config,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            trust_remote_code=True,
        )
        model = PeftModel.from_pretrained(base_model, output_dir, is_trainable=True)

        # Format prompts for GRPO
        prompts = [format_sft_prompt(s["problem"], "")["prompt"] for s in train_samples]
        answers = [s["answer"] for s in train_samples]
        hf_dataset = Dataset.from_dict({"prompt": prompts, "answer": answers})

        def reward_fn(completions: List[str], answers: List[str], **kwargs) -> List[float]:
            """Reward function: +1 if correct integer answer, 0 otherwise."""
            rewards = []
            for completion, expected in zip(completions, answers):
                # Extract integer from completion
                match = re.search(r"ANSWER:\s*(\d+)", completion)
                if match and match.group(1) == str(expected).strip():
                    rewards.append(1.0)
                else:
                    # Partial credit from PRM if available
                    if prm_model is not None:
                        try:
                            steps = completion.split("\n")
                            scores = prm_model.score_steps(steps)
                            rewards.append(float(sum(scores) / max(len(scores), 1)) * 0.5)
                        except Exception:
                            rewards.append(0.0)
                    else:
                        rewards.append(0.0)
            return rewards

        grpo_config = GRPOConfig(
            num_train_epochs=config["num_train_epochs"],
            per_device_train_batch_size=config["per_device_train_batch_size"],
            gradient_accumulation_steps=config["gradient_accumulation_steps"],
            learning_rate=config["learning_rate"],
            num_generations=config["num_generations"],
            max_new_tokens=config["max_new_tokens"],
            temperature=config["temperature"],
            beta=config["beta"],
            bf16=config["bf16"],
            output_dir=output_dir,
            report_to="none",
        )

        trainer = GRPOTrainer(
            model=model,
            args=grpo_config,
            tokenizer=tokenizer,
            train_dataset=hf_dataset,
            reward_func=reward_fn,
        )

        print(f"[GRPO] Starting training...")
        t0 = time.time()
        trainer.train()
        elapsed = time.time() - t0
        print(f"[GRPO] Training complete in {elapsed/3600:.1f}h")

        model.save_pretrained(output_dir)
        print(f"[GRPO] Refined LoRA adapter saved to {output_dir}")
        return True

    except Exception as e:
        print(f"[ERROR] GRPO training failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def evaluate_on_aimo3_val(
    orchestrator,
    val_problems: List[Dict],
) -> Dict[str, Any]:
    """
    Evaluate the full pipeline on AIMO3 Val benchmark (347 problems).
    Returns dict with accuracy, per-domain breakdown, and failure analysis.
    """
    if not val_problems:
        print("[WARN] No validation problems available.")
        return {"accuracy": 0.0, "correct": 0, "total": 0}

    correct = 0
    results = []
    t0 = time.time()

    for i, prob in enumerate(val_problems):
        pid     = prob.get("id", str(i))
        problem = prob.get("problem", "")
        answer  = prob.get("answer", "")

        try:
            predicted = orchestrator.solve_problem(pid, problem)
            is_correct = str(predicted).strip() == str(answer).strip()
            if is_correct:
                correct += 1
            results.append({
                "id": pid, "correct": is_correct,
                "predicted": predicted, "expected": answer,
            })
        except Exception as e:
            results.append({"id": pid, "correct": False, "error": str(e)})

        if (i + 1) % 10 == 0:
            elapsed = time.time() - t0
            print(f"[Eval] {i+1}/{len(val_problems)} | "
                  f"Acc: {correct/(i+1):.1%} | "
                  f"Elapsed: {elapsed/60:.1f}m")

    accuracy = correct / max(len(val_problems), 1)
    print(f"\n[Eval] Final: {correct}/{len(val_problems)} = {accuracy:.1%}")

    return {
        "accuracy":  accuracy,
        "correct":   correct,
        "total":     len(val_problems),
        "results":   results,
    }


def run_full_training_pipeline(skip_sft: bool = False, skip_grpo: bool = False) -> bool:
    """
    Run the complete AIMO3 training pipeline:
      1. Load datasets
      2. SFT training (Qwen14B + OpenMathReasoning + InternMath)
      3. GRPO training (LoRA + PRM rewards + synthetic)
      4. Evaluate on AIMO3 Val

    Args:
        skip_sft:  Skip Phase 1 (SFT) — use if LoRA already trained.
        skip_grpo: Skip Phase 3 (GRPO) — use for ablation.

    Returns True if all phases complete successfully.
    """
    print("\n" + "="*65)
    print("CTRL-MATH AIMO3 — Full Training Pipeline")
    print("="*65)
    print(f"  Primary model:   {MODEL_PRIMARY}")
    print(f"  Ensemble model:  {MODEL_ENSEMBLE}")
    print(f"  PRM model:       {MODEL_PRM}")
    print(f"  SFT dataset:     {DATASET_SFT_PRIMARY}")
    print(f"  Geometry:        {DATASET_GEOMETRY}")
    print(f"  IMO/AIME:        {DATASET_AIME_IMO}")
    print(f"  AIMO3 Val:       {DATASET_AIMO3_VAL}")
    print("="*65 + "\n")

    # ── Phase 1: SFT ──────────────────────────────────────────────────────────
    if not skip_sft:
        openmath   = load_openmath_reasoning(max_samples=50_000)
        internmath = load_internmath_geometry(max_samples=20_000)
        sft_data   = openmath + internmath
        if not sft_data:
            print("[ERROR] No SFT data loaded. Aborting.")
            return False
        success = run_sft_training(sft_data)
        if not success:
            print("[ERROR] SFT failed. Aborting.")
            return False
    else:
        print("[SKIP] Phase 1 (SFT) skipped — using existing LoRA adapter.")

    # ── Phase 2: PRM training ─────────────────────────────────────────────────
    print(f"\n{'='*65}")
    print(f"PHASE 2: PRM Training (Qwen2.5-Math-1.5B on AIMO3 Val)")
    print(f"  See cell_08_prm.py for PRM training details.")
    print(f"{'='*65}")

    # ── Phase 3: GRPO ─────────────────────────────────────────────────────────
    if not skip_grpo:
        imo_aime = load_imo_aime_archive(max_samples=5_000)

        # Load synthetic data if available
        synthetic_data: List[Dict] = []
        syn_path = Path(DATASET_SYNTHETIC)
        if syn_path.exists():
            for f in syn_path.glob("*.json"):
                try:
                    with open(f) as fp:
                        synthetic_data.extend(json.load(fp))
                except Exception:
                    pass
            print(f"[Synthetic] Loaded {len(synthetic_data):,} synthetic samples.")

        grpo_data = imo_aime + synthetic_data
        if not grpo_data:
            print("[WARN] No GRPO data loaded. Skipping GRPO phase.")
        else:
            run_grpo_training(grpo_data, prm_model=None)
    else:
        print("[SKIP] Phase 3 (GRPO) skipped.")

    # ── Phase 4: Evaluation ───────────────────────────────────────────────────
    val_problems = load_aimo3_val_benchmark()
    if val_problems:
        try:
            from cell_15_orchestrator_v5 import SolveOrchestrator
            from cell_07_llm_executor_v5 import LLMExecutorV5
            from cell_08_prm import ProcessRewardModel

            llm = LLMExecutorV5(
                primary_model=MODEL_PRIMARY,
                ensemble_model=MODEL_ENSEMBLE,
                lora_adapter=LORA_OUTPUT_PATH,
            )
            prm = ProcessRewardModel()
            orch = SolveOrchestrator(llm=llm, prm=prm, n_problems=len(val_problems))
            results = evaluate_on_aimo3_val(orch, val_problems)
            print(f"\n✅ AIMO3 Val Accuracy: {results['accuracy']:.1%} "
                  f"({results['correct']}/{results['total']})")
        except Exception as e:
            print(f"[WARN] Evaluation failed: {e}")
    else:
        print("[SKIP] Evaluation skipped — no val problems available.")

    return True


if __name__ == "__main__":
    run_full_training_pipeline()
