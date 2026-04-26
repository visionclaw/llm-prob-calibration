#!/usr/bin/env python3
"""
Main experiment loop for LLM probability calibration.

Runs both Condition A (P-first) and Condition B (C-first) on every question,
extracting full logit distributions over integers 0-100.

Results are saved incrementally to allow resumption after interruption.

Usage:
    python src/run_experiment.py --model Qwen/Qwen2.5-7B-Instruct --output results/
    python src/run_experiment.py --model Qwen/Qwen2.5-7B-Instruct --output results/ --resume
    python src/run_experiment.py --model Qwen/Qwen2.5-7B-Instruct --output results/ --limit 10
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

# Add parent dir to path
sys.path.insert(0, str(Path(__file__).parent))
from logit_extractor import LogitExtractor, compute_entropy, kl_divergence, count_modes


def load_model(model_name: str, device: str = "cuda", dtype=None):
    """Load model and tokenizer from HuggingFace."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from huggingface_hub import login

    # Try to login with HF token
    hf_token_path = "/mnt/workspace/.hf/token"
    if os.path.exists(hf_token_path):
        token = open(hf_token_path).read().strip()
        login(token=token)
        print(f"[+] Logged in to HuggingFace")

    if dtype is None:
        dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    print(f"[+] Loading tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    print(f"[+] Loading model: {model_name} (dtype={dtype})")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        device_map="auto" if device == "cuda" else None,
        trust_remote_code=True,
    )
    if device != "cuda" or "device_map" not in str(model.hf_device_map if hasattr(model, 'hf_device_map') else ""):
        model = model.to(device)
    model.eval()

    print(f"[+] Model loaded on {device}")
    return model, tokenizer


def load_questions(path: str) -> list[dict]:
    """Load questions from JSON file."""
    with open(path) as f:
        return json.load(f)


def load_completed_ids(results_dir: str) -> set:
    """Load IDs of already-completed questions for resume support."""
    completed = set()
    results_file = Path(results_dir) / "results.jsonl"
    if results_file.exists():
        with open(results_file) as f:
            for line in f:
                try:
                    entry = json.loads(line.strip())
                    completed.add(entry["id"])
                except (json.JSONDecodeError, KeyError):
                    continue
    return completed


def save_result(results_dir: str, result: dict):
    """Append a single result to the JSONL file."""
    results_file = Path(results_dir) / "results.jsonl"
    with open(results_file, "a") as f:
        # Convert numpy arrays to lists for JSON serialization
        serializable = {}
        for k, v in result.items():
            if isinstance(v, np.ndarray):
                serializable[k] = v.tolist()
            elif isinstance(v, (np.integer, np.floating)):
                serializable[k] = v.item()
            else:
                serializable[k] = v
        f.write(json.dumps(serializable) + "\n")


def process_question(
    extractor: LogitExtractor,
    question: dict,
    temperature: float = 1.0,
) -> dict:
    """Run both conditions on a single question and compute all metrics.

    Returns a dict with all extracted distributions and computed metrics.
    """
    q_text = question["question"]
    q_id = question["id"]
    ground_truth = question.get("ground_truth")

    # Condition A: P-first
    result_a = extractor.extract_full_response(
        q_text, condition="P_first", temperature=temperature
    )

    # Condition B: C-first
    result_b = extractor.extract_full_response(
        q_text, condition="C_first", temperature=temperature
    )

    # Compute metrics
    p_dist_a = result_a["p_dist"]
    c_dist_a = result_a["c_dist"]
    p_dist_b = result_b["p_dist"]
    c_dist_b = result_b["c_dist"]

    p_entropy_a = compute_entropy(p_dist_a)
    c_entropy_a = compute_entropy(c_dist_a)
    p_entropy_b = compute_entropy(p_dist_b)
    c_entropy_b = compute_entropy(c_dist_b)

    p_argmax_a = int(np.argmax(p_dist_a))
    c_argmax_a = int(np.argmax(c_dist_a))
    p_argmax_b = int(np.argmax(p_dist_b))
    c_argmax_b = int(np.argmax(c_dist_b))

    n_modes_p_a = count_modes(p_dist_a)
    n_modes_p_b = count_modes(p_dist_b)
    n_modes_c_a = count_modes(c_dist_a)
    n_modes_c_b = count_modes(c_dist_b)

    # Order effects: KL divergence between conditions
    order_effect_p = kl_divergence(p_dist_a, p_dist_b)
    order_effect_c = kl_divergence(c_dist_a, c_dist_b)

    result = {
        "id": q_id,
        "tier": question["tier"],
        "category": question.get("category", "unknown"),
        "question": q_text,
        "ground_truth": ground_truth,
        "ground_truth_confidence": question.get("ground_truth_confidence"),
        # Condition A (P-first)
        "p_dist_A": p_dist_a,
        "c_dist_A": c_dist_a,
        "p_argmax_A": p_argmax_a,
        "c_argmax_A": c_argmax_a,
        "p_entropy_A": p_entropy_a,
        "c_entropy_A": c_entropy_a,
        "n_modes_p_A": n_modes_p_a,
        "n_modes_c_A": n_modes_c_a,
        # Condition B (C-first)
        "p_dist_B": p_dist_b,
        "c_dist_B": c_dist_b,
        "p_argmax_B": p_argmax_b,
        "c_argmax_B": c_argmax_b,
        "p_entropy_B": p_entropy_b,
        "c_entropy_B": c_entropy_b,
        "n_modes_p_B": n_modes_p_b,
        "n_modes_c_B": n_modes_c_b,
        # Order effects
        "order_effect_p": order_effect_p,
        "order_effect_c": order_effect_c,
    }

    # Error metrics (only for Tier 1 & 2 with ground truth)
    if ground_truth is not None and question["tier"] in [1, 2]:
        result["p_error_A"] = abs(p_argmax_a - ground_truth)
        result["p_error_B"] = abs(p_argmax_b - ground_truth)

    return result


def main():
    parser = argparse.ArgumentParser(description="LLM Probability Calibration Experiment")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-7B-Instruct",
                        help="HuggingFace model name")
    parser.add_argument("--output", type=str, default="results/",
                        help="Output directory for results")
    parser.add_argument("--questions", type=str, default=None,
                        help="Path to questions.json (default: data/questions.json)")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from existing results")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit number of questions to process")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device (cuda, cpu)")
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="Softmax temperature for logit extraction")
    parser.add_argument("--dtype", type=str, default="float16",
                        choices=["float16", "bfloat16", "float32"],
                        help="Model dtype")
    args = parser.parse_args()

    # Resolve paths
    project_root = Path(__file__).parent.parent
    questions_path = args.questions or str(project_root / "data" / "questions.json")
    output_dir = args.output

    # Create output dir
    os.makedirs(output_dir, exist_ok=True)

    # Save experiment config
    config = {
        "model": args.model,
        "temperature": args.temperature,
        "dtype": args.dtype,
        "device": args.device,
        "questions_path": questions_path,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(Path(output_dir) / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    # Load questions
    print(f"[+] Loading questions from {questions_path}")
    questions = load_questions(questions_path)
    print(f"[+] Loaded {len(questions)} questions")

    # Resume support
    completed_ids = set()
    if args.resume:
        completed_ids = load_completed_ids(output_dir)
        print(f"[+] Resuming: {len(completed_ids)} questions already completed")

    # Filter to remaining questions
    remaining = [q for q in questions if q["id"] not in completed_ids]
    if args.limit:
        remaining = remaining[:args.limit]
    print(f"[+] Processing {len(remaining)} questions")

    if len(remaining) == 0:
        print("[+] All questions already processed!")
        return

    # Load model
    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    model, tokenizer = load_model(
        args.model,
        device=args.device,
        dtype=dtype_map[args.dtype],
    )

    # Create extractor
    extractor = LogitExtractor(model, tokenizer, device=args.device)

    # Main loop
    print(f"\n[+] Starting experiment: {args.model}")
    print(f"[+] Conditions: P-first (A) and C-first (B)")
    print(f"[+] Temperature: {args.temperature}")
    print()

    start_time = time.time()
    errors = []

    for i, question in enumerate(tqdm(remaining, desc="Processing questions")):
        try:
            result = process_question(
                extractor, question, temperature=args.temperature
            )
            save_result(output_dir, result)

            # Progress logging every 50 questions
            if (i + 1) % 50 == 0:
                elapsed = time.time() - start_time
                rate = (i + 1) / elapsed
                eta = (len(remaining) - i - 1) / rate if rate > 0 else 0
                print(
                    f"\n[Progress] {i+1}/{len(remaining)} | "
                    f"Rate: {rate:.1f} q/s | "
                    f"ETA: {eta/60:.1f} min"
                )

        except Exception as e:
            error_msg = f"Error on question {question['id']}: {e}"
            print(f"\n[!] {error_msg}")
            errors.append({"id": question["id"], "error": str(e)})
            continue

    elapsed = time.time() - start_time
    print(f"\n[+] Experiment complete!")
    print(f"[+] Processed: {len(remaining) - len(errors)}/{len(remaining)} questions")
    print(f"[+] Errors: {len(errors)}")
    print(f"[+] Time: {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print(f"[+] Results saved to: {output_dir}")

    # Save error log
    if errors:
        with open(Path(output_dir) / "errors.json", "w") as f:
            json.dump(errors, f, indent=2)
        print(f"[+] Error log saved to {output_dir}/errors.json")

    # Save summary
    total_completed = len(completed_ids) + len(remaining) - len(errors)
    summary = {
        "model": args.model,
        "total_questions": len(questions),
        "completed": total_completed,
        "errors": len(errors),
        "elapsed_seconds": elapsed,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(Path(output_dir) / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
