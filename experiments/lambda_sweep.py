#!/usr/bin/env python3
"""
Lambda sweep runner — no SLURM needed.

Runs evaluate_on_dataset.py sequentially for each lambda value and saves
results to separate output directories so they can be compared directly.

Usage:
    python lambda_sweep.py                        # full 200-example run
    python lambda_sweep.py --smoke_test           # 10 examples per lambda (fast sanity check)
    python lambda_sweep.py --lambdas 1.5 2.0 3.0 # custom lambda subset
"""

import subprocess
import sys
import os
import json
import argparse
from pathlib import Path

LAMBDAS_DEFAULT = [0.5, 1.0, 1.5, 2.0, 3.0, 5.0]

EVAL_SCRIPT = os.path.join(os.path.dirname(__file__), "evaluate_on_dataset.py")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--lambdas",        type=float, nargs="+", default=LAMBDAS_DEFAULT,
                   help="Lambda values to sweep (default: 0.5 1.0 1.5 2.0 3.0 5.0)")
    p.add_argument("--dataset_path",   default="~/testing/secure-codegen/data/combined_dataset")
    p.add_argument("--scorer_path",    default="~/testing/secure-codegen/data/models/security_scorer/final")
    p.add_argument("--n_samples",      type=int, default=200)
    p.add_argument("--per_cwe",        type=int, default=30)
    p.add_argument("--max_new_tokens", type=int, default=128)
    p.add_argument("--gamma",          type=int, default=5)
    p.add_argument("--threshold",      type=float, default=0.5)
    p.add_argument("--temperature",    type=float, default=0.7)
    p.add_argument("--output_base",    default="./eval_results_sweep",
                   help="Parent directory; each lambda gets a subdirectory inside")
    p.add_argument("--smoke_test",     action="store_true",
                   help="Run only 10 examples per lambda (fast sanity check)")
    return p.parse_args()


def run_lambda(lam: float, args) -> dict | None:
    """Run evaluation for a single lambda value. Returns parsed metrics or None on failure."""
    tag = str(lam).replace(".", "_")
    output_dir = os.path.join(args.output_base, f"lambda_{tag}")
    os.makedirs(output_dir, exist_ok=True)

    cmd = [
        sys.executable, EVAL_SCRIPT,
        "--dataset_path",   args.dataset_path,
        "--scorer_path",    args.scorer_path,
        "--n_samples",      str(args.n_samples),
        "--per_cwe",        str(args.per_cwe),
        "--max_new_tokens", str(args.max_new_tokens),
        "--gamma",          str(args.gamma),
        "--security_lambda", str(lam),
        "--threshold",      str(args.threshold),
        "--temperature",    str(args.temperature),
        "--output_dir",     output_dir,
    ]
    if args.smoke_test:
        cmd.append("--smoke_test")

    print(f"\n{'='*60}")
    print(f"  λ = {lam}  →  {output_dir}")
    print(f"{'='*60}")

    result = subprocess.run(cmd, text=True)

    if result.returncode != 0:
        print(f"  ✗ λ={lam} failed (exit code {result.returncode})")
        return None

    # Load metrics for the summary table
    metrics_path = os.path.join(output_dir, "metrics.json")
    if os.path.exists(metrics_path):
        with open(metrics_path) as f:
            return json.load(f)
    return None


def print_summary(results: list[tuple[float, dict]]):
    """Print a compact comparison table across all lambda values."""
    print(f"\n{'='*80}")
    print("LAMBDA SWEEP SUMMARY")
    print(f"{'='*80}")
    print(f"  {'λ':>6}  {'Avg Score (vul=1)':>20}  {'Delta':>8}  {'% Improved':>12}  {'Accept Δ':>10}")
    print(f"  {'-'*6}  {'-'*20}  {'-'*8}  {'-'*12}  {'-'*10}")

    for lam, m in results:
        if m is None:
            print(f"  {lam:>6.1f}  {'FAILED':>20}")
            continue
        b_score = m['baseline']['avg_security_score_vul1']
        s_score = m['security_aware']['avg_security_score_vul1']
        delta   = m['improvement']['score_delta_vul1']
        frac    = m['improvement']['fraction_improved_vul1']
        adelta  = m['improvement']['acceptance_rate_delta']
        sign    = "+" if delta >= 0 else ""
        print(f"  {lam:>6.1f}  {b_score:>8.4f} → {s_score:>8.4f}  "
              f"{sign+f'{delta:.4f}':>8}  {frac:>12.1%}  {adelta:>+10.4f}")

    print(f"{'='*80}\n")


def main():
    args = parse_args()
    os.makedirs(args.output_base, exist_ok=True)

    mode = "SMOKE TEST (10 examples)" if args.smoke_test else f"{args.n_samples} examples"
    print(f"\nLambda sweep — {mode}")
    print(f"Lambdas: {args.lambdas}")
    print(f"Output:  {args.output_base}/")

    sweep_results = []
    for lam in args.lambdas:
        metrics = run_lambda(lam, args)
        sweep_results.append((lam, metrics))

    print_summary(sweep_results)

    # Save the summary as JSON too
    summary_path = os.path.join(args.output_base, "sweep_summary.json")
    summary = {
        str(lam): m for lam, m in sweep_results if m is not None
    }
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary saved → {summary_path}")


if __name__ == "__main__":
    main()
