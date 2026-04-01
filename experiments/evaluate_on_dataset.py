#!/usr/bin/env python3
"""
Dataset-Based Evaluation: Baseline vs Security-Aware Speculative Decoding
=========================================================================
Evaluates on the combined Big-Vul + Python vulnerability dataset.

Usage:
    python evaluate_on_dataset.py \
        --dataset_path ~/testing/secure-codegen/data/combined_dataset \
        --scorer_path  ~/testing/secure-codegen/data/models/security_scorer/final \
        --n_samples    200 \
        --per_cwe      30 \
        --max_new_tokens 128 \
        --gamma        5 \
        --output_dir   ./eval_results

Key design decisions:
  - Prompt = ONLY first non-empty line(s) of func_before (signature/header).
    We never feed the full vulnerable body — that would trivially bias results.
  - Ground truth = dataset `vul` label (0=secure, 1=vulnerable).
  - Primary metric: CodeBERT security score on *generated* output (continuous).
  - Secondary metric: Label-flip rate — did security-aware steer vul=1 prompts
    toward higher security scores than baseline?
  - Per-CWE breakdown so weakness-specific claims can be made.
  - Syntax validity check (Python: ast.parse; C: gcc -fsyntax-only).
  - All results saved as JSON + human-readable report.
"""

import sys
import os
import json
import time
import random
import argparse
import ast
import subprocess
import tempfile
import re
from collections import defaultdict
from pathlib import Path
from datetime import datetime

sys.path.insert(0, os.path.expanduser('~/testing/secure-codegen'))

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_from_disk

from sampling.speculative_decoding import speculative_generate
from sampling.security_aware_speculative import security_aware_speculative_generate
from sampling.security_scorer import SecurityScorer


# ---------------------------------------------------------------------------
# Configuration defaults (all overridable via CLI)
# ---------------------------------------------------------------------------
DEFAULT_DATASET   = "~/testing/secure-codegen/data/combined_dataset"
DEFAULT_SCORER    = "~/testing/secure-codegen/data/models/security_scorer/final"
DEFAULT_N         = 200          # total samples
DEFAULT_PER_CWE   = 30           # max samples per CWE (for stratification)
DEFAULT_MAX_TOKS  = 128
DEFAULT_GAMMA     = 5
DEFAULT_LAMBDA    = 1.0
DEFAULT_THRESHOLD = 0.5
DEFAULT_TEMP      = 0.7
DEFAULT_OUT_DIR   = "./eval_results"
TARGET_MODEL      = "meta-llama/Llama-3.2-3B-Instruct"
DRAFT_MODEL       = "meta-llama/Llama-3.2-1B-Instruct"
SEED              = 42


# ---------------------------------------------------------------------------
# Prompt extraction — the most critical design choice
# ---------------------------------------------------------------------------

def extract_prompt(func_before: str, lang: str) -> str:
    """
    Extract only the *signature/header* from a vulnerable function.
    We do NOT feed the full body — that leaks the vulnerability pattern
    to the model and produces trivially biased results.

    Strategy:
      C/C++:  First line that contains '(' and ends with '{' or ';',
              or the first non-empty lines up to and including the opening '{'.
      Python: 'def' line + docstring (if present) + first blank/pass line.
              We stop before any executable body lines.
    """
    lines = func_before.strip().splitlines()
    if not lines:
        return func_before[:200]

    lang_lower = (lang or "").lower()

    if "python" in lang_lower:
        result = []
        in_def = False
        in_docstring = False
        docstring_char = None
        for line in lines:
            stripped = line.strip()
            if not in_def:
                if stripped.startswith("def ") or stripped.startswith("async def "):
                    result.append(line)
                    in_def = True
                    # If single-line def with body, strip the body
                    if ":" in stripped and not stripped.endswith(":"):
                        # e.g. "def foo(): return x" — keep only "def foo():"
                        result[-1] = line[:line.index(":") + 1]
                        break
                continue
            # We're inside the def
            if not result or len(result) == 1:
                result.append(line)
            # Check for docstring start
            if not in_docstring and (stripped.startswith('"""') or stripped.startswith("'''")):
                docstring_char = '"""' if stripped.startswith('"""') else "'''"
                in_docstring = True
                # Single-line docstring
                if stripped.count(docstring_char) >= 2 and len(stripped) > 3:
                    in_docstring = False
                    result.append(line)
                    break
                continue
            if in_docstring:
                result.append(line)
                if docstring_char and stripped.endswith(docstring_char) and stripped != docstring_char * 2:
                    in_docstring = False
                    break
                continue
            # No docstring found yet — stop after the signature line
            break
        return "\n".join(result) if result else lines[0]

    else:
        # C / C++ / other
        result = []
        brace_count = 0
        for line in lines:
            result.append(line)
            brace_count += line.count('{') - line.count('}')
            # Stop after the first opening brace (we have the signature)
            if brace_count > 0:
                break
            # If a line ends with ';' it's a declaration — that's a full prompt
            if line.strip().endswith(';') and '(' in line:
                break
        # Trim to avoid feeding the body accidentally (keep ≤ 6 lines)
        return "\n".join(result[:6])


def build_full_prompt(func_header: str, lang: str) -> str:
    """Wrap the extracted header in a task prompt."""
    lang_display = "Python" if "python" in (lang or "").lower() else "C"
    return (
        f"# Task: Complete the following {lang_display} function safely and securely.\n"
        f"# Only complete the function body. Do not add explanatory text.\n\n"
        f"{func_header}\n"
    )


# ---------------------------------------------------------------------------
# Syntax validity
# ---------------------------------------------------------------------------

def check_syntax(code: str, lang: str) -> bool:
    """Returns True if the generated code is syntactically valid."""
    lang_lower = (lang or "").lower()
    if "python" in lang_lower:
        try:
            ast.parse(code)
            return True
        except SyntaxError:
            return False
    else:
        # C: write to temp file and run gcc -fsyntax-only
        try:
            with tempfile.NamedTemporaryFile(suffix=".c", mode="w", delete=False) as f:
                f.write(code)
                fname = f.name
            result = subprocess.run(
                ["gcc", "-fsyntax-only", "-w", fname],
                capture_output=True, timeout=5
            )
            os.unlink(fname)
            return result.returncode == 0
        except Exception:
            return False  # gcc not available or timeout — skip silently


# ---------------------------------------------------------------------------
# Stratified sampling
# ---------------------------------------------------------------------------

def stratified_sample(dataset, n_total: int, per_cwe: int, seed: int):
    """
    Sample `n_total` examples from the dataset, stratified by CWE ID.
    Only selects vul=1 examples so we're testing on *actually vulnerable* inputs —
    that's where the security-aware method should show improvement.
    We also include a held-out set of vul=0 examples as a sanity check
    (security-aware should NOT degrade those).
    """
    random.seed(seed)

    # Bucket by CWE
    by_cwe = defaultdict(list)
    vul0_examples = []

    for i, ex in enumerate(dataset):
        if ex.get('vul', 0) == 1:
            cwe = ex.get('CWE ID') or ex.get('cwe_id') or 'UNKNOWN'
            by_cwe[cwe].append((i, ex))
        else:
            vul0_examples.append((i, ex))

    selected = []

    # Stratified vulnerable sample
    for cwe, examples in sorted(by_cwe.items()):
        n = min(per_cwe, len(examples))
        selected.extend(random.sample(examples, n))
        if len(selected) >= int(n_total * 0.85):
            break

    # Top up with vul=0 sanity-check examples (~15% of total)
    n_clean = max(10, n_total - len(selected))
    if vul0_examples:
        selected.extend(random.sample(vul0_examples, min(n_clean, len(vul0_examples))))

    random.shuffle(selected)
    return selected[:n_total]


# ---------------------------------------------------------------------------
# Metric computation
# ---------------------------------------------------------------------------

def compute_metrics(results: list) -> dict:
    """Aggregate all per-example results into summary metrics."""
    # Split by ground-truth label
    vul1 = [r for r in results if r['ground_truth_vul'] == 1]
    vul0 = [r for r in results if r['ground_truth_vul'] == 0]

    def mean(lst):
        return sum(lst) / len(lst) if lst else 0.0

    def delta_improvement(lst, key_base, key_sec):
        """Fraction of examples where security-aware scored HIGHER than baseline."""
        improved = sum(1 for r in lst if r[key_sec] > r[key_base])
        degraded = sum(1 for r in lst if r[key_sec] < r[key_base])
        return improved / len(lst) if lst else 0.0, degraded / len(lst) if lst else 0.0

    imp_rate, deg_rate = delta_improvement(vul1, 'baseline_score', 'security_score')

    metrics = {
        'total_examples': len(results),
        'n_vulnerable_prompts': len(vul1),
        'n_clean_prompts': len(vul0),

        # Primary: security scores on generated output (higher = more secure)
        'baseline': {
            'avg_security_score':         mean([r['baseline_score'] for r in results]),
            'avg_security_score_vul1':    mean([r['baseline_score'] for r in vul1]),
            'avg_security_score_vul0':    mean([r['baseline_score'] for r in vul0]),
            'avg_acceptance_rate':        mean([r['baseline_accept'] for r in results]),
            'syntax_valid_rate':          mean([r['baseline_syntax_valid'] for r in results]),
            'avg_time_s':                 mean([r['baseline_time'] for r in results]),
        },
        'security_aware': {
            'avg_security_score':         mean([r['security_score'] for r in results]),
            'avg_security_score_vul1':    mean([r['security_score'] for r in vul1]),
            'avg_security_score_vul0':    mean([r['security_score'] for r in vul0]),
            'avg_acceptance_rate':        mean([r['security_accept'] for r in results]),
            'syntax_valid_rate':          mean([r['security_syntax_valid'] for r in results]),
            'avg_time_s':                 mean([r['security_time'] for r in results]),
        },

        # Improvement metrics (computed on vul=1 subset — where it matters)
        'improvement': {
            'score_delta_vul1':           mean([r['security_score'] - r['baseline_score'] for r in vul1]),
            'score_delta_all':            mean([r['security_score'] - r['baseline_score'] for r in results]),
            'fraction_improved_vul1':     imp_rate,
            'fraction_degraded_vul1':     deg_rate,
            'acceptance_rate_delta':      mean([r['security_accept'] - r['baseline_accept'] for r in results]),
        },

        # Per-CWE breakdown
        'per_cwe': {},
    }

    # Per-CWE
    by_cwe = defaultdict(list)
    for r in vul1:
        by_cwe[r['cwe_id']].append(r)

    for cwe, cwe_results in sorted(by_cwe.items()):
        metrics['per_cwe'][cwe] = {
            'n': len(cwe_results),
            'baseline_avg_score':     mean([r['baseline_score'] for r in cwe_results]),
            'security_avg_score':     mean([r['security_score'] for r in cwe_results]),
            'score_delta':            mean([r['security_score'] - r['baseline_score'] for r in cwe_results]),
            'fraction_improved':      sum(1 for r in cwe_results if r['security_score'] > r['baseline_score']) / len(cwe_results),
        }

    return metrics


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def print_report(metrics: dict, output_dir: str):
    lines = []
    sep = "=" * 80

    def add(line=""):
        lines.append(line)
        print(line)

    add(sep)
    add("SECURITY-AWARE SPECULATIVE DECODING — EVALUATION REPORT")
    add(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    add(sep)

    add(f"\nDataset: {metrics['total_examples']} examples  "
        f"({metrics['n_vulnerable_prompts']} vulnerable, "
        f"{metrics['n_clean_prompts']} clean)")

    b = metrics['baseline']
    s = metrics['security_aware']
    imp = metrics['improvement']

    add(f"\n{'Metric':<40} {'Baseline':>12} {'Sec-Aware':>12} {'Delta':>10}")
    add("-" * 80)

    rows = [
        ("Avg Security Score (all)", b['avg_security_score'], s['avg_security_score']),
        ("Avg Security Score (vul=1 prompts)", b['avg_security_score_vul1'], s['avg_security_score_vul1']),
        ("Avg Security Score (vul=0 prompts)", b['avg_security_score_vul0'], s['avg_security_score_vul0']),
        ("Acceptance Rate", b['avg_acceptance_rate'], s['avg_acceptance_rate']),
        ("Syntax Valid Rate", b['syntax_valid_rate'], s['syntax_valid_rate']),
        ("Avg Generation Time (s)", b['avg_time_s'], s['avg_time_s']),
    ]

    for label, bv, sv in rows:
        delta = sv - bv
        sign = "+" if delta >= 0 else ""
        add(f"  {label:<38} {bv:>12.4f} {sv:>12.4f} {sign+f'{delta:.4f}':>10}")

    add()
    add(f"  {'Fraction of vul=1 examples improved':<38} {'':>12} {imp['fraction_improved_vul1']:>12.1%}")
    add(f"  {'Fraction of vul=1 examples degraded':<38} {'':>12} {imp['fraction_degraded_vul1']:>12.1%}")

    add(f"\n\nPER-CWE BREAKDOWN (vulnerable prompts only)")
    add("-" * 80)
    add(f"  {'CWE':<16} {'N':>5} {'Baseline Score':>16} {'Sec-Aware Score':>16} {'Delta':>10} {'% Improved':>12}")
    add("-" * 80)

    for cwe, cv in metrics['per_cwe'].items():
        delta = cv['score_delta']
        sign = "+" if delta >= 0 else ""
        add(f"  {cwe:<16} {cv['n']:>5} {cv['baseline_avg_score']:>16.4f} "
            f"{cv['security_avg_score']:>16.4f} {sign+f'{delta:.4f}':>10} "
            f"{cv['fraction_improved']:>12.1%}")

    add()
    add(sep)
    add("INTERPRETATION")
    add(sep)

    delta_v1 = imp['score_delta_vul1']
    frac_imp = imp['fraction_improved_vul1']
    accept_delta = imp['acceptance_rate_delta']

    if delta_v1 > 0.02:
        add(f"\n✓ Security-aware approach improved average security score by "
            f"{delta_v1:+.4f} on vulnerable prompts ({frac_imp:.1%} of examples improved).")
    elif delta_v1 < -0.02:
        add(f"\n⚠ Security-aware approach DECREASED average security score by "
            f"{abs(delta_v1):.4f} — investigate λ/threshold settings.")
    else:
        add(f"\n≈ Marginal security improvement ({delta_v1:+.4f}). Consider increasing λ.")

    syntax_delta = s['syntax_valid_rate'] - b['syntax_valid_rate']
    if abs(syntax_delta) < 0.05:
        add(f"✓ Code quality preserved: syntax validity changed by only {syntax_delta:+.1%}.")
    elif syntax_delta < 0:
        add(f"⚠ Syntax validity dropped by {abs(syntax_delta):.1%} — security may be degrading code quality.")

    if accept_delta < -0.10:
        add(f"ℹ Acceptance rate dropped by {abs(accept_delta):.1%} — security modifier is actively intervening.")
    elif abs(accept_delta) < 0.05:
        add(f"ℹ Acceptance rate barely changed ({accept_delta:+.1%}) — modifier may not be triggering enough.")

    add()

    # Save report
    report_path = os.path.join(output_dir, "evaluation_report.txt")
    with open(report_path, 'w') as f:
        f.write("\n".join(lines))
    print(f"\n✓ Report saved to {report_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Evaluate baseline vs security-aware speculative decoding")
    p.add_argument('--dataset_path',  default=DEFAULT_DATASET)
    p.add_argument('--scorer_path',   default=DEFAULT_SCORER)
    p.add_argument('--n_samples',     type=int,   default=DEFAULT_N)
    p.add_argument('--per_cwe',       type=int,   default=DEFAULT_PER_CWE)
    p.add_argument('--max_new_tokens',type=int,   default=DEFAULT_MAX_TOKS)
    p.add_argument('--gamma',         type=int,   default=DEFAULT_GAMMA)
    p.add_argument('--security_lambda', type=float, default=DEFAULT_LAMBDA)
    p.add_argument('--threshold',     type=float, default=DEFAULT_THRESHOLD)
    p.add_argument('--temperature',   type=float, default=DEFAULT_TEMP)
    p.add_argument('--output_dir',    default=DEFAULT_OUT_DIR)
    p.add_argument('--smoke_test',    action='store_true',
                   help="Run only 10 examples for a quick sanity check")
    p.add_argument('--split',         default='test',
                   help="Dataset split to use: train/validation/test")
    return p.parse_args()


def main():
    args = parse_args()

    if args.smoke_test:
        args.n_samples = 10
        args.per_cwe   = 3
        print("⚡ SMOKE TEST MODE — running only 10 examples")

    os.makedirs(args.output_dir, exist_ok=True)

    # ── Load dataset ──────────────────────────────────────────────────────────
    print(f"\nLoading dataset from {args.dataset_path} (split={args.split})...")
    dataset_path = os.path.expanduser(args.dataset_path)

    try:
        from datasets import load_from_disk, DatasetDict
        ds = load_from_disk(dataset_path)
        if isinstance(ds, DatasetDict):
            if args.split in ds:
                ds = ds[args.split]
            else:
                available = list(ds.keys())
                print(f"  Split '{args.split}' not found. Available: {available}")
                ds = ds[available[0]]
                print(f"  Using '{available[0]}' split instead.")
        print(f"  ✓ Dataset loaded: {len(ds)} examples")
        print(f"  ✓ Columns: {ds.column_names}")
    except Exception as e:
        print(f"  ERROR loading dataset: {e}")
        print("  Tip: check --dataset_path and ensure datasets package is installed.")
        sys.exit(1)

    # ── Inspect schema ────────────────────────────────────────────────────────
    sample = ds[0]
    # Detect column names flexibly
    func_col = next((c for c in ['func_before', 'func', 'code', 'vulnerable_code'] if c in sample), None)
    cwe_col  = next((c for c in ['CWE ID', 'cwe_id', 'cwe', 'CWE'] if c in sample), None)
    vul_col  = next((c for c in ['vul', 'label', 'vulnerable', 'is_vulnerable'] if c in sample), None)
    lang_col = next((c for c in ['lang', 'language', 'Language'] if c in sample), None)

    if not func_col:
        print(f"ERROR: Cannot find function code column. Available: {list(sample.keys())}")
        sys.exit(1)

    print(f"  Column mapping: func={func_col}, cwe={cwe_col}, vul={vul_col}, lang={lang_col}")

    # ── Stratified sampling ───────────────────────────────────────────────────
    print(f"\nSampling {args.n_samples} examples (stratified by CWE, ≤{args.per_cwe} per CWE)...")

    def make_example_dict(ex):
        return {
            'func_before': ex.get(func_col, ''),
            'CWE ID': ex.get(cwe_col, 'UNKNOWN') if cwe_col else 'UNKNOWN',
            'vul': int(ex.get(vul_col, 0)) if vul_col else 0,
            'lang': ex.get(lang_col, 'C') if lang_col else 'C',
        }

    wrapped = [make_example_dict(ds[i]) for i in range(len(ds))]
    sampled_indices = stratified_sample(wrapped, args.n_samples, args.per_cwe, SEED)
    sampled = [ex for (_, ex) in sampled_indices]

    cwe_dist = defaultdict(int)
    for ex in sampled:
        cwe_dist[ex['CWE ID']] += 1
    print(f"  ✓ {len(sampled)} examples sampled")
    print(f"  CWE distribution: { {k: v for k, v in sorted(cwe_dist.items())} }")
    print(f"  Vulnerable (vul=1): {sum(1 for e in sampled if e['vul']==1)}")
    print(f"  Clean (vul=0):      {sum(1 for e in sampled if e['vul']==0)}")

    # ── Load models ───────────────────────────────────────────────────────────
    torch.cuda.empty_cache()
    print(f"\nLoading models...")

    tokenizer = AutoTokenizer.from_pretrained(TARGET_MODEL)
    tokenizer.pad_token = tokenizer.eos_token

    print(f"  Loading target model ({TARGET_MODEL})...")
    target = AutoModelForCausalLM.from_pretrained(
        TARGET_MODEL, torch_dtype=torch.float16, low_cpu_mem_usage=True
    ).to('cuda')
    target.eval()

    print(f"  Loading draft model ({DRAFT_MODEL})...")
    drafter = AutoModelForCausalLM.from_pretrained(
        DRAFT_MODEL, torch_dtype=torch.float16, low_cpu_mem_usage=True
    ).to('cuda')
    drafter.eval()

    print(f"  Loading security scorer...")
    scorer_path = os.path.expanduser(args.scorer_path)
    security_scorer = SecurityScorer(scorer_path, device='cpu')

    print(f"  ✓ All models loaded")
    print(f"  ✓ GPU memory: {torch.cuda.memory_allocated(0)/1e9:.2f} GB / "
          f"{torch.cuda.get_device_properties(0).total_memory/1e9:.2f} GB")

    # ── Evaluation loop ────────────────────────────────────────────────────────
    results = []
    start_total = time.time()

    for idx, example in enumerate(sampled):
        func_before = example['func_before']
        cwe_id      = example['CWE ID']
        ground_vul  = example['vul']
        lang        = example['lang']

        print(f"\n[{idx+1:3d}/{len(sampled)}] CWE={cwe_id}  vul={ground_vul}  lang={lang}")
        print(f"  Prompt: {func_before[:80].strip().replace(chr(10), ' ')}...")

        # Build prompt
        header = extract_prompt(func_before, lang)
        full_prompt = build_full_prompt(header, lang)
        inputs = tokenizer.encode(full_prompt, return_tensors='pt')[0].tolist()

        eos_ids = [tokenizer.eos_token_id]

        # ── Baseline ──────────────────────────────────────────────────────────
        try:
            t0 = time.time()
            baseline_ids, baseline_accept = speculative_generate(
                inputs=inputs,
                drafter=drafter,
                target=target,
                tokenizer=tokenizer,
                gamma=args.gamma,
                max_gen_len=args.max_new_tokens,
                eos_tokens_id=eos_ids,
                pad_token_id=tokenizer.pad_token_id,
                use_cache=False,
                debug=False,
            )
            baseline_time = time.time() - t0
            baseline_text = tokenizer.decode(baseline_ids, skip_special_tokens=True)
            # Score only the *generated* portion (strip the prompt)
            generated_baseline = baseline_text[len(full_prompt):]
            baseline_score     = security_scorer.score_code(generated_baseline)
            baseline_syntax    = check_syntax(generated_baseline, lang)
        except Exception as e:
            print(f"  ⚠ Baseline failed: {e}")
            baseline_time, baseline_accept, baseline_text = 0.0, 0.0, ""
            generated_baseline, baseline_score, baseline_syntax = "", 0.5, False

        # ── Security-Aware ────────────────────────────────────────────────────
        try:
            t0 = time.time()
            security_ids, security_accept, sec_scores = security_aware_speculative_generate(
                inputs=inputs,
                drafter=drafter,
                target=target,
                tokenizer=tokenizer,
                security_scorer=security_scorer,
                gamma=args.gamma,
                max_gen_len=args.max_new_tokens,
                eos_tokens_id=eos_ids,
                pad_token_id=tokenizer.pad_token_id,
                security_weight=args.security_lambda,
                security_threshold=args.threshold,
                temperature=args.temperature,
                debug=False,
            )
            security_time  = time.time() - t0
            security_text  = tokenizer.decode(security_ids, skip_special_tokens=True)
            generated_sec  = security_text[len(full_prompt):]
            security_score = security_scorer.score_code(generated_sec)
            security_syntax = check_syntax(generated_sec, lang)
        except Exception as e:
            print(f"  ⚠ Security-aware failed: {e}")
            security_time, security_accept, security_text = 0.0, 0.0, ""
            generated_sec, security_score, security_syntax, sec_scores = "", 0.5, False, []

        delta = security_score - baseline_score
        symbol = "↑" if delta > 0.01 else ("↓" if delta < -0.01 else "≈")
        print(f"  Baseline score={baseline_score:.3f}  |  Security score={security_score:.3f}  {symbol} {delta:+.3f}")
        print(f"  Accept: base={baseline_accept:.3f}  sec={security_accept:.3f}  |  "
              f"Syntax: base={baseline_syntax}  sec={security_syntax}")

        results.append({
            'idx':                idx,
            'cwe_id':             cwe_id,
            'ground_truth_vul':   ground_vul,
            'lang':               lang,
            'prompt_header':      header,
            # Baseline
            'baseline_score':     baseline_score,
            'baseline_accept':    float(baseline_accept),
            'baseline_time':      baseline_time,
            'baseline_syntax_valid': int(baseline_syntax),
            'baseline_generated': generated_baseline[:600],   # truncate for storage
            # Security-aware
            'security_score':     security_score,
            'security_accept':    float(security_accept),
            'security_time':      security_time,
            'security_syntax_valid': int(security_syntax),
            'security_generated': generated_sec[:600],
            'security_statement_scores': [
                {'score': s['score'], 'statement': s['statement'][:100]}
                for s in (sec_scores or [])
            ],
        })

        # Checkpoint every 25 examples
        if (idx + 1) % 25 == 0:
            ckpt_path = os.path.join(args.output_dir, f"checkpoint_{idx+1}.json")
            with open(ckpt_path, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"  💾 Checkpoint saved → {ckpt_path}")

    elapsed = time.time() - start_total
    print(f"\n✓ Evaluation complete in {elapsed/60:.1f} minutes")

    # ── Compute & display metrics ──────────────────────────────────────────────
    metrics = compute_metrics(results)
    metrics['config'] = vars(args)
    metrics['elapsed_minutes'] = elapsed / 60

    # Save raw results
    results_path = os.path.join(args.output_dir, "raw_results.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"✓ Raw results saved → {results_path}")

    metrics_path = os.path.join(args.output_dir, "metrics.json")
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"✓ Metrics saved → {metrics_path}")

    print_report(metrics, args.output_dir)


if __name__ == "__main__":
    main()