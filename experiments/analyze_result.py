#!/usr/bin/env python3
"""
analyze_results.py — Post-hoc analysis & visualization of evaluation results.

Run AFTER evaluate_on_dataset.py has produced eval_results/raw_results.json.

Generates:
  - per_cwe_bar.png      : Bar chart of score delta per CWE
  - score_distribution.png : Overlapping histograms baseline vs sec-aware
  - acceptance_scatter.png : Acceptance rate vs security score improvement
  - paper_table.txt       : LaTeX-ready results table
  - summary.txt           : Plain-English findings paragraph

Usage:
    python analyze_results.py --results_dir ./eval_results
"""

import json
import argparse
import os
from collections import defaultdict


def load(results_dir):
    with open(os.path.join(results_dir, "raw_results.json")) as f:
        results = json.load(f)
    with open(os.path.join(results_dir, "metrics.json")) as f:
        metrics = json.load(f)
    return results, metrics


def generate_latex_table(metrics: dict) -> str:
    b = metrics['baseline']
    s = metrics['security_aware']
    imp = metrics['improvement']

    lines = [
        r"\begin{table}[ht]",
        r"\centering",
        r"\caption{Baseline vs.\ Security-Aware Speculative Decoding}",
        r"\label{tab:main_results}",
        r"\begin{tabular}{lccc}",
        r"\hline",
        r"\textbf{Metric} & \textbf{Baseline} & \textbf{Security-Aware} & \textbf{$\Delta$} \\",
        r"\hline",
    ]

    rows = [
        ("Avg.\ Security Score (all)",
         b['avg_security_score'], s['avg_security_score']),
        ("Avg.\ Security Score (vul=1 prompts)",
         b['avg_security_score_vul1'], s['avg_security_score_vul1']),
        ("Avg.\ Security Score (vul=0 prompts)",
         b['avg_security_score_vul0'], s['avg_security_score_vul0']),
        ("Acceptance Rate",
         b['avg_acceptance_rate'], s['avg_acceptance_rate']),
        ("Syntax Validity Rate",
         b['syntax_valid_rate'], s['syntax_valid_rate']),
    ]

    for label, bv, sv in rows:
        delta = sv - bv
        sign = "+" if delta >= 0 else ""
        lines.append(
            f"{label} & {bv:.4f} & {sv:.4f} & {sign}{delta:.4f} \\\\"
        )

    lines += [
        r"\hline",
        rf"\% Examples Improved (vul=1) & \multicolumn{{2}}{{c}}{{---}} & "
        rf"{imp['fraction_improved_vul1']:.1%} \\",
        r"\hline",
        r"\end{tabular}",
        r"\end{table}",
    ]
    return "\n".join(lines)


def generate_per_cwe_table(metrics: dict) -> str:
    lines = [
        r"\begin{table}[ht]",
        r"\centering",
        r"\caption{Per-CWE Security Score Improvement (vul=1 prompts only)}",
        r"\label{tab:per_cwe}",
        r"\begin{tabular}{lrcccc}",
        r"\hline",
        r"\textbf{CWE} & \textbf{N} & \textbf{Baseline} & "
        r"\textbf{Sec-Aware} & \textbf{$\Delta$} & \textbf{\% Improved} \\",
        r"\hline",
    ]
    for cwe, cv in sorted(metrics['per_cwe'].items()):
        delta = cv['score_delta']
        sign = "+" if delta >= 0 else ""
        lines.append(
            f"{cwe} & {cv['n']} & {cv['baseline_avg_score']:.4f} & "
            f"{cv['security_avg_score']:.4f} & {sign}{delta:.4f} & "
            f"{cv['fraction_improved']:.1%} \\\\"
        )
    lines += [r"\hline", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


def generate_summary_paragraph(metrics: dict) -> str:
    b = metrics['baseline']
    s = metrics['security_aware']
    imp = metrics['improvement']
    n = metrics['total_examples']
    n_vul = metrics['n_vulnerable_prompts']

    delta = imp['score_delta_vul1']
    frac  = imp['fraction_improved_vul1']
    accept_delta = imp['acceptance_rate_delta']

    best_cwe = max(metrics['per_cwe'].items(), key=lambda x: x[1]['score_delta'], default=(None, {}))
    worst_cwe = min(metrics['per_cwe'].items(), key=lambda x: x[1]['score_delta'], default=(None, {}))

    paragraph = (
        f"We evaluated our security-aware speculative decoding approach on {n} examples "
        f"({n_vul} from vulnerable functions) drawn from the combined Big-Vul/Python "
        f"vulnerability dataset. For each example, we extracted only the function signature "
        f"as a generation prompt, preventing the model from simply copying the known-vulnerable body. "
        f"\n\n"
        f"On the vulnerable-prompt subset (vul=1), our method achieved a mean CodeBERT security "
        f"score of {s['avg_security_score_vul1']:.4f} versus {b['avg_security_score_vul1']:.4f} "
        f"for baseline speculative decoding, a delta of {delta:+.4f}. "
        f"{frac:.1%} of vulnerable-prompt examples showed a security improvement. "
        f"\n\n"
    )

    if best_cwe[0]:
        paragraph += (
            f"The largest per-class improvement was on {best_cwe[0]} "
            f"({best_cwe[1]['score_delta']:+.4f}), while {worst_cwe[0]} showed the "
            f"smallest improvement ({worst_cwe[1]['score_delta']:+.4f}), "
            f"suggesting the security modifier is most effective for certain vulnerability classes. "
        )

    paragraph += (
        f"\n\nThe acceptance rate changed by {accept_delta:+.4f}, indicating the security modifier "
        f"{'actively intervened' if abs(accept_delta) > 0.05 else 'had limited intervention'} "
        f"during generation. Syntax validity rates were {b['syntax_valid_rate']:.1%} (baseline) "
        f"and {s['syntax_valid_rate']:.1%} (security-aware), suggesting code quality was "
        f"{'preserved' if abs(s['syntax_valid_rate'] - b['syntax_valid_rate']) < 0.05 else 'slightly affected'}."
    )
    return paragraph


def try_plot(results, metrics, results_dir):
    """Generate plots — graceful fallback if matplotlib not available."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("  matplotlib not available — skipping plots. Install with: pip install matplotlib")
        return

    plt.style.use('seaborn-v0_8-whitegrid')
    fig_dir = os.path.join(results_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    # 1. Score distribution: overlapping histograms
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, subset, title in zip(
        axes,
        [results, [r for r in results if r['ground_truth_vul'] == 1]],
        ["All Prompts", "Vulnerable Prompts Only (vul=1)"]
    ):
        b_scores = [r['baseline_score'] for r in subset]
        s_scores = [r['security_score'] for r in subset]
        bins = np.linspace(0, 1, 25)
        ax.hist(b_scores, bins=bins, alpha=0.55, label='Baseline', color='#E15759')
        ax.hist(s_scores, bins=bins, alpha=0.55, label='Security-Aware', color='#4E79A7')
        ax.axvline(np.mean(b_scores), color='#E15759', linestyle='--', linewidth=1.5)
        ax.axvline(np.mean(s_scores), color='#4E79A7', linestyle='--', linewidth=1.5)
        ax.set_xlabel("CodeBERT Security Score (P(secure))", fontsize=11)
        ax.set_ylabel("Count", fontsize=11)
        ax.set_title(title, fontsize=12)
        ax.legend()

    plt.suptitle("Security Score Distribution: Baseline vs Security-Aware", fontsize=13, y=1.02)
    plt.tight_layout()
    path = os.path.join(fig_dir, "score_distribution.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved {path}")

    # 2. Per-CWE bar chart
    per_cwe = metrics['per_cwe']
    if per_cwe:
        cwes = list(per_cwe.keys())
        base_vals = [per_cwe[c]['baseline_avg_score'] for c in cwes]
        sec_vals  = [per_cwe[c]['security_avg_score'] for c in cwes]
        x = np.arange(len(cwes))
        width = 0.35

        fig, ax = plt.subplots(figsize=(max(8, len(cwes) * 1.5), 5))
        ax.bar(x - width/2, base_vals, width, label='Baseline', color='#E15759', alpha=0.8)
        ax.bar(x + width/2, sec_vals,  width, label='Security-Aware', color='#4E79A7', alpha=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(cwes, rotation=30, ha='right', fontsize=9)
        ax.set_ylabel("Avg CodeBERT Security Score", fontsize=11)
        ax.set_title("Per-CWE Security Score Comparison", fontsize=13)
        ax.legend()
        ax.set_ylim(0, 1)

        plt.tight_layout()
        path = os.path.join(fig_dir, "per_cwe_bar.png")
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Saved {path}")

    # 3. Score delta scatter vs acceptance rate delta
    deltas        = [r['security_score'] - r['baseline_score'] for r in results]
    accept_deltas = [r['security_accept'] - r['baseline_accept'] for r in results]
    vul_colors    = ['#E15759' if r['ground_truth_vul'] == 1 else '#4E79A7' for r in results]

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(accept_deltas, deltas, c=vul_colors, alpha=0.5, s=20)
    ax.axhline(0, color='black', linewidth=0.8, linestyle='--')
    ax.axvline(0, color='black', linewidth=0.8, linestyle='--')
    ax.set_xlabel("Acceptance Rate Δ (sec-aware − baseline)", fontsize=11)
    ax.set_ylabel("Security Score Δ (sec-aware − baseline)", fontsize=11)
    ax.set_title("Security Improvement vs Acceptance Rate Change", fontsize=12)
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#E15759', alpha=0.6, label='vul=1 (vulnerable prompt)'),
        Patch(facecolor='#4E79A7', alpha=0.6, label='vul=0 (clean prompt)'),
    ]
    ax.legend(handles=legend_elements)
    plt.tight_layout()
    path = os.path.join(fig_dir, "score_vs_acceptance_scatter.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved {path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--results_dir', default='./eval_results')
    args = p.parse_args()

    print(f"Loading results from {args.results_dir}...")
    results, metrics = load(args.results_dir)
    print(f"  ✓ {len(results)} examples loaded")

    # LaTeX tables
    main_table  = generate_latex_table(metrics)
    cwe_table   = generate_per_cwe_table(metrics)
    summary     = generate_summary_paragraph(metrics)

    paper_path = os.path.join(args.results_dir, "paper_tables.tex")
    with open(paper_path, 'w') as f:
        f.write("% ── Main Results Table ──────────────────────────────────────────\n\n")
        f.write(main_table)
        f.write("\n\n% ── Per-CWE Table ───────────────────────────────────────────────\n\n")
        f.write(cwe_table)
    print(f"✓ LaTeX tables saved → {paper_path}")

    summary_path = os.path.join(args.results_dir, "summary_paragraph.txt")
    with open(summary_path, 'w') as f:
        f.write(summary)
    print(f"✓ Summary paragraph saved → {summary_path}")
    print(f"\n--- FINDINGS ---\n{summary}\n")

    # Plots
    print("\nGenerating plots...")
    try_plot(results, metrics, args.results_dir)

    print(f"\n✓ Analysis complete. Outputs in {args.results_dir}/")


if __name__ == "__main__":
    main()