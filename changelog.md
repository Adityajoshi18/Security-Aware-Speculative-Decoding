# secure-codegen Change Log
**Sprint 4 | April 2026 | Siddharth Patondikar**

---

## S4 Test 1: Trying Smarter Token Gate and Lambda Values

The Sprint 3 evaluation showed the security modifier barely fired — acceptance rate only dropped by 0.3% and roughly half the examples improved vs half degraded (near coin-flip). These two changes directly target the two root causes of that.

---

### Change 1 — Smarter Token Gate

**The problem.**
The pipeline had a rule: only score a code statement if it has at least 5 "words" in it. But words were counted by splitting on spaces — so a line like `pCluster = (Cluster*)malloc(sizeof(Cluster))` only counted as 3 words and got silently skipped, even though it's a meaningful statement CodeBERT could score. This was a major reason CodeBERT rarely triggered.

**What was changed.**
The gate now uses CodeBERT's own tokenizer to count tokens instead of splitting on spaces. CodeBERT breaks code into subword pieces — so `malloc` becomes 1 token but `(sizeof(Cluster))` becomes roughly 6. The minimum threshold was bumped from 5 whitespace words to 10 CodeBERT subword tokens.

**Files changed.**
- `sampling/security_scorer.py` — added `count_tokens()` method
- `sampling/security_aware_speculative.py` — gate now calls `count_tokens()` and `MIN_SCOREABLE_TOKENS` is 10

**What to watch for.**
The acceptance rate delta should increase from the near-zero Sprint 3 value. That's the signal that CodeBERT is actually scoring more statements. If it's still near 0%, the gate fix didn't unlock enough statements and something else is filtering them.

---

### Change 2 — Lambda Sweep

**What lambda is.**
Lambda (λ) controls how hard the pipeline pushes back on insecure-looking code. At λ=1.0 (the Sprint 3 setting) the penalty was mild. At λ=2.0 an insecure statement gets rejected roughly twice as hard. At λ=5.0 the modifier is very aggressive.

**What was built.**
A new script — `experiments/lambda_sweep.py` — runs the full evaluation back-to-back for six lambda values: `0.5, 1.0, 1.5, 2.0, 3.0, 5.0`. Each run saves to its own folder inside `eval_results_sweep/`, and at the end a summary table is printed comparing score delta and fraction-improved across all lambdas.

**How to run it.**
```bash
# Quick test (10 examples per lambda)
python lambda_sweep.py --smoke_test

# Full run (200 examples per lambda)
python lambda_sweep.py

# Specific lambdas only
python lambda_sweep.py --lambdas 1.5 2.0 3.0 --smoke_test
```

**What to look for.**
- `fraction_improved` rises with lambda then drops → there's a sweet spot, pick the peak
- `fraction_improved` flat across all lambdas → lambda isn't the bottleneck, scorer mismatch is the real issue
- `vul=0` scores drop at high lambda → CodeBERT is falsely penalising clean code, don't push lambda that high

---

### Results
*To be filled in after testing.*

---
