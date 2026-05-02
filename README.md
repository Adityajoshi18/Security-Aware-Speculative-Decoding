# Security-Aware Speculative Decoding

> Steering frozen LLMs toward secure code generation at inference time — zero fine-tuning required.

## Overview

This project presents a novel approach to making Large Language Models (LLMs) generate more secure code **without modifying the model's weights**. We integrate a neural security classifier into the speculative decoding acceptance criterion, biasing token acceptance toward secure code patterns at inference time.

The approach is **model-agnostic** — it works with any drafter-target model pair and requires no fine-tuning of any LLM.

---

## The Core Idea

Standard speculative decoding uses a small drafter model to propose candidate tokens, which a larger target model then verifies:

```
Standard: accept token if Uniform(0,1) < p_target(x) / p_drafter(x)
```

We modify the acceptance criterion by multiplying with a **security modifier** derived from a CodeBERT-based classifier:

```
Ours:     accept token if Uniform(0,1) < [p_target(x) / p_drafter(x)] × security_modifier
```

Where:
- `security_modifier < 1.0` → token is harder to accept (penalizes insecure-looking code)
- `security_modifier > 1.0` → token is easier to accept (rewards secure-looking code)

The classifier scores complete statements as they are generated, and the modifier is applied to all subsequent tokens until the next statement boundary.

---

## Results

Evaluated on the **SecurityEval benchmark** (121 prompts) using Llama-3.2-3B (target) + Llama-3.2-1B (drafter):

| Method | Bandit Issues/Prompt | Vuln% | Accept Rate | Tok/s |
|---|---|---|---|---|
| Baseline (greedy) | 0.0331 | 3.3% | — | 37.6 |
| Prompt-only | 0.0331 | 3.3% | — | 37.7 |
| **Ours (sec_spec)** | **0.0083** | **0.8%** | **0.575** | **34.3** |

**75.8% relative reduction** in Bandit-detected vulnerabilities.  
**Prompt engineering alone = 0% improvement.**

Vulnerabilities measured independently by [Bandit](https://bandit.readthedocs.io/), a third-party static analysis tool that has no knowledge of our classifier or training data.

### Configuration
```
γ (gamma)         = 5       # draft tokens per step
λ (lambda)        = 1.5     # security modifier strength
threshold         = 0.2     # security score cutoff
target model      = meta-llama/Llama-3.2-3B
drafter model     = meta-llama/Llama-3.2-1B
classifier        = microsoft/codebert-base (fine-tuned)
classifier acc    = 99.4%
```

---

## Project Structure

```
secure-codegen/
├── src/
│   ├── pipeline.py               # Shared functions (classifier, speculative decoding)
│   ├── prepare_dataset.py        # Build security fragment dataset from CodeSearchNet
│   ├── train_classifier.py       # Fine-tune CodeBERT security classifier
│   └── evaluate.py               # Full evaluation on SecurityEval benchmark
├── data/
│   ├── fragments_train.jsonl     # 7,757 training fragments
│   ├── fragments_val.jsonl       # 1,662 validation fragments
│   └── fragments_test.jsonl      # 1,663 test fragments
├── models/
│   └── fragment_classifier/      # fine-tuned model (tracked via Git LFS)
│       ├── best_model.pt         # 476MB — stored with Git LFS
│       ├── metadata.json         # training results and configuration
│       ├── tokenizer.json        # CodeBERT tokenizer
│       └── tokenizer_config.json # tokenizer configuration
├── results/
│   └── evaluation_results.json   # Full results from our run
├── requirements.txt
└── README.md
```

---

## Approach Details

### Step 1 — Dataset Preparation (`prepare_dataset.py`)

We scan **457,461 Python functions** from CodeSearchNet and extract code fragments around security-relevant lines using regex pattern matching.

**Insecure patterns (label=1):**
- CWE-89: SQL injection via string concatenation or f-strings in `execute()`
- CWE-78: Command injection via `shell=True` or `os.system()`
- CWE-502: Unsafe deserialization via `pickle.loads()`
- CWE-798: Hardcoded credentials
- CWE-327: Weak cryptography (`md5`, `sha1`)
- CWE-22: Path traversal via `open()` with string concatenation

**Secure patterns (label=0):**
- Parameterized SQL (`%s`, `?`)
- Safe subprocess (list form, no `shell=True`)
- Safe deserialization (`json.loads`, `yaml.safe_load`)
- Strong crypto (`pbkdf2_hmac`, `bcrypt`, `argon2`)
- Environment variables for secrets (`os.environ`, `os.getenv`)
- Input validation (`isinstance`, `raise ValueError`)

Each matched line is extracted with **3 lines of context** before and after. The dataset is balanced (equal insecure/secure) and split 70/15/15 (train/val/test).

### Step 2 — Classifier Training (`train_classifier.py`)

Fine-tunes `microsoft/codebert-base` as a binary classifier:

```
CodeBERT (124M params) → Dropout(0.1) → Linear(768 → 2) → logits
```

The `[CLS]` token representation encodes the meaning of the entire fragment. After training, **temperature calibration** (LBFGS) is applied to the validation set so the classifier's confidence scores are well-calibrated for use as modifiers.

**Training config:**
- 5 epochs, batch size 32, lr=2e-5
- Linear warmup (10%) + linear decay
- Best checkpoint saved by validation loss

**Results:** Test accuracy=99.4%, AUC=99.98%

### Step 3 — Security-Aware Speculative Decoding (`pipeline.py`, `evaluate.py`)

The key components:

**StatementDetector** — watches the stream of generated tokens and fires when a complete Python statement has been generated (skips control flow headers, continuations, comments). Returns the statement with 5 lines of surrounding context so the classifier sees relevant code, not just one line.

**score_fragment** — runs the fragment through the classifier with calibration temperature T, returns P(secure) ∈ [0,1].

**compute_modifier** — maps P(secure) to an acceptance multiplier:
```python
# threshold=0.2 because this classifier scores neutral code at 0.01-0.15
# Using 0.5 would penalise almost everything
if score < threshold:
    modifier = max(0.05, 1.0 - 1.5 * (threshold - score))  # penalty
else:
    modifier = min(2.0,  1.0 + 1.5 * (score - threshold))  # reward
```

**security_aware_speculative_generate** — the main loop:
1. Drafter generates γ=5 candidate tokens
2. StatementDetector checks if a new statement boundary has been reached
3. If yes, classifier scores the statement → compute_modifier updates current_modifier
4. Target verifies all γ tokens in one forward pass
5. Accept/reject each token: `Uniform(0,1) < (p_target / p_drafter) * current_modifier`
6. Correction or bonus token sampled as per standard speculative decoding

---

## Setup & Installation

### Prerequisites
- Python 3.9+
- CUDA GPU (recommended — CPU will be very slow for LLM inference)
- HuggingFace account with access to Llama-3.2 models

### Installation

```bash
# Clone the repo
git clone git@github.com:sidpatondikar/secure-codegen.git
cd secure-codegen

# Create virtual environment
python -m venv venv
source venv/bin/activate   # Linux/Mac
# venv\Scripts\activate    # Windows

# Install dependencies
pip install -r requirements.txt
```

### Git LFS

The trained model weights are stored with Git LFS. To download them after cloning:

```bash
git lfs pull
```

If you clone without LFS, `best_model.pt` will appear as a small pointer file instead of the actual weights. Run `git lfs pull` to get the full 476MB file.

### HuggingFace Access

Llama-3.2 models require accepting the license on HuggingFace and logging in:

```bash
huggingface-cli login
```

---

## How to Run

### View existing results

The dataset, trained model, and full evaluation results from our run are already included in the repo. You can inspect them directly without running anything:

- Dataset: `data/`
- Model: `models/fragment_classifier/` (download via `git lfs pull`)
- Results: `results/evaluation_results.json`

### Run the full pipeline from scratch

**Step 1: Build the dataset** *(optional — dataset already included in `data/`)*
```bash
python src/prepare_dataset.py
```
Downloads CodeSearchNet Python (~500MB), scans all functions, saves balanced fragments to `data/`. Skip this step if using the included dataset.

**Step 2: Train the classifier** *(optional — trained model already included in `models/`)*
```bash
python src/train_classifier.py
```
Fine-tunes CodeBERT for 5 epochs. Requires GPU. Skip this step if using the included model weights — the trained model is already available in `models/fragment_classifier/` via Git LFS.

**Step 3: Run evaluation**
```bash
python src/evaluate.py
```
Runs all 121 SecurityEval prompts across 3 conditions (baseline, prompt-only, sec_spec). Requires a GPU with enough VRAM to load Llama-3.2-3B and Llama-3.2-1B simultaneously (~10GB). Saves results to `results/evaluation_results.json`.

### Quick test run

To verify the pipeline works before running all 121 prompts, set `MAX_PROMPTS = 10` in `src/evaluate.py`:

```python
MAX_PROMPTS = 10  # line 40 in evaluate.py
```

---

## Using Different Models

To use Qwen instead of Llama, change these lines in `src/evaluate.py`:

```python
TARGET_MODEL  = 'Qwen/Qwen2.5-Coder-7B-Instruct'
DRAFTER_MODEL = 'Qwen/Qwen2.5-Coder-1.5B-Instruct'
```

The drafter and target must share the same tokenizer (same model family).

---

## Why This Works

**Why not just use a security system prompt?**  
Our results show prompt-only engineering achieves 0% improvement over baseline. The model ignores security instructions when the completion context strongly suggests an insecure pattern. Our method intervenes at the token level, making insecure completions statistically harder to generate regardless of context.

**Why speculative decoding?**  
Speculative decoding already requires a drafter + target pair and a per-token acceptance decision. Our modification adds a single multiplication to the acceptance criterion — minimal overhead, maximum leverage.

**Why threshold=0.2 and not 0.5?**  
This classifier scores neutral, non-security-related code at 0.01–0.15 due to training data distribution. A threshold of 0.5 would penalize almost all generated code. Threshold=0.2 makes the modifier selective — it only fires on genuinely insecure-looking patterns.

---

## Limitations

- Baseline vulnerability rate was already low (3.3%) — Llama-3.2-3B is already mostly safe on SecurityEval
- Classifier trained on pattern-matched labels, not ground-truth CVEs
- Covers 7 CWE types — not all possible vulnerability classes
- Drafter and target must be from the same model family (shared tokenizer)

---

## Dependencies

| Package | Purpose |
|---|---|
| `torch` | Model training and inference |
| `transformers` | CodeBERT, Llama, tokenizers |
| `datasets` | CodeSearchNet, SecurityEval |
| `scikit-learn` | Train/val/test split, metrics |
| `bandit` | Static vulnerability analysis |
| `tqdm` | Progress bars |
