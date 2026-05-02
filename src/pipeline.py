"""
pipeline.py — Shared functions for Security-Aware Speculative Decoding

Contains:
- FragmentSecurityClassifier  : CodeBERT-based security classifier
- score_fragment              : returns P(secure) for a code fragment
- is_scoreable                : checks if fragment is worth scoring
- detect_language             : Python vs C detection
- strip_comments              : removes comments before scoring
- StatementDetector           : detects complete statements in generated code
- compute_modifier            : maps security score to acceptance multiplier
- normalise_logits            : handles drafter/target vocab size mismatch
- security_aware_speculative_generate : main generation function
"""

import re
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional
from transformers import AutoTokenizer, AutoModel

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ── Classifier ─────────────────────────────────────────────────────────────

class FragmentSecurityClassifier(nn.Module):
    def __init__(self, model_name, dropout=0.1):
        super().__init__()
        self.encoder    = AutoModel.from_pretrained(model_name, use_safetensors=True)
        self.dropout    = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.encoder.config.hidden_size, 2)

    def forward(self, input_ids, attention_mask):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls = out.last_hidden_state[:, 0, :]
        return self.classifier(self.dropout(cls))


def load_classifier(model_dir: str):
    """
    Load trained classifier, tokenizer, and metadata from model_dir.
    Returns (sec_model, sec_tokenizer, meta)
    """
    model_dir = Path(model_dir)
    with open(model_dir / 'metadata.json') as f:
        meta = json.load(f)

    sec_tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    sec_model     = FragmentSecurityClassifier(meta['base_model'])
    sec_model.load_state_dict(
        torch.load(model_dir / 'best_model.pt', map_location=DEVICE, weights_only=True)
    )
    sec_model = sec_model.to(DEVICE)
    sec_model.eval()
    return sec_model, sec_tokenizer, meta


# ── Scoring helpers ─────────────────────────────────────────────────────────

@torch.no_grad()
def score_fragment(code: str, sec_model, sec_tokenizer, max_len: int, calib_t: float) -> float:
    """Returns P(secure) in [0, 1]. Higher = more secure."""
    enc = sec_tokenizer(
        code, max_length=max_len, padding='max_length',
        truncation=True, return_tensors='pt'
    )
    logits = sec_model(
        enc['input_ids'].to(DEVICE),
        enc['attention_mask'].to(DEVICE)
    )
    return F.softmax(logits / calib_t, dim=-1)[0, 0].item()


def is_scoreable(code: str, sec_tokenizer) -> bool:
    """Returns True if fragment has enough tokens to be worth scoring."""
    if not code or len(code.strip()) < 15:
        return False
    return len(sec_tokenizer.tokenize(code)) >= 10


# ── Language detection & comment stripping ──────────────────────────────────

def detect_language(code: str) -> str:
    py = sum(1 for s in ['def ', 'import ', 'print(', 'os.', 'cursor.'] if s in code)
    c  = sum(1 for s in ['#include', 'malloc(', 'strcpy', 'printf('] if s in code)
    return 'python' if py >= c else 'c'


def strip_comments(code: str, lang: str) -> str:
    if 'python' in lang:
        code = re.sub(r'"""[\s\S]*?"""', '', code)
        code = re.sub(r'#.*', '', code)
    else:
        code = re.sub(r'/\*[\s\S]*?\*/', '', code)
        code = re.sub(r'//.*', '', code)
    return code.strip()


# ── Statement detector ──────────────────────────────────────────────────────

class StatementDetector:
    """
    Detects complete statement boundaries in incrementally growing generated code.
    Returns the statement with context window (last 5 lines) so the classifier
    sees surrounding code, not just a single line.
    Deduplication ensures the same statement is never scored twice.
    """

    def __init__(self, context_lines: int = 5):
        self.context_lines = context_lines
        self._last_len     = 0
        self._seen         = set()

    def reset(self):
        self._last_len = 0
        self._seen.clear()

    def has_new_statement(self, code: str) -> Tuple[bool, str]:
        lang = detect_language(code)
        return self._check_python(code, lang) if 'python' in lang else self._check_c(code, lang)

    def _check_python(self, code: str, lang: str) -> Tuple[bool, str]:
        lines = code.split('\n')
        for i in range(len(lines) - 2, -1, -1):
            line = lines[i].strip()
            if not line:
                continue
            if re.match(r'^(def |class |if |elif |else:|for |while |try:|except|finally:|with |#|@)', line):
                continue
            if line.endswith(('\\', ',', '(')):
                continue
            offset = sum(len(l) + 1 for l in lines[:i + 1])
            if offset <= self._last_len:
                break
            clean = strip_comments(line, lang)
            if not clean:
                continue
            norm = re.sub(r'\s+', ' ', clean).strip()
            if norm in self._seen:
                continue
            self._seen.add(norm)
            self._last_len = offset
            ctx = '\n'.join(lines[max(0, i - self.context_lines): i + 1])
            return True, ctx
        return False, ''

    def _check_c(self, code: str, lang: str) -> Tuple[bool, str]:
        new_text = code[self._last_len:]
        if not (';' in new_text or '}' in new_text):
            return False, ''
        parts = re.split(r'[;{}]', code)
        if len(parts) < 2:
            return False, ''
        candidate = parts[-2].strip()
        if not candidate:
            self._last_len = len(code)
            return False, ''
        clean = strip_comments(candidate, lang)
        if not clean:
            self._last_len = len(code)
            return False, ''
        norm = re.sub(r'\s+', ' ', clean).strip()
        if norm in self._seen:
            self._last_len = len(code)
            return False, ''
        self._seen.add(norm)
        self._last_len = len(code)
        all_lines = code.split('\n')
        end_line  = code[:code.rfind(candidate)].count('\n')
        ctx = '\n'.join(all_lines[max(0, end_line - self.context_lines): end_line + 1])
        return True, ctx


# ── Security modifier ───────────────────────────────────────────────────────

def compute_modifier(
    score:     float,
    threshold: float = 0.2,
    lam:       float = 1.5,
    floor:     float = 0.05,
    ceiling:   float = 2.0,
) -> float:
    """
    Maps P(secure) score to an acceptance probability multiplier.

    score < threshold  → penalty:  max(floor,   1 - λ*(threshold - score))
    score >= threshold → bonus:    min(ceiling,  1 + λ*(score - threshold))

    Default threshold=0.2 because this classifier scores neutral code at 0.01-0.15.
    Using 0.5 would penalise almost everything.
    """
    if score < threshold:
        return max(floor,   1.0 - lam * (threshold - score))
    return min(ceiling, 1.0 + lam * (score - threshold))


# ── Vocab normalisation ─────────────────────────────────────────────────────

def normalise_logits(logits: torch.Tensor, target_vocab: int) -> torch.Tensor:
    """Clip or pad logits to target_vocab size (handles drafter/target vocab mismatch)."""
    actual = logits.shape[-1]
    if actual == target_vocab:
        return logits
    if actual > target_vocab:
        return logits[..., :target_vocab]
    pad = torch.full(
        (*logits.shape[:-1], target_vocab - actual),
        float('-inf'), dtype=logits.dtype, device=logits.device
    )
    return torch.cat([logits, pad], dim=-1)


# ── Core generation function ────────────────────────────────────────────────

@torch.no_grad()
def security_aware_speculative_generate(
    inputs:             List[int],
    drafter,
    target,
    gen_tokenizer,
    sec_model,
    sec_tokenizer,
    max_len_sec:        int,
    calib_t:            float,
    gamma:              int   = 5,
    max_gen_len:        int   = 256,
    eos_tokens_id:      Optional[List[int]] = None,
    pad_token_id:       int   = 0,
    security_weight:    float = 1.5,
    security_threshold: float = 0.2,
    temperature:        float = 1.0,
) -> Tuple[List[int], float, List[Dict[str, Any]]]:
    """
    Security-aware speculative decoding.

    Core contribution:
        Standard:  accept if Uniform(0,1) < p(x) / q(x)
        Ours:      accept if Uniform(0,1) < [p(x) / q(x)] * security_modifier

    Returns: (generated_token_ids, acceptance_rate, security_log)
    """
    if eos_tokens_id is None:
        eos_tokens_id = [gen_tokenizer.eos_token_id]

    gen_device  = next(target.parameters()).device
    vocab_size  = target.config.vocab_size
    stop_tokens = torch.tensor(eos_tokens_id, dtype=torch.long, device=gen_device).unsqueeze(1)

    prompt_len = len(inputs)
    max_seq    = getattr(target.config, 'max_position_embeddings', 4096)
    total_len  = min(max_seq, prompt_len + max_gen_len)

    input_ids = torch.full((1, total_len), pad_token_id, dtype=torch.long, device=gen_device)
    input_ids[0, :prompt_len] = torch.tensor(inputs, dtype=torch.long, device=gen_device)

    detector         = StatementDetector(context_lines=5)
    security_log     = []
    current_modifier = 1.0
    drafts_accepted  = 0.0
    drafts_speculated= 0.0
    current_pos      = prompt_len

    # Seed: first token from target
    out    = target(input_ids=input_ids[..., :current_pos], use_cache=False)
    logits = normalise_logits(out.logits[..., -1, :], vocab_size)
    first  = torch.argmax(F.softmax(logits / temperature, dim=-1), dim=-1, keepdim=True)
    input_ids[0, current_pos] = first
    current_pos += 1

    if torch.isin(first, stop_tokens):
        return input_ids[0, prompt_len:current_pos].tolist(), 0.0, []

    while current_pos < total_len:
        gamma_t = min(gamma, total_len - current_pos - 1)
        if gamma_t <= 0:
            break

        # 1. Drafter generates γ candidate tokens
        q = torch.zeros((1, gamma_t, vocab_size), device=gen_device)
        for k in range(gamma_t):
            dout = drafter(input_ids=input_ids[..., :current_pos + k], use_cache=False)
            dp   = F.softmax(normalise_logits(dout.logits[..., -1, :], vocab_size) / temperature, dim=-1)
            q[0, k] = dp
            input_ids[0, current_pos + k] = torch.argmax(dp, dim=-1, keepdim=True)

        drafts_speculated += gamma_t

        # 2. Statement detection
        text = gen_tokenizer.decode(input_ids[0, :current_pos + gamma_t], skip_special_tokens=True)
        found, ctx = detector.has_new_statement(text)

        # 3. Security scoring
        if found and ctx and is_scoreable(ctx, sec_tokenizer):
            sc               = score_fragment(ctx, sec_model, sec_tokenizer, max_len_sec, calib_t)
            current_modifier = compute_modifier(sc, security_threshold, security_weight)
            security_log.append({'score': sc, 'modifier': current_modifier})

        # 4. Target verifies all γ tokens in one forward pass
        tout = target(input_ids=input_ids[..., :current_pos + gamma_t], use_cache=False)
        tl   = normalise_logits(
            tout.logits[..., current_pos - 1: current_pos + gamma_t - 1, :], vocab_size
        )
        p = F.softmax(tl / temperature, dim=-1)

        # 5. Accept/reject with security modifier
        r = torch.rand(gamma_t, device=gen_device)
        n = gamma_t
        for i in range(gamma_t):
            tok   = input_ids[0, current_pos + i]
            ratio = p[0, i, tok].item() / (q[0, i, tok].item() + 1e-9)
            if r[i].item() > ratio * current_modifier:
                n = i
                break

        drafts_accepted += n

        # 6. Stop token check
        stop_locs = torch.nonzero(torch.eq(input_ids[..., current_pos: current_pos + n], stop_tokens))
        if stop_locs.shape[0] > 0:
            current_pos += stop_locs[0, 1].item() + 1
            rate = drafts_accepted / drafts_speculated if drafts_speculated else 0.0
            return input_ids[0, prompt_len:current_pos].tolist(), rate, security_log

        # 7. Correction or bonus token
        if n == gamma_t:
            bl = normalise_logits(tout.logits[..., current_pos + gamma_t - 1, :], vocab_size)
            correction = torch.argmax(F.softmax(bl / temperature, dim=-1), dim=-1, keepdim=True)
        else:
            adj   = torch.clamp(p[0, n] - q[0, n], min=0.0)
            total = adj.sum()
            adj   = adj / total if total > 0 else torch.ones_like(adj) / vocab_size
            correction = torch.argmax(adj, dim=-1, keepdim=True)

        input_ids[0, current_pos + n: current_pos + gamma_t] = pad_token_id
        input_ids[0, current_pos + n] = correction
        current_pos += n + 1

        if torch.isin(correction, stop_tokens):
            rate = drafts_accepted / drafts_speculated if drafts_speculated else 0.0
            return input_ids[0, prompt_len:current_pos].tolist(), rate, security_log

    rate = drafts_accepted / drafts_speculated if drafts_speculated else 0.0
    return input_ids[0, prompt_len:current_pos].tolist(), rate, security_log
