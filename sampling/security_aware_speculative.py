#!/usr/bin/env python3
"""
Security-Aware Speculative Decoding

Integrates CodeBERT security scoring into speculative decoding:
1. Draft model generates γ tokens
2. Check for statement boundaries in drafted code
3. Score completed statements with CodeBERT
   - Comments and docstrings are stripped before scoring
   - Pure comment/docstring statements are skipped entirely
   - Statements shorter than MIN_SCOREABLE_TOKENS are skipped
     (short declarations like "int i" or "int ret = 0" don't carry
      enough semantic signal for CodeBERT to score reliably)
4. Modify target acceptance based on security scores
"""

import torch
import torch.nn.functional as F
from typing import List, Tuple
import sys
sys.path.insert(0, '../')

from sampling.security_scorer import SecurityScorer
from sampling.statement_detector import StatementDetector, is_scoreable_statement

# Minimum number of CodeBERT subword tokens a statement must have before
# it is sent to the scorer. Using CodeBERT's own tokenizer instead of
# whitespace splitting means dense expressions like
#   pCluster = (Cluster*)malloc(sizeof(Cluster))
# correctly count as ~15 subword tokens and pass the gate, whereas the
# old whitespace split counted them as only 3 "words" and silently dropped
# them. Threshold of 10 subword tokens filters trivial declarations
# ("int i", "int ret = 0") while letting real statements through.
MIN_SCOREABLE_TOKENS = 10


@torch.no_grad()
def security_aware_speculative_generate(
    inputs: List[int],
    drafter,
    target,
    tokenizer,
    security_scorer: SecurityScorer,
    gamma: int = 5,
    max_gen_len: int = 200,
    eos_tokens_id: List[int] = [1],
    pad_token_id: int = 0,
    security_weight: float = 1.0,
    security_threshold: float = 0.5,
    temperature: float = 1.0,
    debug: bool = False,
) -> Tuple[List[int], float, List[dict]]:
    """
    Security-aware speculative decoding.

    Args:
        inputs: Input token IDs
        drafter: Draft model (small, fast)
        target: Target model (large, accurate)
        tokenizer: Tokenizer
        security_scorer: CodeBERT security model
        gamma: Number of tokens to draft at once
        max_gen_len: Maximum tokens to generate
        eos_tokens_id: End-of-sequence token IDs
        pad_token_id: Padding token ID
        security_weight: How much security affects acceptance (λ)
        security_threshold: Security score threshold (default 0.5)
        temperature: Sampling temperature
        debug: Print debug information

    Returns:
        (generated_tokens, acceptance_rate, security_scores)
    """

    device = target.device
    vocab_size = target.config.vocab_size

    # Setup
    stop_tokens = torch.tensor(eos_tokens_id, dtype=torch.long, device=device).unsqueeze(1)
    drafter_cache, target_cache = None, None
    drafts_accepted, drafts_speculated = 0.0, 0.0

    # Prepare input buffer
    prompt_len = len(inputs)
    max_seq_length = getattr(target.config, 'max_position_embeddings', 4096)
    total_len = min(max_seq_length, prompt_len + max_gen_len)

    input_ids = torch.full((1, total_len), pad_token_id, dtype=torch.long, device=device)
    input_ids[0, :prompt_len] = torch.tensor(inputs, dtype=torch.long, device=device)

    # Statement tracking
    statement_detector = StatementDetector(language='auto')
    statement_detector.reset()
    security_scores = []
    current_security_score = 0.5  # Neutral initially

    current_position = prompt_len

    # Initial token from target
    Mp = target(
        input_ids=input_ids[..., :current_position],
        past_key_values=target_cache,
        use_cache=False
    )
    target_cache = Mp.past_key_values

    # Sample first token
    probs = F.softmax(Mp.logits[..., -1, :] / temperature, dim=-1)
    t = torch.argmax(probs, dim=-1).unsqueeze(-1)
    input_ids[0, current_position] = t
    current_position += 1

    if torch.isin(t, stop_tokens):
        return input_ids[0, prompt_len:current_position].tolist(), 0.0, []

    if debug:
        print("=" * 70)
        print("SECURITY-AWARE SPECULATIVE DECODING")
        print("=" * 70)
        decoded = tokenizer.decode(input_ids[0, :current_position], skip_special_tokens=True)
        print(f"Initial: {decoded}")
        print("=" * 70)

    # Main generation loop
    while current_position < total_len:
        corrected_gamma = min(gamma, total_len - current_position - 1)
        if corrected_gamma <= 0:
            break

        # Storage for draft probabilities
        q = torch.zeros((1, corrected_gamma, vocab_size), device=device)

        # === STEP 1: DRAFT GENERATES γ TOKENS ===
        for k in range(corrected_gamma):
            Mq = drafter(
                input_ids=input_ids[..., :current_position + k],
                past_key_values=drafter_cache,
                use_cache=False
            )
            drafter_cache = Mq.past_key_values

            draft_probs = F.softmax(Mq.logits[..., -1, :] / temperature, dim=-1)
            q[0, k] = draft_probs

            # Sample token from draft
            xi = torch.argmax(draft_probs, dim=-1).unsqueeze(-1)
            input_ids[0, current_position + k] = xi

        drafts_speculated += corrected_gamma

        # === STEP 2: CHECK FOR STATEMENT BOUNDARIES ===
        # Decode the drafted sequence to check for complete statements.
        # has_new_statement() already strips comments/docstrings from the
        # returned statement, so `statement` is always clean code only.
        current_text = tokenizer.decode(
            input_ids[0, :current_position + corrected_gamma],
            skip_special_tokens=True
        )

        has_statement, statement = statement_detector.has_new_statement(current_text)

        # Detect language for scorability check
        detected_lang = statement_detector.detect_language(current_text)

        if has_statement and statement:
            # === STEP 3: SCORE WITH CODEBERT ===

            # Gate 1: filter pure comments, docstrings, bare braces
            if not is_scoreable_statement(statement, detected_lang):
                has_statement = False
                if debug:
                    print(f"\n[Skipped] Non-code statement filtered: '{statement[:60]}'")

            # Gate 2: filter short declarations that CodeBERT can't score reliably.
            # Uses CodeBERT's own subword tokenizer (same tokenization used at
            # scoring time) rather than whitespace splitting, so dense C expressions
            # like "pCluster = (Cluster*)malloc(sizeof(Cluster))" are no longer
            # silently dropped by a whitespace-word count of 3.
            elif security_scorer.count_tokens(statement) < MIN_SCOREABLE_TOKENS:
                has_statement = False
                if debug:
                    n_toks = security_scorer.count_tokens(statement)
                    print(f"\n[Skipped] Too short to score "
                          f"({n_toks} CodeBERT tokens): '{statement[:60]}'")

            else:
                security_score = security_scorer.score_code(statement)
                current_security_score = security_score

                security_scores.append({
                    'statement': statement,
                    'score': security_score,
                    'position': current_position
                })

                if debug:
                    status = "⚠️  INSECURE" if security_score < security_threshold else "✓  SECURE"
                    print(f"\n[Statement Detected] Score: {security_score:.4f} | {status}")
                    print(f"  {statement[:80]}")

        # === STEP 4: TARGET VERIFICATION WITH SECURITY MODIFICATION ===
        Mp = target(
            input_ids=input_ids[..., :current_position + corrected_gamma],
            past_key_values=target_cache,
            use_cache=False
        )
        target_cache = Mp.past_key_values

        # Get target probabilities for drafted tokens
        target_logits = Mp.logits[..., current_position - 1:current_position + corrected_gamma - 1, :]
        p = F.softmax(target_logits / temperature, dim=-1)

        # === STEP 5: ACCEPTANCE WITH SECURITY PENALTY ===
        # Standard acceptance: r < p(x) / q(x)
        # Security-aware: r < p(x) / q(x) * security_modifier
        #
        # Modifier rules (only applied when a real code statement was scored):
        #   score < threshold  →  penalty:  modifier = 1 - λ × (threshold - score)
        #   score >= threshold →  bonus:    modifier = 1 + λ × (score - threshold)
        #   no statement yet   →  neutral:  modifier = 1.0
        security_modifier = 1.0
        if has_statement:
            if current_security_score < security_threshold:
                penalty = security_weight * (security_threshold - current_security_score)
                security_modifier = max(0.1, 1.0 - penalty)  # floor at 0.1
            else:
                bonus = security_weight * (current_security_score - security_threshold)
                security_modifier = min(2.0, 1.0 + bonus)    # cap at 2.0

        # Rejection sampling with security modification
        r = torch.rand(corrected_gamma, device=device)
        n = corrected_gamma

        for i in range(corrected_gamma):
            token_i = input_ids[0, current_position + i]
            acceptance_prob = (p[0, i, token_i] / q[0, i, token_i]) * security_modifier

            if r[i] > acceptance_prob:
                n = i
                break

        drafts_accepted += n

        if debug and has_statement:
            print(f"  Security modifier: {security_modifier:.3f}")
            print(f"  Accepted {n}/{corrected_gamma} drafts")

        # Check for stop tokens in accepted drafts
        stop_locs = torch.nonzero(
            torch.eq(input_ids[..., current_position:current_position + n], stop_tokens)
        )
        if stop_locs.shape[0] > 0:
            stop_at = stop_locs[0, 1].item()
            current_position += stop_at + 1
            acceptance_rate = drafts_accepted / drafts_speculated if drafts_speculated > 0 else 0
            return input_ids[0, prompt_len:current_position].tolist(), acceptance_rate, security_scores

        # Sample correction or bonus token
        if n == corrected_gamma:
            # All accepted — sample bonus token from target
            bonus_logits = Mp.logits[..., current_position + corrected_gamma - 1, :]
            bonus_probs = F.softmax(bonus_logits / temperature, dim=-1)
            x = torch.argmax(bonus_probs, dim=-1).unsqueeze(-1)
        else:
            # Some rejected — sample correction from adjusted distribution
            adjusted_probs = p[..., n, :] - q[0, n, :]
            adjusted_probs = torch.where(
                adjusted_probs > 0, adjusted_probs, torch.zeros_like(adjusted_probs)
            )
            adjusted_probs = adjusted_probs / torch.sum(adjusted_probs, dim=-1, keepdim=True)
            x = torch.argmax(adjusted_probs, dim=-1).unsqueeze(-1)

        # Clear rejected tokens and place correction/bonus
        input_ids[0, current_position + n:current_position + corrected_gamma] = pad_token_id
        input_ids[0, current_position + n] = x

        current_position += n + 1

        # Check if correction/bonus is stop token
        if torch.isin(x, stop_tokens):
            acceptance_rate = drafts_accepted / drafts_speculated if drafts_speculated > 0 else 0
            return input_ids[0, prompt_len:current_position].tolist(), acceptance_rate, security_scores

    acceptance_rate = drafts_accepted / drafts_speculated if drafts_speculated > 0 else 0
    return input_ids[0, prompt_len:current_position].tolist(), acceptance_rate, security_scores


# Test function
if __name__ == "__main__":
    import sys
    sys.path.insert(0, '../')

    from transformers import AutoTokenizer, AutoModelForCausalLM
    from sampling.security_scorer import SecurityScorer

    print("=" * 70)
    print("SECURITY-AWARE SPECULATIVE DECODING - TEST")
    print("=" * 70)

    # Load models
    print("\nLoading models...")
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B")
    tokenizer.pad_token = tokenizer.eos_token

    target = AutoModelForCausalLM.from_pretrained(
        "meta-llama/Llama-3.2-3B",
        torch_dtype=torch.float16,
        device_map="auto"
    )

    drafter = AutoModelForCausalLM.from_pretrained(
        "meta-llama/Llama-3.2-1B",
        torch_dtype=torch.float16,
        device_map="auto"
    )

    security_scorer = SecurityScorer(
        "../data/models/security_scorer/final",
        device='cuda'
    )

    print("✓ Models loaded")

    # Test prompt
    prompt = "Write a C function to copy a string:\n"
    print(f"\nPrompt: {prompt}")

    inputs = tokenizer.encode(prompt, return_tensors='pt')[0].tolist()

    # Generate
    print("\nGenerating...")
    output_ids, accept_rate, sec_scores = security_aware_speculative_generate(
        inputs=inputs,
        drafter=drafter,
        target=target,
        tokenizer=tokenizer,
        security_scorer=security_scorer,
        gamma=5,
        max_gen_len=150,
        eos_tokens_id=[tokenizer.eos_token_id],
        pad_token_id=tokenizer.pad_token_id,
        security_weight=1.0,
        security_threshold=0.5,
        temperature=0.7,
        debug=True
    )

    # Results
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)

    output_text = tokenizer.decode(output_ids, skip_special_tokens=True)
    print(f"\nGenerated code:\n{output_text}")

    print(f"\nAcceptance rate: {accept_rate:.3f}")
    print(f"Security scores: {len(sec_scores)} statements")

    if sec_scores:
        print("\nSecurity Analysis:")
        for i, s in enumerate(sec_scores, 1):
            status = "⚠️  INSECURE" if s['score'] < 0.5 else "✓  SECURE"
            print(f"  {i}. {status} ({s['score']:.4f}): {s['statement'][:60]}")

    print("=" * 70)