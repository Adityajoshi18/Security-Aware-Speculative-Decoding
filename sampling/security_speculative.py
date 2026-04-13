import torch
import torch.nn.functional as F
from typing import List, Tuple

# Security system prompt (same as your teammate's)
SECURITY_SYSTEM_PROMPT = (
    "You are a security-focused code assistant. Always generate code that "
    "follows security best practices:\n"
    "- Use parameterized queries for database operations\n"
    "- Use subprocess with argument lists instead of os.system\n"
    "- Validate and sanitize all user inputs\n"
    "- Avoid eval(), exec(), and dynamic code execution\n"
    "- Use safe file handling with path validation\n"
    "- Use proper cryptographic libraries\n"
    "- Handle errors without exposing sensitive information"
)

def build_drafter_prompt(user_prompt, tokenizer):
    """Wrap user prompt with security system prompt using chat template."""
    messages = [
        {"role": "system", "content": SECURITY_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]
    return tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)

def build_target_prompt(user_prompt):
    """Plain code-completion prompt for the base model."""
    return f"# Task: {user_prompt}\n# Solution:\n"

@torch.no_grad()
def security_speculative_generate(
    target_inputs: List[int],
    drafter_inputs: List[int],
    drafter,
    target,
    gamma: int = 5,
    max_gen_len: int = 200,
    eos_tokens_id: List[int] = [1],
    pad_token_id: int = 0,
    security_lambda: float = 0.0,
) -> Tuple[List[int], float]:
    """
    Security-guided speculative decoding.
    
    Args:
        target_inputs: tokenized prompt for target (base model)
        drafter_inputs: tokenized prompt for drafter (instruct + security prompt)
        security_lambda: blending weight (0 = pure target, 1 = pure drafter)
    
    Returns:
        (generated_token_ids, acceptance_rate)
    """
    assert target.config.vocab_size == drafter.config.vocab_size
    vocab_size = target.config.vocab_size
    device = target.device
    
    stop_tokens = torch.tensor(eos_tokens_id, dtype=torch.long, device=device).unsqueeze(1)
    drafter_cache, target_cache = None, None
    drafts_accepted, drafts_speculated = 0.0, 0.0
    
    # Setup dual input buffers
    target_prompt_len = len(target_inputs)
    drafter_prompt_len = len(drafter_inputs)
    target_max = getattr(target.config, 'max_position_embeddings', 4096)
    drafter_max = getattr(drafter.config, 'max_position_embeddings', 4096)
    effective_max_gen = min(max_gen_len, target_max - target_prompt_len, drafter_max - drafter_prompt_len)
    
    target_ids = torch.full((1, target_prompt_len + effective_max_gen), pad_token_id, dtype=torch.long, device=device)
    target_ids[0, :target_prompt_len] = torch.tensor(target_inputs, dtype=torch.long, device=device)
    drafter_ids = torch.full((1, drafter_prompt_len + effective_max_gen), pad_token_id, dtype=torch.long, device=device)
    drafter_ids[0, :drafter_prompt_len] = torch.tensor(drafter_inputs, dtype=torch.long, device=device)
    
    gen_pos = 0
    
    # Prefill both models, blend first token
    Mp = target(input_ids=target_ids[..., :target_prompt_len], past_key_values=target_cache, use_cache=False)
    target_cache = Mp.past_key_values
    target_logits_0 = Mp.logits[..., -1, :]
    
    Mq = drafter(input_ids=drafter_ids[..., :drafter_prompt_len], past_key_values=drafter_cache, use_cache=False)
    drafter_cache = Mq.past_key_values
    drafter_logits_0 = Mq.logits[..., -1, :].to(device)
    
    blended = (1 - security_lambda) * target_logits_0 + security_lambda * drafter_logits_0 if security_lambda > 0 else target_logits_0
    t = torch.argmax(F.softmax(blended, dim=-1), dim=-1).unsqueeze(-1)
    
    target_ids[0, target_prompt_len] = t
    drafter_ids[0, drafter_prompt_len] = t
    gen_pos = 1
    
    if torch.isin(t, stop_tokens):
        return target_ids[0, target_prompt_len:target_prompt_len + 1].tolist(), 0.0
    
    # Main loop
    while gen_pos < effective_max_gen:
        t_pos = target_prompt_len + gen_pos
        d_pos = drafter_prompt_len + gen_pos
        corrected_gamma = min(gamma, effective_max_gen - gen_pos - 1)
        if corrected_gamma <= 0:
            break
        
        q = torch.zeros((1, corrected_gamma, vocab_size), device=device)
        drafter_raw = torch.zeros((1, corrected_gamma, vocab_size), device=device)
        
        # Draft gamma tokens
        for k in range(corrected_gamma):
            Mq = drafter(input_ids=drafter_ids[..., :d_pos + k], past_key_values=drafter_cache, use_cache=False)
            drafter_cache = Mq.past_key_values
            raw_k = Mq.logits[..., -1, :]
            drafter_raw[0, k] = raw_k.to(device)
            probs_k = F.softmax(raw_k, dim=-1)
            q[0, k] = probs_k.to(device)
            xi = torch.argmax(probs_k, dim=-1).unsqueeze(-1)
            target_ids[0, t_pos + k] = xi.to(device)
            drafter_ids[0, d_pos + k] = xi.to(device)
        
        drafts_speculated += corrected_gamma
        
        # Verify with target, blend logits
        Mp = target(input_ids=target_ids[..., :t_pos + corrected_gamma], past_key_values=target_cache, use_cache=False)
        target_cache = Mp.past_key_values
        target_raw = Mp.logits[..., t_pos - 1:t_pos + corrected_gamma - 1, :]
        
        # BLEND (the key modification!)
        if security_lambda > 0:
            blended = (1 - security_lambda) * target_raw + security_lambda * drafter_raw
        else:
            blended = target_raw
        p = F.softmax(blended, dim=-1)
        
        # Rejection sampling
        r = torch.rand(corrected_gamma, device=device)
        n = corrected_gamma
        for i in range(corrected_gamma):
            token_i = target_ids[0, t_pos + i]
            if r[i] > p[0, i, token_i] / q[0, i, token_i]:
                n = i
                break
        drafts_accepted += n
        
        # Check for stop tokens
        stop_locs = torch.nonzero(torch.eq(target_ids[..., t_pos:t_pos + n], stop_tokens))
        if stop_locs.shape[0] > 0:
            stop_at = stop_locs[0, 1].item()
            gen_pos += stop_at + 1
            rate = drafts_accepted / drafts_speculated if drafts_speculated > 0 else 0
            return target_ids[0, target_prompt_len:target_prompt_len + gen_pos].tolist(), rate
        
        # Sample correction / bonus token
        if n == corrected_gamma:
            bonus = F.softmax(Mp.logits[..., t_pos + corrected_gamma - 1, :], dim=-1)
            x = torch.argmax(bonus, dim=-1).unsqueeze(-1)
        else:
            # Adjusted distribution when rejected
            adjusted_probs = p[..., n, :] - q[0, n, :]
            adjusted_probs = torch.where(adjusted_probs > 0, adjusted_probs, torch.zeros_like(adjusted_probs))
            adjusted_probs = adjusted_probs / torch.sum(adjusted_probs, dim=-1, keepdim=True)
            x = torch.argmax(adjusted_probs, dim=-1).unsqueeze(-1)
        
        # Write correction token, clear rejected positions
        for off in range(n, corrected_gamma):
            target_ids[0, t_pos + off] = pad_token_id
            drafter_ids[0, d_pos + off] = pad_token_id
        target_ids[0, t_pos + n] = x.to(device)
        drafter_ids[0, d_pos + n] = x.to(device)
        
        gen_pos += n + 1
        if torch.isin(x, stop_tokens):
            rate = drafts_accepted / drafts_speculated if drafts_speculated > 0 else 0
            return target_ids[0, target_prompt_len:target_prompt_len + gen_pos].tolist(), rate
    
    rate = drafts_accepted / drafts_speculated if drafts_speculated > 0 else 0
    return target_ids[0, target_prompt_len:target_prompt_len + gen_pos].tolist(), rate