#!/usr/bin/env python3
"""
Compare Original vs Security-Guided Speculative Decoding

"""

import sys
sys.path.insert(0, '..')

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import time
from sampling.speculative_decoding import speculative_generate
from sampling.security_speculative import (
    security_speculative_generate, 
    build_drafter_prompt, 
    build_target_prompt
)

# Test prompts (security-sensitive scenarios)
TEST_PROMPTS = [
    "Write a Python function to query a user from a database by name",
    "Write a Python function to execute a shell command based on user input",
    "Write a Python function to read a file specified by the user",
]

def load_models():
    """Load Llama 3.2 models"""
    print("Loading models...")
    
    target_name = "meta-llama/Llama-3.2-3B-Instruct"
    drafter_name = "meta-llama/Llama-3.2-1B-Instruct"
    
    tokenizer = AutoTokenizer.from_pretrained(target_name)

    tokenizer.pad_token = tokenizer.eos_token  # Use EOS as pad token
    tokenizer.padding_side = "left"             # Pad on the left
    
    target = AutoModelForCausalLM.from_pretrained(
        target_name,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    drafter = AutoModelForCausalLM.from_pretrained(
        drafter_name,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    
    print(f"Target on: {target.device}")
    print(f"Drafter on: {drafter.device}")
    
    return tokenizer, target, drafter

def test_original_speculative(prompt, tokenizer, target, drafter):
    """Test original speculative decoding"""
    print("\n" + "="*70)
    print("ORIGINAL SPECULATIVE DECODING (λ=0.0)")
    print("="*70)
    
    # Use same prompt for both models (standard speculative)
    full_prompt = f"# Task: {prompt}\n# Solution:\n"
    inputs = tokenizer(full_prompt, return_tensors="pt").input_ids[0].tolist()
    
    start = time.time()
    output_ids, accept_rate = speculative_generate(
        inputs=inputs,
        drafter=drafter,
        target=target,
        tokenizer=tokenizer,
        gamma=4,
        max_gen_len=200,
        eos_tokens_id=[tokenizer.eos_token_id],
        pad_token_id=tokenizer.pad_token_id,
        use_cache=False,
        debug=False
    )
    elapsed = time.time() - start
    
    output = tokenizer.decode(output_ids, skip_special_tokens=True)
    tok_per_sec = len(output_ids) / elapsed if elapsed > 0 else 0
    
    print(f"Acceptance Rate: {accept_rate:.3f}")
    print(f"Tokens: {len(output_ids)}")
    print(f"Time: {elapsed:.1f}s")
    print(f"Speed: {tok_per_sec:.1f} tok/s")
    print(f"\nGenerated Code:\n{output}")
    
    return {
        'method': 'Original',
        'lambda': 0.0,
        'accept_rate': accept_rate,
        'tokens': len(output_ids),
        'time': elapsed,
        'tok_per_sec': tok_per_sec,
        'output': output
    }

def test_security_guided(prompt, tokenizer, target, drafter, security_lambda):
    """Test security-guided speculative decoding"""
    print("\n" + "="*70)
    print(f"SECURITY-GUIDED SPECULATIVE DECODING (λ={security_lambda})")
    print("="*70)
    
    # Different prompts for target vs drafter
    drafter_text = build_drafter_prompt(prompt, tokenizer)
    drafter_tokens = tokenizer(drafter_text, return_tensors="pt").input_ids[0].tolist()
    
    target_text = build_target_prompt(prompt)
    target_tokens = tokenizer(target_text, return_tensors="pt").input_ids[0].tolist()
    
    start = time.time()
    output_ids, accept_rate = security_speculative_generate(
        target_inputs=target_tokens,
        drafter_inputs=drafter_tokens,
        drafter=drafter,
        target=target,
        gamma=4,
        max_gen_len=200,
        eos_tokens_id=[tokenizer.eos_token_id],
        pad_token_id=tokenizer.pad_token_id,
        security_lambda=security_lambda,
    )
    elapsed = time.time() - start
    
    output = tokenizer.decode(output_ids, skip_special_tokens=True)
    tok_per_sec = len(output_ids) / elapsed if elapsed > 0 else 0
    
    print(f"Acceptance Rate: {accept_rate:.3f}")
    print(f"Tokens: {len(output_ids)}")
    print(f"Time: {elapsed:.1f}s")
    print(f"Speed: {tok_per_sec:.1f} tok/s")
    print(f"\nGenerated Code:\n{output}")
    
    return {
        'method': 'Security-Guided',
        'lambda': security_lambda,
        'accept_rate': accept_rate,
        'tokens': len(output_ids),
        'time': elapsed,
        'tok_per_sec': tok_per_sec,
        'output': output
    }

def main():
    print("="*70)
    print("SPECULATIVE DECODING COMPARISON")
    print("="*70)
    
    tokenizer, target, drafter = load_models()
    
    all_results = []
    
    for prompt in TEST_PROMPTS:
        print(f"\n{'#'*70}")
        print(f"# PROMPT: {prompt}")
        print(f"{'#'*70}")
        
        # Test 1: Original speculative (λ=0.0)
        result_original = test_original_speculative(prompt, tokenizer, target, drafter)
        all_results.append(result_original)
        
        # Test 2: Security-guided with λ=0.5
        result_security_05 = test_security_guided(prompt, tokenizer, target, drafter, security_lambda=0.5)
        all_results.append(result_security_05)
        
        # Test 3: Security-guided with λ=1.0
        result_security_10 = test_security_guided(prompt, tokenizer, target, drafter, security_lambda=1.0)
        all_results.append(result_security_10)
    
    # Summary table
    print("\n" + "="*90)
    print("SUMMARY")
    print("="*90)
    print(f"{'Prompt':<50} {'Method':<20} {'λ':<6} {'Accept':<8} {'Tok/s':<8}")
    print("-"*90)
    
    for i, prompt in enumerate(TEST_PROMPTS):
        for j in range(3):  # 3 tests per prompt
            result = all_results[i*3 + j]
            prompt_short = prompt[:47] + "..." if len(prompt) > 50 else prompt
            print(f"{prompt_short:<50} {result['method']:<20} {result['lambda']:<6.1f} {result['accept_rate']:<8.3f} {result['tok_per_sec']:<8.1f}")
    
    print("="*90)

if __name__ == "__main__":
    main()