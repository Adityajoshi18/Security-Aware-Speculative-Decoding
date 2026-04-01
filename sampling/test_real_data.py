#!/usr/bin/env python3
"""
Test security scorer on actual examples from the dataset
"""

import sys
sys.path.append('.')
from security_scorer import SecurityScorer
from datasets import load_from_disk

print("Loading security model...")
scorer = SecurityScorer("../data/models/security_scorer/final")

print("Loading dataset...")
dataset = load_from_disk('../data/combined_dataset')

print("\n" + "="*70)
print("Testing on REAL dataset examples")
print("="*70)

# Get some vulnerable and safe examples
test_data = dataset['test'].select(range(100))

# Find vulnerable examples
vulnerable_examples = [ex for ex in test_data if ex['vul'] == 1][:5]
safe_examples = [ex for ex in test_data if ex['vul'] == 0][:5]

print(f"\nFound {len(vulnerable_examples)} vulnerable and {len(safe_examples)} safe examples")

# Test vulnerable examples
print("\n" + "="*70)
print("VULNERABLE EXAMPLES (should score LOW)")
print("="*70)

for i, ex in enumerate(vulnerable_examples, 1):
    code = ex['func_before'][:200]  # First 200 chars
    score = scorer.score_code(ex['func_before'])
    
    print(f"\n[Example {i}]")
    print(f"CWE: {ex.get('CWE ID', 'N/A')}")
    print(f"Code preview: {code}...")
    print(f"Security score: {score:.4f}")
    print(f"Prediction: {'SECURE' if score > 0.5 else 'VULNERABLE'}")
    print(f"Actual label: VULNERABLE (vul=1)")
    print(f"✓ CORRECT" if score < 0.5 else "✗ WRONG")

# Test safe examples  
print("\n" + "="*70)
print("SAFE EXAMPLES (should score HIGH)")
print("="*70)

for i, ex in enumerate(safe_examples, 1):
    code = ex['func_before'][:200]
    score = scorer.score_code(ex['func_before'])
    
    print(f"\n[Example {i}]")
    print(f"CWE: {ex.get('CWE ID', 'N/A')}")
    print(f"Code preview: {code}...")
    print(f"Security score: {score:.4f}")
    print(f"Prediction: {'SECURE' if score > 0.5 else 'VULNERABLE'}")
    print(f"Actual label: SAFE (vul=0)")
    print(f"✓ CORRECT" if score > 0.5 else "✗ WRONG")

print("\n" + "="*70)