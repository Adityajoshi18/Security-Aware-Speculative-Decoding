#!/usr/bin/env python3
"""
Analyze Big-Vul dataset structure and content.
"""
from datasets import load_dataset
from collections import Counter

print("Loading Big-Vul dataset...")
dataset = load_dataset("bstee615/bigvul")

print(f"✓ Dataset loaded!")
print(f"  Train: {len(dataset['train'])} examples")
print(f"  Val: {len(dataset['validation'])} examples")
print(f"  Test: {len(dataset['test'])} examples")

# ============================================================
# CHECK 1: What Vulnerability Types (CWEs) Are Present?
# ============================================================
print("\n" + "="*60)
print("CHECK 1: Top 20 Most Common CWE Types in Big-Vul")
print("="*60)

cwe_ids = [ex['CWE ID'] for ex in dataset['train']]
cwe_counts = Counter(cwe_ids)

for cwe, count in cwe_counts.most_common(20):
    print(f"{cwe}: {count} examples")

print(f"\nTotal unique CWE types: {len(cwe_counts)}")

# ============================================================
# CHECK 2: See Complete Example Structure (VULNERABLE)
# ============================================================
print("\n" + "="*60)
print("CHECK 2: Complete VULNERABLE Example Structure")
print("="*60)

vuln_ex = next(ex for ex in dataset['train'] if ex['vul'] == 1)

print("\n[VULNERABLE EXAMPLE - vul=1]")
for key, value in vuln_ex.items():
    print(f"\n--- {key} ---")
    if isinstance(value, str):
        # Show first 400 chars for code
        if 'func' in key:
            print(value[:400] + ("..." if len(value) > 400 else ""))
        else:
            print(value[:200] + ("..." if len(value) > 200 else ""))
    else:
        print(value)

# ============================================================
# CHECK 3: See a Safe (Non-Vulnerable) Example
# ============================================================
print("\n" + "="*60)
print("CHECK 3: Complete SAFE Example Structure")
print("="*60)

safe_ex = next(ex for ex in dataset['train'] if ex['vul'] == 0)

print("\n[SAFE EXAMPLE - vul=0]")
for key, value in safe_ex.items():
    print(f"\n--- {key} ---")
    if isinstance(value, str):
        if 'func' in key:
            print(value[:400] + ("..." if len(value) > 400 else ""))
        else:
            print(value[:200] + ("..." if len(value) > 200 else ""))
    else:
        print(value)

# ============================================================
# CHECK 4: See Exact Field Names and Types
# ============================================================
print("\n" + "="*60)
print("CHECK 4: Dataset Schema (Field Names and Types)")
print("="*60)

example = dataset['train'][0]
print("\nAll fields in dataset:")
for key in example.keys():
    value = example[key]
    value_type = type(value).__name__
    
    # Show sample value
    if isinstance(value, str):
        sample = value[:50] + "..." if len(value) > 50 else value
        print(f"  {key:20s} ({value_type:10s}) = {sample}")
    else:
        print(f"  {key:20s} ({value_type:10s}) = {value}")

print("\n" + "="*60)
print("Analysis Complete!")
print("="*60)