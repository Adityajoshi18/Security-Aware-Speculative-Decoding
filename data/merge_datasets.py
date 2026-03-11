#!/usr/bin/env python3
"""
Merge Big-Vul (C/C++) with Python vulnerabilities dataset
"""
from datasets import load_dataset, Dataset, DatasetDict, concatenate_datasets, Features, Value
import json

print("="*70)
print("Merging Big-Vul + Python Vulnerabilities")
print("="*70)

# Step 1: Load Big-Vul from HuggingFace
print("\n[1/4] Loading Big-Vul dataset from HuggingFace...")
bigvul = load_dataset("bstee615/bigvul")

print(f"  ✓ Train: {len(bigvul['train'])} examples")
print(f"  ✓ Val: {len(bigvul['validation'])} examples")
print(f"  ✓ Test: {len(bigvul['test'])} examples")

# Get the features schema from Big-Vul
bigvul_features = bigvul['train'].features
print(f"\n  Big-Vul schema: {bigvul_features}")

# Step 2: Load Python vulnerabilities from local JSON
print("\n[2/4] Loading Python vulnerabilities from JSON...")
with open('python_vulnerabilities.json', 'r') as f:
    python_data = json.load(f)

# Create dataset with MATCHING schema (int8 for vul, not int64)
python_dataset = Dataset.from_dict(
    {
        'CVE ID': [ex['CVE ID'] for ex in python_data],
        'CVE Page': [ex['CVE Page'] for ex in python_data],
        'CWE ID': [ex['CWE ID'] for ex in python_data],
        'codeLink': [ex['codeLink'] for ex in python_data],
        'commit_id': [ex['commit_id'] for ex in python_data],
        'commit_message': [ex['commit_message'] for ex in python_data],
        'func_after': [ex['func_after'] for ex in python_data],
        'func_before': [ex['func_before'] for ex in python_data],
        'lang': [ex['lang'] for ex in python_data],
        'project': [ex['project'] for ex in python_data],
        'vul': [ex['vul'] for ex in python_data],
    },
    features=bigvul_features  # Use Big-Vul's schema!
)

print(f"  ✓ Python: {len(python_dataset)} examples")

# Step 3: Merge datasets
print("\n[3/4] Concatenating datasets...")
combined_train = concatenate_datasets([bigvul['train'], python_dataset])

combined = DatasetDict({
    'train': combined_train,
    'validation': bigvul['validation'],
    'test': bigvul['test']
})

print(f"  ✓ Combined train: {len(combined['train'])} examples")

# Step 4: Show statistics
print("\n[4/4] Dataset Statistics")
print("="*70)

# Language distribution in training set
from collections import Counter
langs = [ex['lang'] for ex in combined['train']]
lang_counts = Counter(langs)

print("\nLanguage Distribution (Train):")
for lang, count in sorted(lang_counts.items()):
    pct = count / len(combined['train']) * 100
    print(f"  {lang}: {count} ({pct:.1f}%)")

# Vulnerability distribution
vul_count = sum([ex['vul'] for ex in combined['train']])
safe_count = len(combined['train']) - vul_count

print("\nLabel Distribution (Train):")
print(f"  Vulnerable (vul=1): {vul_count} ({vul_count/len(combined['train'])*100:.1f}%)")
print(f"  Safe (vul=0): {safe_count} ({safe_count/len(combined['train'])*100:.1f}%)")

# CWE distribution for Python examples
python_cwes = [ex['CWE ID'] for ex in combined['train'] if ex['lang'] == 'Python']
python_cwe_counts = Counter(python_cwes)

print("\nPython CWE Distribution:")
for cwe, count in sorted(python_cwe_counts.items()):
    print(f"  {cwe}: {count} examples")

print("\n" + "="*70)
print("✓ Merge Complete!")
print("="*70)
print(f"\nFinal Dataset:")
print(f"  Train: {len(combined['train'])} examples (C/C++: {len(bigvul['train'])}, Python: {len(python_dataset)})")
print(f"  Val:   {len(combined['validation'])} examples")
print(f"  Test:  {len(combined['test'])} examples")

# Save for later use
print("\n[5/5] Saving combined dataset to disk...")
combined.save_to_disk('combined_dataset')
print("  ✓ Saved to: combined_dataset/")

print("\n" + "="*70)
print("How to use this dataset:")
print("="*70)
print("\nOption 1 - Load from disk:")
print("  from datasets import load_from_disk")
print("  dataset = load_from_disk('data/combined_dataset')")
print("\nOption 2 - In this session:")
print("  # The 'combined' variable is already loaded")
print("  train_data = combined['train']")

print("\n✓ Ready for Phase 2: Training the security model!")