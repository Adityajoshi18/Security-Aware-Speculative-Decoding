"""
prepare_dataset.py — Build security fragment dataset from CodeSearchNet Python

Scans 457K Python functions, extracts code fragments around security-relevant
lines, labels them insecure (1) or secure (0), and saves a balanced dataset.

Output:
    data/fragments_train.jsonl
    data/fragments_val.jsonl
    data/fragments_test.jsonl

Usage:
    python src/prepare_dataset.py
"""

import re
import json
import random
import hashlib
from pathlib import Path

import numpy as np
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm

random.seed(42)
np.random.seed(42)

OUT_DIR = Path('data')
OUT_DIR.mkdir(exist_ok=True)

# ── Security patterns ───────────────────────────────────────────────────────
# Each pattern: regex to match on a single line, label (1=insecure, 0=secure),
# and optional exclude regex (if present on that line, skip it)

PATTERNS = [
    # ── Insecure (label=1) ─────────────────────────────────────────────────

    # CWE-89: SQL injection — execute with % formatting
    {'regex': r'execute\s*\(.*%\s*\(',          'label': 1, 'exclude': None},
    # CWE-89: SQL injection — execute with string concat
    {'regex': r'execute\s*\(.*["\'].*\+',        'label': 1, 'exclude': r'%s|\?'},
    # CWE-89: SQL injection — format() in execute
    {'regex': r'execute\s*\(.*\.format\s*\(',    'label': 1, 'exclude': None},
    # CWE-89: f-string in execute
    {'regex': r'execute\s*\(\s*f["\']',          'label': 1, 'exclude': None},

    # CWE-78: Command injection
    {'regex': r'shell\s*=\s*True',               'label': 1, 'exclude': None},
    {'regex': r'os\.system\s*\(',                'label': 1, 'exclude': None},

    # CWE-502: Unsafe deserialization
    {'regex': r'pickle\.loads?\s*\(',            'label': 1, 'exclude': None},
    {'regex': r'yaml\.load\s*\(',                'label': 1, 'exclude': r'Loader\s*='},

    # CWE-798: Hardcoded credentials
    {'regex': r'(password|passwd|api_key|secret_key|token)\s*=\s*["\'][^"\'\.\\$\{]{6,}["\']',
                                                 'label': 1, 'exclude': r'os\.environ|os\.getenv|test|example|dummy|placeholder'},

    # CWE-327: Weak crypto
    {'regex': r'hashlib\.(md5|sha1)\s*\(',       'label': 1, 'exclude': r'hmac|compare_digest'},

    # CWE-22: Path traversal — open() with user input concat
    {'regex': r'open\s*\(.*\+',                  'label': 1, 'exclude': r'os\.path\.(abspath|realpath|join)'},

    # ── Secure (label=0) ───────────────────────────────────────────────────

    # CWE-89 fixed: parameterized SQL
    {'regex': r'execute\s*\(.*(%s|\?)',          'label': 0, 'exclude': None},

    # CWE-78 fixed: safe subprocess list form
    {'regex': r'subprocess\.(run|call|Popen)\s*\(\s*\[',
                                                 'label': 0, 'exclude': r'shell\s*=\s*True'},

    # CWE-502 fixed: safe deserialization
    {'regex': r'json\.loads\s*\(',               'label': 0, 'exclude': r'pickle\.loads?\s*\('},
    {'regex': r'yaml\.safe_load\s*\(',           'label': 0, 'exclude': None},

    # CWE-327 fixed: strong crypto
    {'regex': r'pbkdf2_hmac|bcrypt\.|hmac\.compare_digest|argon2|secrets\.token',
                                                 'label': 0, 'exclude': None},

    # CWE-798 fixed: env vars for secrets
    {'regex': r'os\.environ\[|os\.getenv\(',     'label': 0, 'exclude': None},

    # CWE-20 fixed: input validation
    {'regex': r'raise\s+(ValueError|TypeError)\(', 'label': 0, 'exclude': None},
    {'regex': r'isinstance\s*\(.*,\s*(int|str|list|dict|float|bool)', 'label': 0, 'exclude': None},
]

CONTEXT_LINES = 3  # lines of context before and after the matching line


def extract_security_fragments(code: str) -> list:
    """
    For each line matching a security pattern, extract that line
    plus CONTEXT_LINES before and after.
    Returns list of {'code': str, 'label': int}
    """
    lines   = code.splitlines()
    results = []
    seen    = set()

    for i, line in enumerate(lines):
        for pattern in PATTERNS:
            if not re.search(pattern['regex'], line, re.IGNORECASE):
                continue
            if pattern['exclude'] and re.search(pattern['exclude'], line, re.IGNORECASE):
                continue

            start    = max(0, i - CONTEXT_LINES)
            end      = min(len(lines), i + CONTEXT_LINES + 1)
            fragment = '\n'.join(lines[start:end]).strip()

            if len(fragment.strip()) < 30:
                continue

            real = [l for l in fragment.splitlines()
                    if l.strip() and not l.strip().startswith('#')]
            if len(real) < 2:
                continue

            h = hashlib.md5(re.sub(r'\s+', ' ', fragment).encode()).hexdigest()
            if h in seen:
                continue
            seen.add(h)

            results.append({'code': fragment, 'label': pattern['label']})
            break  # one pattern match per line is enough

    return results


def save_jsonl(data: list, path: Path):
    with open(path, 'w') as f:
        for item in data:
            f.write(json.dumps(item) + '\n')
    print(f'  Saved {len(data):,} → {path}')


def main():
    # Sanity check
    sanity_tests = [
        ("x = conn.cursor()\nresult = cursor.execute('SELECT * FROM users WHERE id = %s', (uid,))\nreturn result", 0, 'parameterized SQL'),
        ("cmd = 'ls ' + path\nresult = subprocess.run(cmd, shell=True)\nreturn result",                            1, 'shell=True'),
        ("data = request.get_data()\nobj = pickle.loads(data)\nreturn obj",                                        1, 'pickle.loads'),
        ("data = request.get_data()\nobj = json.loads(data)\nreturn obj",                                          0, 'json.loads'),
        ("token = 'abc123secretXYZ'\nheaders = {'Authorization': token}",                                          1, 'hardcoded token'),
        ("token = os.environ['API_TOKEN']\nheaders = {'Authorization': token}",                                    0, 'env var token'),
    ]

    print('Sanity check:')
    passed = 0
    for code, expected_label, desc in sanity_tests:
        frags = extract_security_fragments(code)
        got   = frags[0]['label'] if frags else None
        ok    = (got == expected_label)
        passed += ok
        print(f'  {"✓" if ok else "✗"}  label={got}  {desc}')
    print(f'{passed}/{len(sanity_tests)} correct\n')

    # Load CodeSearchNet Python
    print('Loading CodeSearchNet Python...')
    all_functions = []
    for split in ['train', 'validation', 'test']:
        ds = load_dataset('code_search_net', 'python', split=split)
        all_functions.extend(item['whole_func_string'] for item in ds)
        print(f'  {split}: {len(ds):,}')
    print(f'Total: {len(all_functions):,}\n')

    # Extract fragments
    all_fragments = []
    for code in tqdm(all_functions, desc='Scanning'):
        if code:
            all_fragments.extend(extract_security_fragments(code))

    n_secure   = sum(1 for f in all_fragments if f['label'] == 0)
    n_insecure = sum(1 for f in all_fragments if f['label'] == 1)
    print(f'\nExtracted: {len(all_fragments):,}  (secure={n_secure:,}  insecure={n_insecure:,})')

    # Dedup
    seen, deduped = set(), []
    for f in all_fragments:
        h = hashlib.md5(re.sub(r'\s+', ' ', f['code']).encode()).hexdigest()
        if h not in seen:
            seen.add(h)
            deduped.append(f)

    secure_frags   = [f for f in deduped if f['label'] == 0]
    insecure_frags = [f for f in deduped if f['label'] == 1]
    print(f'After dedup — Secure: {len(secure_frags):,}  Insecure: {len(insecure_frags):,}')

    # Balance
    n = min(len(secure_frags), len(insecure_frags))
    random.shuffle(secure_frags)
    random.shuffle(insecure_frags)
    balanced = secure_frags[:n] + insecure_frags[:n]
    random.shuffle(balanced)
    print(f'Balanced: {len(balanced):,} total  ({n:,} per class)\n')

    # Split 70/15/15
    labels = [f['label'] for f in balanced]
    idx    = list(range(len(balanced)))

    idx_train, idx_temp = train_test_split(idx, test_size=0.30, stratify=labels, random_state=42)
    labels_temp         = [labels[i] for i in idx_temp]
    idx_val, idx_test   = train_test_split(idx_temp, test_size=0.50, stratify=labels_temp, random_state=42)

    print('Saving splits...')
    save_jsonl([balanced[i] for i in idx_train], OUT_DIR / 'fragments_train.jsonl')
    save_jsonl([balanced[i] for i in idx_val],   OUT_DIR / 'fragments_val.jsonl')
    save_jsonl([balanced[i] for i in idx_test],  OUT_DIR / 'fragments_test.jsonl')

    print('\n✓ Dataset preparation complete. Run train_classifier.py next.')


if __name__ == '__main__':
    main()
