"""
train_classifier.py — Fine-tune CodeBERT as a security fragment classifier

Trains a binary classifier (secure=0 / insecure=1) on the fragment dataset
produced by prepare_dataset.py. Saves model, tokenizer, and metadata.

Output:
    models/fragment_classifier/best_model.pt
    models/fragment_classifier/metadata.json
    models/fragment_classifier/tokenizer files

Usage:
    python src/train_classifier.py
"""

import json
import random
import numpy as np
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
from tqdm.auto import tqdm

# ── Config ──────────────────────────────────────────────────────────────────
SEED          = 42
DATA_DIR      = Path('data')
MODEL_DIR     = Path('models/fragment_classifier')
MODEL_DIR.mkdir(parents=True, exist_ok=True)

BASE_MODEL    = 'microsoft/codebert-base'
MAX_LENGTH    = 256
BATCH_SIZE    = 32
LEARNING_RATE = 2e-5
EPOCHS        = 5
WARMUP_RATIO  = 0.1
WEIGHT_DECAY  = 0.01
GRAD_CLIP     = 1.0
EVAL_STEPS    = 50

# ── Reproducibility ─────────────────────────────────────────────────────────
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ── Dataset ─────────────────────────────────────────────────────────────────

def load_jsonl(path):
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


class FragmentDataset(Dataset):
    def __init__(self, data, tokenizer, max_length):
        self.data      = data
        self.tokenizer = tokenizer
        self.max_len   = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        enc  = self.tokenizer(
            item['code'], max_length=self.max_len,
            padding='max_length', truncation=True, return_tensors='pt'
        )
        return {
            'input_ids':      enc['input_ids'].squeeze(0),
            'attention_mask': enc['attention_mask'].squeeze(0),
            'label':          torch.tensor(item['label'], dtype=torch.long),
        }


# ── Model ────────────────────────────────────────────────────────────────────

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


# ── Evaluation ───────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    all_preds, all_labels, all_probs = [], [], []
    total_loss = 0.0

    for batch in loader:
        ids    = batch['input_ids'].to(device)
        mask   = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)
        logits = model(ids, mask)
        total_loss += criterion(logits, labels).item()
        all_probs.extend(F.softmax(logits, dim=-1)[:, 1].cpu().numpy())
        all_preds.extend(logits.argmax(dim=-1).cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    acc         = accuracy_score(all_labels, all_preds)
    _, _, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='binary', zero_division=0)
    try:    auc = roc_auc_score(all_labels, all_probs)
    except: auc = 0.0

    return {'loss': total_loss / len(loader), 'acc': acc, 'f1': f1, 'auc': auc}


def main():
    print(f'Device: {DEVICE}')
    if DEVICE.type == 'cuda':
        print(f'GPU:  {torch.cuda.get_device_name(0)}')

    # Load data
    print('\nLoading data...')
    train_data = load_jsonl(DATA_DIR / 'fragments_train.jsonl')
    val_data   = load_jsonl(DATA_DIR / 'fragments_val.jsonl')
    test_data  = load_jsonl(DATA_DIR / 'fragments_test.jsonl')
    print(f'Train: {len(train_data):,}  Val: {len(val_data):,}  Test: {len(test_data):,}')

    # Tokenizer and dataloaders
    print('\nLoading tokenizer...')
    tokenizer    = AutoTokenizer.from_pretrained(BASE_MODEL)
    train_loader = DataLoader(FragmentDataset(train_data, tokenizer, MAX_LENGTH),
                              batch_size=BATCH_SIZE, shuffle=True,  num_workers=4, pin_memory=True)
    val_loader   = DataLoader(FragmentDataset(val_data,   tokenizer, MAX_LENGTH),
                              batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    test_loader  = DataLoader(FragmentDataset(test_data,  tokenizer, MAX_LENGTH),
                              batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    print(f'Steps per epoch: {len(train_loader):,}')

    # Model
    print('\nLoading CodeBERT...')
    model = FragmentSecurityClassifier(BASE_MODEL).to(DEVICE)
    print(f'Parameters: {sum(p.numel() for p in model.parameters()):,}')

    # Optimizer and scheduler
    optimizer    = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    total_steps  = len(train_loader) * EPOCHS
    warmup_steps = int(total_steps * WARMUP_RATIO)
    scheduler    = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    criterion    = nn.CrossEntropyLoss()
    print(f'Total steps: {total_steps:,}  Warmup: {warmup_steps:,}\n')

    # Training loop
    best_val_loss = float('inf')
    history       = []
    global_step   = 0

    for epoch in range(1, EPOCHS + 1):
        model.train()
        epoch_loss = 0.0
        pbar = tqdm(train_loader, desc=f'Epoch {epoch}/{EPOCHS}')

        for batch in pbar:
            ids    = batch['input_ids'].to(DEVICE)
            mask   = batch['attention_mask'].to(DEVICE)
            labels = batch['label'].to(DEVICE)

            optimizer.zero_grad()
            loss = criterion(model(ids, mask), labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()
            scheduler.step()

            epoch_loss  += loss.item()
            global_step += 1
            pbar.set_postfix({'loss': f'{loss.item():.4f}', 'step': global_step})

            if global_step % EVAL_STEPS == 0:
                m = evaluate(model, val_loader, criterion, DEVICE)
                print(f'\n  [step {global_step}]  val_loss={m["loss"]:.4f}  val_acc={m["acc"]:.4f}  val_auc={m["auc"]:.4f}')
                model.train()

        val_m     = evaluate(model, val_loader, criterion, DEVICE)
        avg_train = epoch_loss / len(train_loader)
        history.append({'epoch': epoch, 'train_loss': avg_train, **{f'val_{k}': v for k, v in val_m.items()}})

        print(f'\n{"-"*65}')
        print(f'Epoch {epoch}  train={avg_train:.4f}  val_loss={val_m["loss"]:.4f}  val_acc={val_m["acc"]:.4f}  val_auc={val_m["auc"]:.4f}')
        print(f'{"-"*65}')

        if val_m['loss'] < best_val_loss:
            best_val_loss = val_m['loss']
            torch.save(model.state_dict(), MODEL_DIR / 'best_model.pt')
            print(f'  ✓ Best model saved (val_loss={best_val_loss:.4f})')

    print('\n✓ Training complete!')

    # Test set evaluation
    model.load_state_dict(torch.load(MODEL_DIR / 'best_model.pt', map_location=DEVICE, weights_only=True))
    model.eval()
    test_m = evaluate(model, test_loader, criterion, DEVICE)
    print(f'\n=== TEST SET RESULTS ===')
    print(f'  Accuracy: {test_m["acc"]:.4f}')
    print(f'  F1:       {test_m["f1"]:.4f}')
    print(f'  AUC-ROC:  {test_m["auc"]:.4f}')

    # Temperature calibration
    all_logits_list, all_labels_list = [], []
    with torch.no_grad():
        for batch in val_loader:
            ids  = batch['input_ids'].to(DEVICE)
            mask = batch['attention_mask'].to(DEVICE)
            all_logits_list.append(model(ids, mask).cpu())
            all_labels_list.append(batch['label'])

    all_logits = torch.cat(all_logits_list)
    all_labels = torch.cat(all_labels_list)

    T_param   = nn.Parameter(torch.ones(1) * 1.5)
    cal_optim = torch.optim.LBFGS([T_param], lr=0.01, max_iter=200)
    def cal_step():
        cal_optim.zero_grad()
        loss = nn.CrossEntropyLoss()(all_logits / T_param, all_labels)
        loss.backward()
        return loss
    cal_optim.step(cal_step)
    T = T_param.item()
    print(f'\nCalibration temperature T = {T:.4f}')

    # Save
    tokenizer.save_pretrained(MODEL_DIR)
    with open(MODEL_DIR / 'metadata.json', 'w') as f:
        json.dump({
            'base_model':       BASE_MODEL,
            'max_length':       MAX_LENGTH,
            'temperature':      T,
            'test_accuracy':    float(test_m['acc']),
            'test_f1':          float(test_m['f1']),
            'test_auc':         float(test_m['auc']),
            'best_val_loss':    float(best_val_loss),
            'training_history': history,
        }, f, indent=2)

    print(f'\nSaved to {MODEL_DIR}:')
    for p in sorted(MODEL_DIR.iterdir()):
        print(f'  {p.name}  ({p.stat().st_size/1024:.0f} KB)')

    # Sanity check
    @torch.no_grad()
    def score(code):
        enc = tokenizer(code, max_length=MAX_LENGTH, padding='max_length',
                        truncation=True, return_tensors='pt')
        logits = model(enc['input_ids'].to(DEVICE), enc['attention_mask'].to(DEVICE))
        return F.softmax(logits / T, dim=-1)[0, 0].item()

    tests = [
        ("cursor.execute('SELECT * FROM users WHERE id = ' + uid)",     'INSECURE'),
        ("cursor.execute('SELECT * FROM users WHERE id = %s', (uid,))", 'SECURE'),
        ("subprocess.run('ls ' + path, shell=True)",                    'INSECURE'),
        ("subprocess.run(['ls', path], capture_output=True)",           'SECURE'),
        ("pickle.loads(data)",                                          'INSECURE'),
        ("json.loads(data)",                                            'SECURE'),
        ("password = 'mysecretpassword'",                               'INSECURE'),
        ("password = os.environ['DB_PASSWORD']",                        'SECURE'),
    ]

    print('\n=== Sanity Check ===')
    correct = 0
    for code, expected in tests:
        s     = score(code)
        label = 'SECURE' if s >= 0.5 else 'INSECURE'
        ok    = (label == expected)
        correct += ok
        print(f'  {"✓" if ok else "✗"}  [{s:.3f}]  {label:<10}  {code[:55]}')
    print(f'\n{correct}/{len(tests)} correct')
    print('\n✓ Done. Run evaluate.py next.')


if __name__ == '__main__':
    main()
