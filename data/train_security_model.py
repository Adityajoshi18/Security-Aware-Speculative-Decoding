#!/usr/bin/env python3
"""
Train RoBERTa security classifier - FULL DATASET
"""

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
)
from datasets import load_from_disk
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.utils.class_weight import compute_class_weight
import numpy as np

print("="*70)
print("Security Classifier Training - FULL DATASET")
print("="*70)

# Configuration
MODEL_NAME = "microsoft/codebert-base"
OUTPUT_DIR = "models/security_scorer"
MAX_LENGTH = 256
BATCH_SIZE = 16
EPOCHS = 3
LEARNING_RATE = 2e-5

# Load FULL dataset
print("\n[1/6] Loading combined dataset...")
dataset = load_from_disk('combined_dataset')

print(f"  ✓ Train: {len(dataset['train'])} examples")
print(f"  ✓ Val: {len(dataset['validation'])} examples")
print(f"  ✓ Test: {len(dataset['test'])} examples")

# Calculate class weights for imbalanced data
print("\n[2/6] Calculating class weights...")
train_labels = dataset['train']['vul']
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(train_labels),
    y=train_labels
)

print(f"  Class distribution:")
print(f"    Safe (0): {train_labels.count(0)} ({train_labels.count(0)/len(train_labels)*100:.1f}%)")
print(f"    Vulnerable (1): {train_labels.count(1)} ({train_labels.count(1)/len(train_labels)*100:.1f}%)")
print(f"  Class weights: {class_weights}")

# Load tokenizer and model  
print("\n[3/6] Loading CodeBERT model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=2,
    problem_type="single_label_classification"
)

print(f"  ✓ Model loaded")

# Preprocessing
def preprocess_function(examples):
    inputs = tokenizer(
        examples['func_before'],
        max_length=MAX_LENGTH,
        truncation=True,
        padding='max_length'
    )
    inputs['labels'] = examples['vul']
    return inputs

print("\n[4/6] Preprocessing FULL dataset...")
print("  This will take ~5-10 minutes...")

tokenized_train = dataset['train'].map(
    preprocess_function,
    batched=True,
    batch_size=1000,
    remove_columns=dataset['train'].column_names,
    desc="Tokenizing train (151k examples)"
)

tokenized_val = dataset['validation'].map(
    preprocess_function,
    batched=True,
    batch_size=1000,
    remove_columns=dataset['validation'].column_names,
    desc="Tokenizing validation"
)

tokenized_test = dataset['test'].map(
    preprocess_function,
    batched=True,
    batch_size=1000,
    remove_columns=dataset['test'].column_names,
    desc="Tokenizing test"
)

print("  ✓ Tokenization complete")

# Custom Trainer with class weights
class WeightedTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        
        # Apply class weights
        loss_fct = torch.nn.CrossEntropyLoss(
            weight=torch.tensor(class_weights, dtype=torch.float).to(model.device)
        )
        loss = loss_fct(logits, labels)
        
        return (loss, outputs) if return_outputs else loss

# Metrics
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=-1)
    
    accuracy = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average='binary', zero_division=0
    )
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

# Training arguments
print("\n[5/6] Setting up training...")
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    learning_rate=LEARNING_RATE,
    weight_decay=0.01,
    eval_strategy="steps",
    eval_steps=1000,
    save_strategy="steps",
    save_steps=1000,
    save_total_limit=3,
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    logging_steps=200,
    fp16=torch.cuda.is_available(),
    report_to="none",
    warmup_steps=500,
    gradient_accumulation_steps=2,  # Effective batch size = 32
)

print(f"  ✓ Epochs: {EPOCHS}")
print(f"  ✓ Batch size: {BATCH_SIZE} (effective: 32 with gradient accumulation)")
print(f"  ✓ Total training steps: ~{len(tokenized_train) // (BATCH_SIZE * 2) * EPOCHS}")
print(f"  ✓ Estimated time: 3-4 hours")

# Initialize trainer with class weights
trainer = WeightedTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_val,
    compute_metrics=compute_metrics,
)

# Train
print("\n[6/6] Starting FULL training...")
print("="*70)
print("Training 151,608 examples for 3 epochs...")
print("You can monitor progress or detach and check later")
print("="*70)
print()

trainer.train()

print("\n" + "="*70)
print("Training Complete!")
print("="*70)

# Evaluate on test set
print("\nEvaluating on test set (33,050 examples)...")
test_results = trainer.evaluate(tokenized_test)

print("\n" + "="*70)
print("FINAL TEST SET RESULTS")
print("="*70)
print(f"  Accuracy:  {test_results['eval_accuracy']:.4f}")
print(f"  Precision: {test_results['eval_precision']:.4f}")
print(f"  Recall:    {test_results['eval_recall']:.4f}")
print(f"  F1 Score:  {test_results['eval_f1']:.4f}")
print(f"  Loss:      {test_results['eval_loss']:.4f}")
print("="*70)

# Save final model
print("\nSaving final model...")
trainer.save_model(f"{OUTPUT_DIR}/final")
tokenizer.save_pretrained(f"{OUTPUT_DIR}/final")

print(f"\n✓ Model saved to: {OUTPUT_DIR}/final")
print("="*70)