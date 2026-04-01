#!/usr/bin/env python3
"""
Security scorer wrapper for trained CodeBERT model
Loads model and provides simple API for scoring code snippets
"""

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np

class SecurityScorer:
    def __init__(self, model_path, device='cuda'):
        """
        Initialize security scorer with trained model
        
        Args:
            model_path: Path to trained model directory
            device: 'cuda' or 'cpu'
        """
        print(f"Loading security model from {model_path}...")
        
        self.device = device if torch.cuda.is_available() else 'cpu'
        
        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.model.to(self.device)
        self.model.eval()  # Set to evaluation mode
        
        print(f"✓ Security model loaded on {self.device}")
    
    def score_code(self, code_snippet):
        """
        Score a code snippet for security
        
        Args:
            code_snippet: String of code to evaluate
            
        Returns:
            float: Security score in [0, 1] where 1 = secure, 0 = vulnerable
        """
        # Tokenize
        inputs = self.tokenizer(
            code_snippet,
            max_length=256,
            truncation=True,
            padding='max_length',
            return_tensors='pt'
        )
        
        # Move to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Get prediction
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1)
            
            # Return P(secure) - probability of class 0
            security_score = probs[0][0].item()
        
        return security_score
    
    def score_batch(self, code_snippets):
        """
        Score multiple code snippets at once (faster)
        
        Args:
            code_snippets: List of code strings
            
        Returns:
            list: Security scores for each snippet
        """
        if not code_snippets:
            return []
        
        # Tokenize batch
        inputs = self.tokenizer(
            code_snippets,
            max_length=256,
            truncation=True,
            padding='max_length',
            return_tensors='pt'
        )
        
        # Move to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Get predictions
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1)
            
            # Return P(secure) for each example
            security_scores = probs[:, 0].cpu().numpy().tolist()
        
        return security_scores

# Test if running directly
if __name__ == "__main__":
    # Test the scorer
    scorer = SecurityScorer("../data/models/security_scorer/final")
    
    # Test with C code (what the model was trained on!)
    # INSECURE: Buffer overflow vulnerability
    insecure_c_code = """static int parse_input(char *buffer, char *input) {
    strcpy(buffer, input);
    return 0;
}"""
    
    # SECURE: Fixed buffer overflow with bounds checking
    secure_c_code = """static int parse_input(char *buffer, char *input) {
    strncpy(buffer, input, 255);
    buffer[255] = '\\0';
    return 0;
}"""
    
    print("\nTesting Security Scorer with C/C++ Code:")
    print("="*60)
    
    # Test insecure C code
    print("\n[INSECURE C CODE - strcpy buffer overflow]")
    print(insecure_c_code)
    
    inputs_insecure = scorer.tokenizer(
        insecure_c_code, max_length=256, truncation=True, 
        padding='max_length', return_tensors='pt'
    ).to(scorer.device)
    
    with torch.no_grad():
        outputs = scorer.model(**inputs_insecure)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=-1)
        
        print(f"Logits: {logits[0].cpu().numpy()}")
        print(f"P(secure): {probs[0][0].item():.6f}")
        print(f"P(vulnerable): {probs[0][1].item():.6f}")
        print(f"Predicted: {'VULNERABLE' if torch.argmax(probs[0]).item() == 1 else 'SECURE'}")
    
    # Test secure C code
    print("\n[SECURE C CODE - strncpy with bounds]")
    print(secure_c_code)
    
    inputs_secure = scorer.tokenizer(
        secure_c_code, max_length=256, truncation=True,
        padding='max_length', return_tensors='pt'
    ).to(scorer.device)
    
    with torch.no_grad():
        outputs = scorer.model(**inputs_secure)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=-1)
        
        print(f"Logits: {logits[0].cpu().numpy()}")
        print(f"P(secure): {probs[0][0].item():.6f}")
        print(f"P(vulnerable): {probs[0][1].item():.6f}")
        print(f"Predicted: {'VULNERABLE' if torch.argmax(probs[0]).item() == 1 else 'SECURE'}")
    
    print("="*60)