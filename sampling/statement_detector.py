#!/usr/bin/env python3
"""
Detects statement boundaries in generated code
"""

import re

class StatementDetector:
    def __init__(self, language='auto'):
        """
        Args:
            language: 'c', 'cpp', 'python', or 'auto' (detect from syntax)
        """
        self.language = language
        self.last_check_length = 0
    
    def detect_language(self, code):
        """Auto-detect language from code syntax"""
        # Strong Python indicators
        if any(keyword in code for keyword in ['def ', 'import ', 'class ', 'print(', 'if __name__']):
            return 'python'
        
        # Strong C/C++ indicators  
        if any(keyword in code for keyword in ['#include', 'int main', 'void ', 'printf(', 'struct ']):
            return 'c'
        
        # Check for Python-style assignment without semicolon
        if '=' in code and '\n' in code and ';' not in code:
            return 'python'
        
        # Default to C
        return 'c'
    
    def has_new_statement(self, code):
        """
        Check if a new complete statement was added since last check
        
        Returns:
            (bool, str): (has_new_statement, last_statement)
        """
        # Only check new text since last call
        new_text = code[self.last_check_length:]
        
        if not new_text.strip():
            return False, ""
        
        # Auto-detect language if needed
        lang = self.language
        if lang == 'auto':
            lang = self.detect_language(code)
        
        # Check for statement boundaries
        if lang == 'python':
            # Python: newline that's not a continuation
            lines = code.split('\n')
            
            # Check if we have at least 2 lines (one complete, one incomplete)
            if len(lines) >= 2:
                # Check the line that just became complete (last line that had a \n)
                for i in range(len(lines) - 1, 0, -1):
                    line = lines[i-1].strip()
                    if line and not line.endswith('\\'):
                        # Found a complete line
                        if len(code) > self.last_check_length:
                            self.last_check_length = len(code)
                            return True, line
        
        else:  # C/C++
            # C: semicolon or closing brace
            if ';' in new_text or '}' in new_text:
                # Extract last complete statement
                statements = re.split(r'[;{}]', code)
                if len(statements) >= 2:
                    last_complete = statements[-2].strip()
                    if last_complete:
                        self.last_check_length = len(code)
                        return True, last_complete
        
        return False, ""
    
    def reset(self):
        """Reset for new generation"""
        self.last_check_length = 0

# Quick test
if __name__ == "__main__":
    detector = StatementDetector()
    
    # Test C code
    print("Testing C code:")
    c_code = "int x = 5"
    print(f"  '{c_code}' -> {detector.has_new_statement(c_code)}")
    
    c_code += ";"
    print(f"  '{c_code}' -> {detector.has_new_statement(c_code)}")
    
    # Test Python code
    detector.reset()
    print("\nTesting Python code:")
    py_code = "x = 5"
    print(f"  '{py_code}' -> {detector.has_new_statement(py_code)}")
    
    py_code += "\n"
    print(f"  '{py_code}' -> {detector.has_new_statement(py_code)}")