#!/usr/bin/env python3
"""
Detects statement boundaries in generated code
"""

import re


def strip_comments_and_docstrings(code: str, lang: str) -> str:
    """
    Remove comments and docstrings from a code snippet before security scoring.
    Returns the cleaned code, or empty string if nothing substantive remains.
    """
    lang_lower = (lang or '').lower()

    if 'python' in lang_lower:
        # Remove triple-quoted docstrings (both ''' and \""")
        code = re.sub(r'("""[\s\S]*?"""|\'\'\'[\s\S]*?\'\'\')', '', code)
        # Remove single-line # comments
        code = re.sub(r'#.*', '', code)
    else:
        # C/C++: remove block comments /* ... */
        code = re.sub(r'/\*[\s\S]*?\*/', '', code)
        # Remove single-line // comments
        code = re.sub(r'//.*', '', code)

    return code.strip()


def is_scoreable_statement(statement: str, lang: str) -> bool:
    """
    Returns False if the statement is purely a comment, docstring,
    or contains no actual code tokens worth scoring.
    """
    cleaned = strip_comments_and_docstrings(statement, lang)

    # Nothing left after stripping
    if not cleaned:
        return False

    # Only punctuation/braces left (e.g. statement was just "{" or "}")
    if re.fullmatch(r'[\s\{\}\(\);,]*', cleaned):
        return False

    return True


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
        Check if a new complete statement was added since last check.

        Returns:
            (bool, str): (has_new_statement, last_statement)
            The returned statement is already stripped of comments/docstrings
            so it is safe to pass directly to the security scorer.
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
            lines = code.split('\n')

            if len(lines) >= 2:
                for i in range(len(lines) - 1, 0, -1):
                    line = lines[i - 1].strip()
                    if line and not line.endswith('\\'):
                        if len(code) > self.last_check_length:
                            self.last_check_length = len(code)
                            # Strip comments before returning
                            clean = strip_comments_and_docstrings(line, lang)
                            if not clean:
                                return False, ""
                            return True, clean

        else:  # C/C++
            if ';' in new_text or '}' in new_text:
                statements = re.split(r'[;{}]', code)
                if len(statements) >= 2:
                    last_complete = statements[-2].strip()
                    if last_complete:
                        self.last_check_length = len(code)
                        # Strip comments before returning
                        clean = strip_comments_and_docstrings(last_complete, lang)
                        if not clean:
                            return False, ""
                        return True, clean

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

    # Test comment-only statement (should be filtered)
    detector.reset()
    print("\nTesting comment filtering (C):")
    comment_code = "// Allocate memory for the cluster\n pCluster = malloc(sizeof(Cluster));"
    result = detector.has_new_statement(comment_code)
    print(f"  Comment+code -> scoreable={result[0]}, cleaned='{result[1]}'")

    detector.reset()
    pure_comment = "// This is just a comment;"
    result = detector.has_new_statement(pure_comment)
    print(f"  Pure comment -> scoreable={result[0]}, statement='{result[1]}'")

    # Test Python code
    detector.reset()
    print("\nTesting Python code:")
    py_code = "x = 5"
    print(f"  '{py_code}' -> {detector.has_new_statement(py_code)}")

    py_code += "\n"
    print(f"  '{py_code}' -> {detector.has_new_statement(py_code)}")

    # Test Python comment filtering
    detector.reset()
    print("\nTesting comment filtering (Python):")
    py_comment = "# This sets up the database\nx = get_db()\n"
    detector.has_new_statement("# This sets up the database\n")
    result = detector.has_new_statement(py_comment)
    print(f"  Comment line -> scoreable={result[0]}, cleaned='{result[1]}'")

    # Test is_scoreable_statement directly
    print("\nTesting is_scoreable_statement:")
    cases = [
        ("// Allocate memory", "c", False),
        ("/* block comment */", "c", False),
        ("pCluster = malloc(sizeof(Cluster))", "c", True),
        ("// comment\n pCluster = malloc(sizeof(Cluster))", "c", True),
        ("{", "c", False),
        ("# python comment", "python", False),
        ("x = get_db()", "python", True),
    ]
    for stmt, lang, expected in cases:
        result = is_scoreable_statement(stmt, lang)
        status = "✓" if result == expected else "✗"
        print(f"  {status} is_scoreable('{stmt[:40]}', {lang}) = {result} (expected {expected})")