from sampling.statement_detector import is_scoreable_statement, strip_comments_and_docstrings

print("=== Should be False (filtered) ===")
print(is_scoreable_statement('// Allocate memory for the cluster', 'c'))
print(is_scoreable_statement('/* block comment */', 'c'))
print(is_scoreable_statement('{', 'c'))
print(is_scoreable_statement('# python comment', 'python'))

print("\n=== Should be True (real code) ===")
print(is_scoreable_statement('pCluster = malloc(sizeof(Cluster))', 'c'))
print(is_scoreable_statement('if (!compositor)\n    return ret', 'c'))
print(is_scoreable_statement('x = get_db()', 'python'))

print("\n=== Mixed comment + code (should strip comment) ===")
cleaned = strip_comments_and_docstrings('// Allocate memory\n pCluster = malloc(sizeof(Cluster))', 'c')
print(repr(cleaned))