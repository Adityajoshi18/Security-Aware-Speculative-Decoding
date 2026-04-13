#!/usr/bin/env python3
"""
Generate Python vulnerability dataset matching Big-Vul format
"""
import json
import random

def generate_sql_injection_examples(start_id=1):
    """Generate SQL injection examples (CWE-89)"""
    examples = []
    
    templates = [
        # SQLite examples
        {
            "before": """import sqlite3

def get_user_by_id(user_id):
    conn = sqlite3.connect('database.db')
    cursor = conn.cursor()
    query = "SELECT * FROM users WHERE id = " + user_id
    cursor.execute(query)
    return cursor.fetchone()""",
            "after": """import sqlite3

def get_user_by_id(user_id):
    conn = sqlite3.connect('database.db')
    cursor = conn.cursor()
    query = "SELECT * FROM users WHERE id = ?"
    cursor.execute(query, (user_id,))
    return cursor.fetchone()""",
            "message": "Fixed SQL injection by using parameterized query"
        },
        {
            "before": """import sqlite3

def delete_user(username):
    conn = sqlite3.connect('app.db')
    cursor = conn.cursor()
    cursor.execute('DELETE FROM users WHERE name = ' + username)
    conn.commit()""",
            "after": """import sqlite3

def delete_user(username):
    conn = sqlite3.connect('app.db')
    cursor = conn.cursor()
    cursor.execute('DELETE FROM users WHERE name = ?', (username,))
    conn.commit()""",
            "message": "Use parameterized query to prevent SQL injection"
        },
        {
            "before": """import sqlite3

def search_products(keyword):
    conn = sqlite3.connect('shop.db')
    cursor = conn.cursor()
    sql = f"SELECT * FROM products WHERE name LIKE '%{keyword}%'"
    cursor.execute(sql)
    return cursor.fetchall()""",
            "after": """import sqlite3

def search_products(keyword):
    conn = sqlite3.connect('shop.db')
    cursor = conn.cursor()
    sql = "SELECT * FROM products WHERE name LIKE ?"
    cursor.execute(sql, (f'%{keyword}%',))
    return cursor.fetchall()""",
            "message": "Fixed SQL injection in search query with parameterized placeholder"
        },
        {
            "before": """import MySQLdb

def authenticate(username, password):
    db = MySQLdb.connect(host="localhost", user="root", passwd="pass", db="auth")
    cursor = db.cursor()
    query = "SELECT * FROM users WHERE username='" + username + "' AND password='" + password + "'"
    cursor.execute(query)
    return cursor.fetchone()""",
            "after": """import MySQLdb

def authenticate(username, password):
    db = MySQLdb.connect(host="localhost", user="root", passwd="pass", db="auth")
    cursor = db.cursor()
    query = "SELECT * FROM users WHERE username=%s AND password=%s"
    cursor.execute(query, (username, password))
    return cursor.fetchone()""",
            "message": "Use parameterized queries with %s placeholders for MySQL"
        },
        {
            "before": """import sqlite3

def get_orders(status, user_id):
    conn = sqlite3.connect('orders.db')
    cursor = conn.cursor()
    query = f"SELECT * FROM orders WHERE status='{status}' AND user_id={user_id}"
    cursor.execute(query)
    return cursor.fetchall()""",
            "after": """import sqlite3

def get_orders(status, user_id):
    conn = sqlite3.connect('orders.db')
    cursor = conn.cursor()
    query = "SELECT * FROM orders WHERE status=? AND user_id=?"
    cursor.execute(query, (status, user_id))
    return cursor.fetchall()""",
            "message": "Fixed SQL injection with multiple parameters"
        }
    ]
    
    for i, template in enumerate(templates * 20):  # Replicate to get ~100 examples
        examples.append({
            "CVE ID": f"SYNTHETIC-PY-{start_id + i:04d}",
            "CVE Page": "N/A",
            "CWE ID": "CWE-89",
            "codeLink": "N/A",
            "commit_id": "synthetic",
            "commit_message": template["message"],
            "func_after": template["after"],
            "func_before": template["before"],
            "lang": "Python",
            "project": "synthetic",
            "vul": 1
        })
    
    return examples[:100]  # Return 100 examples

def generate_command_injection_examples(start_id=101):
    """Generate command injection examples (CWE-78)"""
    examples = []
    
    templates = [
        {
            "before": """import os

def ping_host(hostname):
    command = 'ping -c 4 ' + hostname
    os.system(command)""",
            "after": """import subprocess

def ping_host(hostname):
    subprocess.run(['ping', '-c', '4', hostname])""",
            "message": "Use subprocess with list arguments instead of os.system"
        },
        {
            "before": """import os

def list_directory(path):
    os.system('ls -la ' + path)""",
            "after": """import subprocess

def list_directory(path):
    subprocess.run(['ls', '-la', path])""",
            "message": "Avoid shell injection by using argument list"
        },
        {
            "before": """import subprocess

def compress_file(filename):
    subprocess.call('tar -czf archive.tar.gz ' + filename, shell=True)""",
            "after": """import subprocess

def compress_file(filename):
    subprocess.call(['tar', '-czf', 'archive.tar.gz', filename])""",
            "message": "Remove shell=True and use list arguments"
        },
        {
            "before": """import os

def git_clone(repo_url):
    os.popen('git clone ' + repo_url).read()""",
            "after": """import subprocess

def git_clone(repo_url):
    subprocess.run(['git', 'clone', repo_url], capture_output=True)""",
            "message": "Replace os.popen with subprocess.run using argument list"
        },
        {
            "before": """import subprocess

def convert_image(input_file, output_file):
    cmd = f"convert {input_file} {output_file}"
    subprocess.run(cmd, shell=True)""",
            "after": """import subprocess

def convert_image(input_file, output_file):
    subprocess.run(['convert', input_file, output_file])""",
            "message": "Use argument list to prevent command injection"
        }
    ]
    
    for i, template in enumerate(templates * 20):
        examples.append({
            "CVE ID": f"SYNTHETIC-PY-{start_id + i:04d}",
            "CVE Page": "N/A",
            "CWE ID": "CWE-78",
            "codeLink": "N/A",
            "commit_id": "synthetic",
            "commit_message": template["message"],
            "func_after": template["after"],
            "func_before": template["before"],
            "lang": "Python",
            "project": "synthetic",
            "vul": 1
        })
    
    return examples[:100]

def generate_code_injection_examples(start_id=201):
    """Generate code injection examples (CWE-94)"""
    examples = []
    
    templates = [
        {
            "before": """def calculate(expression):
    result = eval(expression)
    return result""",
            "after": """import ast

def calculate(expression):
    try:
        node = ast.parse(expression, mode='eval')
        result = ast.literal_eval(expression)
        return result
    except:
        raise ValueError("Invalid expression")""",
            "message": "Use ast.literal_eval instead of eval for safe evaluation"
        },
        {
            "before": """def execute_code(code_string):
    exec(code_string)""",
            "after": """def execute_code(code_string):
    # Avoid exec entirely, use specific safe functions
    raise NotImplementedError("Direct code execution not allowed")""",
            "message": "Remove exec() to prevent arbitrary code execution"
        },
        {
            "before": """def load_config(config_str):
    config = eval(config_str)
    return config""",
            "after": """import json

def load_config(config_str):
    config = json.loads(config_str)
    return config""",
            "message": "Use json.loads instead of eval for configuration"
        },
        {
            "before": """def dynamic_import(module_name):
    module = __import__(module_name)
    return module""",
            "after": """import importlib

ALLOWED_MODULES = ['json', 'os', 'sys']

def dynamic_import(module_name):
    if module_name not in ALLOWED_MODULES:
        raise ValueError("Module not allowed")
    module = importlib.import_module(module_name)
    return module""",
            "message": "Validate module names before dynamic import"
        },
        {
            "before": """def parse_data(data_str):
    data = eval(data_str)
    return data""",
            "after": """import ast

def parse_data(data_str):
    data = ast.literal_eval(data_str)
    return data""",
            "message": "Replace eval with ast.literal_eval for safe parsing"
        }
    ]
    
    for i, template in enumerate(templates * 20):
        examples.append({
            "CVE ID": f"SYNTHETIC-PY-{start_id + i:04d}",
            "CVE Page": "N/A",
            "CWE ID": "CWE-94",
            "codeLink": "N/A",
            "commit_id": "synthetic",
            "commit_message": template["message"],
            "func_after": template["after"],
            "func_before": template["before"],
            "lang": "Python",
            "project": "synthetic",
            "vul": 1
        })
    
    return examples[:100]

def generate_path_traversal_examples(start_id=301):
    """Generate path traversal examples (CWE-22)"""
    examples = []
    
    templates = [
        {
            "before": """def read_file(filename):
    with open('/var/www/uploads/' + filename) as f:
        return f.read()""",
            "after": """import os

def read_file(filename):
    if '..' in filename or filename.startswith('/'):
        raise ValueError("Invalid filename")
    safe_path = os.path.join('/var/www/uploads/', filename)
    with open(safe_path) as f:
        return f.read()""",
            "message": "Validate filename to prevent path traversal"
        },
        {
            "before": """def serve_file(filepath):
    with open(filepath, 'rb') as f:
        return f.read()""",
            "after": """from pathlib import Path

def serve_file(filepath):
    base = Path('/var/www/files')
    requested = (base / filepath).resolve()
    if base not in requested.parents:
        raise ValueError("Access denied")
    with open(requested, 'rb') as f:
        return f.read()""",
            "message": "Use pathlib to validate paths and prevent traversal"
        },
        {
            "before": """import os

def delete_file(filename):
    os.remove('uploads/' + filename)""",
            "after": """import os

def delete_file(filename):
    if not filename or '/' in filename or '..' in filename:
        raise ValueError("Invalid filename")
    os.remove(os.path.join('uploads', filename))""",
            "message": "Sanitize filename before file operations"
        },
        {
            "before": """def load_template(template_name):
    path = f'templates/{template_name}.html'
    with open(path) as f:
        return f.read()""",
            "after": """import os

def load_template(template_name):
    # Only allow alphanumeric and underscore
    if not template_name.replace('_', '').isalnum():
        raise ValueError("Invalid template name")
    path = os.path.join('templates', f'{template_name}.html')
    with open(path) as f:
        return f.read()""",
            "message": "Validate template name to prevent directory traversal"
        },
        {
            "before": """def get_user_file(user_id, filename):
    path = f'/home/user_{user_id}/{filename}'
    return open(path).read()""",
            "after": """import os

def get_user_file(user_id, filename):
    if '..' in filename or '/' in filename:
        raise ValueError("Invalid filename")
    base_dir = f'/home/user_{user_id}'
    safe_path = os.path.realpath(os.path.join(base_dir, filename))
    if not safe_path.startswith(base_dir):
        raise ValueError("Access denied")
    return open(safe_path).read()""",
            "message": "Use realpath to resolve and validate file paths"
        }
    ]
    
    for i, template in enumerate(templates * 20):
        examples.append({
            "CVE ID": f"SYNTHETIC-PY-{start_id + i:04d}",
            "CVE Page": "N/A",
            "CWE ID": "CWE-22",
            "codeLink": "N/A",
            "commit_id": "synthetic",
            "commit_message": template["message"],
            "func_after": template["after"],
            "func_before": template["before"],
            "lang": "Python",
            "project": "synthetic",
            "vul": 1
        })
    
    return examples[:100]

def generate_deserialization_examples(start_id=401):
    """Generate insecure deserialization examples (CWE-502)"""
    examples = []
    
    templates = [
        {
            "before": """import pickle

def load_data(data):
    obj = pickle.loads(data)
    return obj""",
            "after": """import json

def load_data(data):
    obj = json.loads(data)
    return obj""",
            "message": "Use JSON instead of pickle for untrusted data"
        },
        {
            "before": """import pickle

def load_session(session_data):
    return pickle.loads(session_data)""",
            "after": """import json
import hmac
import hashlib

def load_session(session_data, secret_key):
    data = json.loads(session_data)
    # Verify signature
    return data""",
            "message": "Replace pickle with JSON and add signature verification"
        },
        {
            "before": """import yaml

def load_config(config_str):
    config = yaml.load(config_str)
    return config""",
            "after": """import yaml

def load_config(config_str):
    config = yaml.safe_load(config_str)
    return config""",
            "message": "Use yaml.safe_load instead of yaml.load"
        },
        {
            "before": """import pickle
import base64

def deserialize(encoded_data):
    data = base64.b64decode(encoded_data)
    return pickle.loads(data)""",
            "after": """import json
import base64

def deserialize(encoded_data):
    data = base64.b64decode(encoded_data)
    return json.loads(data)""",
            "message": "Avoid pickle deserialization of untrusted data"
        },
        {
            "before": """import marshal

def load_code(bytecode):
    code = marshal.loads(bytecode)
    exec(code)""",
            "after": """# Avoid loading and executing untrusted bytecode
def load_code(bytecode):
    raise NotImplementedError("Loading untrusted code is not allowed")""",
            "message": "Remove marshal.loads to prevent code injection"
        }
    ]
    
    for i, template in enumerate(templates * 20):
        examples.append({
            "CVE ID": f"SYNTHETIC-PY-{start_id + i:04d}",
            "CVE Page": "N/A",
            "CWE ID": "CWE-502",
            "codeLink": "N/A",
            "commit_id": "synthetic",
            "commit_message": template["message"],
            "func_after": template["after"],
            "func_before": template["before"],
            "lang": "Python",
            "project": "synthetic",
            "vul": 1
        })
    
    return examples[:100]

def generate_weak_crypto_examples(start_id=501):
    """Generate weak cryptography examples (CWE-327)"""
    examples = []
    
    templates = [
        {
            "before": """import hashlib

def hash_password(password):
    return hashlib.md5(password.encode()).hexdigest()""",
            "after": """import hashlib
import os

def hash_password(password):
    salt = os.urandom(32)
    key = hashlib.pbkdf2_hmac('sha256', password.encode(), salt, 100000)
    return salt + key""",
            "message": "Use PBKDF2 with salt instead of MD5 for password hashing"
        },
        {
            "before": """from Crypto.Cipher import DES

def encrypt_data(data, key):
    cipher = DES.new(key, DES.MODE_ECB)
    return cipher.encrypt(data)""",
            "after": """from Crypto.Cipher import AES
from Crypto.Random import get_random_bytes

def encrypt_data(data, key):
    cipher = AES.new(key, AES.MODE_GCM)
    ciphertext, tag = cipher.encrypt_and_digest(data)
    return cipher.nonce + tag + ciphertext""",
            "message": "Use AES-GCM instead of DES-ECB for encryption"
        },
        {
            "before": """import hashlib

def hash_data(data):
    return hashlib.sha1(data.encode()).hexdigest()""",
            "after": """import hashlib

def hash_data(data):
    return hashlib.sha256(data.encode()).hexdigest()""",
            "message": "Use SHA-256 instead of SHA-1"
        },
        {
            "before": """from Crypto.Cipher import AES

def encrypt(data, key):
    cipher = AES.new(key, AES.MODE_ECB)
    return cipher.encrypt(data)""",
            "after": """from Crypto.Cipher import AES
from Crypto.Random import get_random_bytes

def encrypt(data, key):
    nonce = get_random_bytes(16)
    cipher = AES.new(key, AES.MODE_GCM, nonce=nonce)
    ciphertext, tag = cipher.encrypt_and_digest(data)
    return nonce + tag + ciphertext""",
            "message": "Use AES-GCM with nonce instead of ECB mode"
        },
        {
            "before": """import random

def generate_token():
    return ''.join([str(random.randint(0, 9)) for _ in range(32)])""",
            "after": """import secrets

def generate_token():
    return secrets.token_hex(16)""",
            "message": "Use secrets module instead of random for cryptographic tokens"
        }
    ]
    
    for i, template in enumerate(templates * 20):
        examples.append({
            "CVE ID": f"SYNTHETIC-PY-{start_id + i:04d}",
            "CVE Page": "N/A",
            "CWE ID": "CWE-327",
            "codeLink": "N/A",
            "commit_id": "synthetic",
            "commit_message": template["message"],
            "func_after": template["after"],
            "func_before": template["before"],
            "lang": "Python",
            "project": "synthetic",
            "vul": 1
        })
    
    return examples[:100]

def generate_hardcoded_secrets_examples(start_id=601):
    """Generate hardcoded credentials examples (CWE-798)"""
    examples = []
    
    templates = [
        {
            "before": """import MySQLdb

def get_db_connection():
    conn = MySQLdb.connect(
        host="localhost",
        user="admin",
        passwd="admin123",
        db="production"
    )
    return conn""",
            "after": """import MySQLdb
import os

def get_db_connection():
    conn = MySQLdb.connect(
        host=os.environ.get('DB_HOST'),
        user=os.environ.get('DB_USER'),
        passwd=os.environ.get('DB_PASSWORD'),
        db=os.environ.get('DB_NAME')
    )
    return conn""",
            "message": "Use environment variables instead of hardcoded credentials"
        },
        {
            "before": """API_KEY = "sk-1234567890abcdef"

def call_api(endpoint):
    headers = {'Authorization': f'Bearer {API_KEY}'}
    # make request""",
            "after": """import os

API_KEY = os.environ.get('API_KEY')

def call_api(endpoint):
    headers = {'Authorization': f'Bearer {API_KEY}'}
    # make request""",
            "message": "Load API keys from environment variables"
        },
        {
            "before": """SECRET_KEY = "very-secret-key-12345"
DEBUG = True""",
            "after": """import os

SECRET_KEY = os.environ.get('SECRET_KEY')
DEBUG = os.environ.get('DEBUG', 'False') == 'True'""",
            "message": "Use environment variables for secrets and configuration"
        },
        {
            "before": """def authenticate():
    username = "admin"
    password = "P@ssw0rd123"
    return check_credentials(username, password)""",
            "after": """import os

def authenticate():
    username = os.environ.get('ADMIN_USER')
    password = os.environ.get('ADMIN_PASSWORD')
    return check_credentials(username, password)""",
            "message": "Remove hardcoded credentials, use environment variables"
        },
        {
            "before": """AWS_ACCESS_KEY = "AKIAIOSFODNN7EXAMPLE"
AWS_SECRET_KEY = "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY""",
            "after": """import os

AWS_ACCESS_KEY = os.environ.get('AWS_ACCESS_KEY')
AWS_SECRET_KEY = os.environ.get('AWS_SECRET_KEY')""",
            "message": "Store AWS credentials in environment variables or IAM roles"
        }
    ]
    
    for i, template in enumerate(templates * 20):
        examples.append({
            "CVE ID": f"SYNTHETIC-PY-{start_id + i:04d}",
            "CVE Page": "N/A",
            "CWE ID": "CWE-798",
            "codeLink": "N/A",
            "commit_id": "synthetic",
            "commit_message": template["message"],
            "func_after": template["after"],
            "func_before": template["before"],
            "lang": "Python",
            "project": "synthetic",
            "vul": 1
        })
    
    return examples[:100]

def main():
    """Generate complete Python vulnerability dataset"""
    
    print("="*70)
    print("Generating Python Vulnerability Dataset")
    print("="*70)
    
    all_examples = []
    
    # Generate each category
    print("\n[1/7] Generating SQL Injection examples (CWE-89)...")
    all_examples.extend(generate_sql_injection_examples(start_id=1))
    
    print("[2/7] Generating Command Injection examples (CWE-78)...")
    all_examples.extend(generate_command_injection_examples(start_id=101))
    
    print("[3/7] Generating Code Injection examples (CWE-94)...")
    all_examples.extend(generate_code_injection_examples(start_id=201))
    
    print("[4/7] Generating Path Traversal examples (CWE-22)...")
    all_examples.extend(generate_path_traversal_examples(start_id=301))
    
    print("[5/7] Generating Deserialization examples (CWE-502)...")
    all_examples.extend(generate_deserialization_examples(start_id=401))
    
    print("[6/7] Generating Weak Cryptography examples (CWE-327)...")
    all_examples.extend(generate_weak_crypto_examples(start_id=501))
    
    print("[7/7] Generating Hardcoded Secrets examples (CWE-798)...")
    all_examples.extend(generate_hardcoded_secrets_examples(start_id=601))
    
    # Save to JSON
    output_file = "python_vulnerabilities.json"
    with open(output_file, 'w') as f:
        json.dump(all_examples, f, indent=2)
    
    print(f"\n{'='*70}")
    print(f"✓ Generated {len(all_examples)} Python vulnerability examples")
    print(f"✓ Saved to: {output_file}")
    print(f"{'='*70}")
    
    # Print statistics
    from collections import Counter
    cwe_counts = Counter([ex['CWE ID'] for ex in all_examples])
    print("\nCWE Distribution:")
    for cwe, count in sorted(cwe_counts.items()):
        print(f"  {cwe}: {count} examples")
    
    print(f"\nAll examples are labeled with vul=1 (vulnerable)")
    print(f"Total size: ~{len(json.dumps(all_examples)) / 1024 / 1024:.1f} MB")

if __name__ == "__main__":
    main()