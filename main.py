#!/usr/bin/env python3

import unicodedata
import re
import string
from typing import Any, Callable, Dict
import base64
import os
import gzip
import zipfile

def timed(func):
    """Decorator to measure execution time of a function."""
    import time, functools
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(f"{func.__name__} executed in {end - start:.6f}s")
        return result
    return wrapper

def merge_dicts(a: dict, b: dict, deep: bool = False) -> dict:
def encrypt_text(plaintext: str, key: bytes) -> str:
    """Encrypt *plaintext* with a simple XOR cipher using *key*.
    Returns a base64‑encoded string.
    Note: This is for demonstration only and not suitable for real security purposes.
    """
    data = plaintext.encode('utf-8')
    encrypted = bytes(b ^ key[i % len(key)] for i, b in enumerate(data))
    return base64.b64encode(encrypted).decode('utf-8')

def decrypt_text(ciphertext: str, key: bytes) -> str:
def compress_file(source_path: str, dest_path: str, method: str = "gzip") -> None:
    """Compress *source_path* to *dest_path*.
    Supported *method* values:
    - "gzip": GZIP compression (resulting file usually ends with .gz)
    - "zip": ZIP archive containing a single file (resulting file ends with .zip)
    """
    if method == "gzip":
        with open(source_path, "rb") as f_in, gzip.open(dest_path, "wb") as f_out:
            f_out.writelines(f_in)
    elif method == "zip":
        with zipfile.ZipFile(dest_path, "w", zipfile.ZIP_DEFLATED) as zipf:
            zipf.write(source_path, arcname=os.path.basename(source_path))
    else:
        raise ValueError(f"Unsupported compression method: {method}")

def decompress_file(source_path: str, dest_path: str, method: str = None) -> None:
    """Decompress *source_path* to *dest_path*.
    If *method* is None, the function infers the format from the file extension.
    Supports gzip (".gz") and zip (".zip").
    """
    if method is None:
        if source_path.endswith('.gz'):
            method = "gzip"
        elif source_path.endswith('.zip'):
            method = "zip"
        else:
            raise ValueError("Unable to infer compression method from file extension; please specify 'method'.")
    if method == "gzip":
        with gzip.open(source_path, "rb") as f_in, open(dest_path, "wb") as f_out:
            f_out.writelines(f_in)
    elif method == "zip":
        with zipfile.ZipFile(source_path, "r") as zipf:
            # Assume archive contains a single file; extract it to dest_path
            name_list = zipf.namelist()
            if not name_list:
                raise FileNotFoundError("ZIP archive is empty.")
            # Extract the first entry to a temporary location, then rename
            temp_dir = os.path.dirname(dest_path)
            zipf.extract(name_list[0], path=temp_dir)
            extracted_path = os.path.join(temp_dir, name_list[0])
            os.replace(extracted_path, dest_path)
    else:
        raise ValueError(f"Unsupported decompression method: {method}")

    """Decrypt a base64‑encoded *ciphertext* produced by :func:`encrypt_text`.
    Returns the original plaintext string.
    """
    encrypted = base64.b64decode(ciphertext.encode('utf-8'))
    decrypted = bytes(b ^ key[i % len(key)] for i, b in enumerate(encrypted))
    return decrypted.decode('utf-8')
    """Merge two dictionaries. If deep=True, merge nested dictionaries recursively.
    Returns a new dictionary leaving the originals untouched.
    """
    result = a.copy()
    for key, value in b.items():
        if deep and key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_dicts(result[key], value, deep=True)
        else:
            result[key] = value
    return result\n\nclass AppError(Exception):\n    """Base class for application-specific errors."""\n    pass\n\nclass DatabaseError(AppError):\n    """Raised for database related errors."""\n    pass\n\nclass ValidationError(AppError):\n    """Raised when input validation fails."""\n    pass\n

class Transaction:
    """Simple transaction context manager placeholder.

    In a real application this would manage resources such as database
    connections, file handles, etc., providing ``begin``, ``commit`` and
    ``rollback`` semantics.  Here it merely prints actions so the behaviour
    can be observed during testing.
    """

    def __enter__(self):
        self.begin()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is None:
            self.commit()
        else:
            self.rollback()
        # Do not suppress exceptions
        return False

    def begin(self):
        print("Transaction started")

    def commit(self):
        print("Transaction committed")

    def rollback(self):
        print("Transaction rolled back")

def normalize_string(text: str) -> str:
import hashlib
import os
import base64\nimport sqlite3

def hash_password(password: str, *, salt: bytes | None = None) -> str:
    """Hash a password with PBKDF2-HMAC-SHA256.
    Returns a string containing salt and hash encoded in base64, separated by '$'."""
    if salt is None:
        salt = os.urandom(16)
    dk = hashlib.pbkdf2_hmac('sha256', password.encode('utf-8'), salt, 100_000)
    return f"{base64.b64encode(salt).decode('utf-8')}${base64.b64encode(dk).decode('utf-8')}"

def verify_password(password: str, hashed: str) -> bool:\n\ndef check_db_health(db_path: str) -> bool:\n    \"\"\"Simple health check for a SQLite database.\n\n    Returns True if a connection can be opened and a simple query succeeds.\n    \"\"\"\n    try:\n        conn = sqlite3.connect(db_path)\n        cur = conn.cursor()\n        cur.execute(\"SELECT 1\")\n        conn.close()\n        return True\n    except Exception:\n        return False
    """Verify a password against the stored hash.
    The stored format is 'salt$hash' as produced by ``hash_password``."""
    try:
        salt_b64, hash_b64 = hashed.split('$')
        salt = base64.b64decode(salt_b64)
        expected_hash = base64.b64decode(hash_b64)
    except Exception:
        return False
    dk = hashlib.pbkdf2_hmac('sha256', password.encode('utf-8'), salt, 100_000)
    return hashlib.compare_digest(dk, expected_hash)

    """
    Normalize a string by removing diacritics, punctuation, and extra whitespace.
    Returns a lower‑cased, clean version of the input.
    """
    # Decompose Unicode characters and remove combining marks
    nfkd = unicodedata.normalize('NFKD', text)
    without_diacritics = "".join(ch for ch in nfkd if not unicodedata.combining(ch))
    # Remove punctuation
    without_punct = re.sub(rf"[{re.escape(string.punctuation)}]", "", without_diacritics)
    # Collapse whitespace and lower case
    return " ".join(without_punct.split()).lower()

class Container:
    """A very small dependency injection container.

    Register providers (callables) for keys and resolve them on demand.
    Supports optional singleton scope.
    """

    def __init__(self) -> None:
        self._providers: Dict[Any, Callable[[], Any]] = {}
        self._singletons: Dict[Any, Any] = {}

    def register(self, key: Any, provider: Callable[[], Any], *, singleton: bool = False) -> None:
        """Register a provider for *key*.

        If *singleton* is True, the provider is called at most once and the
        resulting instance is cached.
        """
        if singleton:
            def _singleton_provider() -> Any:
                if key not in self._singletons:
                    self._singletons[key] = provider()
                return self._singletons[key]
            self._providers[key] = _singleton_provider
        else:
            self._providers[key] = provider

    def resolve(self, key: Any) -> Any:
        """Return an instance for *key* using the registered provider.

        Raises ``KeyError`` if no provider has been registered.
        """
        if key not in self._providers:
            raise KeyError(f"No provider registered for {key}")
        return self._providers[key]()

if __name__ == "__main__":
    # Demonstrate password hashing and verification
    pwd = "s3cr3t"
    stored = hash_password(pwd)
    print(f"Stored hash: {stored}")
    print("Verification (correct):", verify_password(pwd, stored))
    print("Verification (incorrect):", verify_password("wrong", stored))

    print("Hello, World!")
    # Set up the DI container
    container = Container()
    container.register(Transaction, lambda: Transaction(), singleton=True)
    # Resolve the Transaction instance and use it
    txn = container.resolve(Transaction)
    with txn:
        print("Doing some work inside a transaction managed by DI container")
    # Example usage of normalize_string
    print(normalize_string("Café, déjà vu!"))
    # Run pending database migrations
    from migrations import MigrationManager
    mgr = MigrationManager()
    def initial_migration():
        print("Applying initial migration: creating tables (demo)")
    mgr.add(1, initial_migration)
    mgr.apply()

