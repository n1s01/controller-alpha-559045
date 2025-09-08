#!/usr/bin/env python3

import unicodedata
import re
import string

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

if __name__ == "__main__":
    print("Hello, World!")
    # Example usage of the Transaction context manager
    with Transaction():
        print("Doing some work inside a transaction")
    # Example usage of normalize_string
    print(normalize_string("Café, déjà vu!"))
