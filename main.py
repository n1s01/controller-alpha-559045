#!/usr/bin/env python3

import unicodedata
import re
import string
from typing import Any, Callable, Dict

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
