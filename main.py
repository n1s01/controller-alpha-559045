#!/usr/bin/env python3

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

if __name__ == "__main__":
    print("Hello, World!")
    # Example usage of the Transaction context manager
    with Transaction():
        print("Doing some work inside a transaction")
