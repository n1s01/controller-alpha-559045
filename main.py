#!/usr/bin/env python3

def heavy_operation():
    """Perform an operation that requires a heavy module.
    The heavy module is imported lazily to avoid startup cost.
    """
    import heavy_module  # Lazy import
    heavy_module.perform()

def main():
    print("Hello, World!")
    # Uncomment the following line to execute the heavy operation:
    # heavy_operation()

if __name__ == "__main__":
    main()
