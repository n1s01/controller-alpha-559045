#!/usr/bin/env python3\n# Simple in-memory file cache\n_file_cache = {}\n\ndef read_file_cached(path):\n    if path in _file_cache:\n        return _file_cache[path]\n    with open(path, 'r') as f:\n        data = f.read()\n    _file_cache[path] = data\n    return data\n

print("Hello, World!")
