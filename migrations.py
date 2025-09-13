#!/usr/bin/env python3

import os
import json
from typing import Callable, List, Tuple

MIGRATIONS_FILE = "migrations_state.json"

def _load_state() -> int:
    if not os.path.exists(MIGRATIONS_FILE):
        return 0
    try:
        with open(MIGRATIONS_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data.get("applied", 0)
    except Exception:
        return 0

def _save_state(version: int) -> None:
    with open(MIGRATIONS_FILE, "w", encoding="utf-8") as f:
        json.dump({"applied": version}, f)

class MigrationManager:
    """Simple migration system.

    Register migrations as callables that accept no arguments.
    Each migration has a monotonically increasing integer version.
    """

    def __init__(self) -> None:
        self._migrations: List[Tuple[int, Callable[[], None]]] = []

    def add(self, version: int, func: Callable[[], None]) -> None:
        self._migrations.append((version, func))
        self._migrations.sort(key=lambda x: x[0])

    def apply(self) -> None:
        current = _load_state()
        for version, func in self._migrations:
            if version > current:
                func()
                _save_state(version)
