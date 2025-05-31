"""Minimal YAML configuration loader for the project."""
from __future__ import annotations

from typing import Any, Dict
import os


def load_config(path: str | None = None) -> Dict[str, Any]:
    if path is None:
        path = os.path.join(os.path.dirname(__file__), "config.yaml")

    config: Dict[str, Any] = {}
    stack = [config]
    indents = [0]
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip()
            if not line or line.lstrip().startswith("#"):
                continue
            indent = len(line) - len(line.lstrip())
            while indent < indents[-1]:
                stack.pop()
                indents.pop()
            key, _, rest = line.lstrip().partition(":")
            key = key.strip()
            rest = rest.strip()
            if rest == "":
                child: Dict[str, Any] = {}
                stack[-1][key] = child
                stack.append(child)
                indents.append(indent + 2)
            else:
                value: Any = rest
                if value.isdigit():
                    value = int(value)
                stack[-1][key] = value
    return config
