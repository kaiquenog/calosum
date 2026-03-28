from __future__ import annotations

import json
from dataclasses import fields, is_dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any


def to_primitive(value: Any) -> Any:
    if is_dataclass(value):
        return {
            field.name: to_primitive(getattr(value, field.name))
            for field in fields(value)
        }
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, dict):
        return {str(key): to_primitive(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [to_primitive(item) for item in value]
    return value


def to_json(value: Any, *, indent: int = 2) -> str:
    return json.dumps(to_primitive(value), indent=indent, ensure_ascii=False, sort_keys=True)


def dump_json(path: str | Path, value: Any, *, indent: int = 2) -> None:
    Path(path).write_text(to_json(value, indent=indent) + "\n", encoding="utf-8")
