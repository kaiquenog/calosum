from __future__ import annotations

import json
import types
from dataclasses import fields, is_dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Type, TypeVar, get_type_hints, Union, get_origin, get_args

T = TypeVar("T")


def from_primitive(cls: Type[T], value: Any) -> T:
    if value is None:
        return None  # type: ignore
    if is_dataclass(cls):
        if not isinstance(value, dict):
            return value # Already hydrated or incompatible
            
        try:
            field_types = get_type_hints(cls)
        except Exception:
            # Fallback to field.type if get_type_hints fails
            field_types = {f.name: f.type for f in fields(cls)}
            
        kwargs = {}
        for k, v in value.items():
            if k in field_types:
                ftype = field_types[k]
                
                # Handle Optional[X] which is Union[X, NoneType]
                origin = get_origin(ftype)
                if origin is Union or (hasattr(types, "UnionType") and origin is getattr(types, "UnionType", None)):
                    args = get_args(ftype)
                    # Pick the first non-None type
                    non_none = [a for a in args if a is not type(None)]
                    if non_none:
                        ftype = non_none[0]

                # Handle recursive dataclasses
                if is_dataclass(ftype) and isinstance(v, dict):
                    kwargs[k] = from_primitive(ftype, v)
                elif (origin is list) and isinstance(v, list):
                    # Handle list of dataclasses
                    args = get_args(ftype)
                    if args and is_dataclass(args[0]):
                        kwargs[k] = [from_primitive(args[0], item) for item in v]
                    else:
                        kwargs[k] = v
                else:
                    kwargs[k] = v
        return cls(**kwargs)
    return value


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
    
    # NumPy support
    try:
        import numpy as np
        if isinstance(value, np.ndarray):
            if value.ndim == 0: # Handle 0-d arrays
                return to_primitive(value.item())
            return [to_primitive(item) for item in value.tolist()]
        if isinstance(value, (np.generic, np.number, np.bool_)):
            return value.item()
        if hasattr(value, "dtype"): # Generic catch-all for remaining numpy scalars
            return value.item()
    except (ImportError, AttributeError):
        pass
        
    return value


def to_json(value: Any, *, indent: int = 2) -> str:
    return json.dumps(to_primitive(value), indent=indent, ensure_ascii=False, sort_keys=True)


def dump_json(path: str | Path, value: Any, *, indent: int = 2) -> None:
    Path(path).write_text(to_json(value, indent=indent) + "\n", encoding="utf-8")
