"""Nested updates for frozen-dataclass specs, with tuple indexing.

``update(spec, {"holes__2__z": 0.14, "window__length": 0.006})`` replaces fields
through nested frozen dataclasses and tuples (a numeric path part indexes a tuple).
"""
from __future__ import annotations

from dataclasses import fields, is_dataclass, replace


def _set(obj, parts, value):
    head, rest = parts[0], parts[1:]
    if isinstance(obj, tuple):
        i = int(head)
        new = value if not rest else _set(obj[i], rest, value)
        return obj[:i] + (new,) + obj[i + 1:]
    if not is_dataclass(obj) or head not in {f.name for f in fields(obj)}:
        raise KeyError(f"no field {head!r} on {type(obj).__name__}")
    new = value if not rest else _set(getattr(obj, head), rest, value)
    return replace(obj, **{head: new})


def update(spec, changes: dict):
    for path, value in changes.items():
        spec = _set(spec, path.split("__"), value)
    return spec
