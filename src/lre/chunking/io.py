# src/lre/chunking/io.py
from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Iterable, Any

def to_jsonable(x: Any) -> Any:
    return asdict(x) if is_dataclass(x) else x

def write_jsonl(path: Path, rows: Iterable[Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(to_jsonable(r), ensure_ascii=False) + "\n")
