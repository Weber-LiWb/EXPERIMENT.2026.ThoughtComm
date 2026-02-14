from __future__ import annotations
import os, random, time, json
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def get_dtype(name: str):
    if name == "float16":
        return torch.float16
    if name == "bfloat16":
        return torch.bfloat16
    if name == "float32":
        return torch.float32
    raise ValueError(f"Unknown dtype: {name}")

def now_utc_compact() -> str:
    # YYYYMMDD_HHMMSS
    return time.strftime("%Y%m%d_%H%M%S", time.gmtime())

def ensure_dir(p: str | Path) -> Path:
    p = Path(p)
    p.mkdir(parents=True, exist_ok=True)
    return p

def atomic_write_json(path: str | Path, obj: Any) -> None:
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)

def torch_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"
