from __future__ import annotations
from pathlib import Path
from typing import Optional
from omegaconf import DictConfig

from .utils import now_utc_compact, ensure_dir
from .config import save_config

def make_run_dir(cfg: DictConfig) -> Path:
    base = Path(cfg.runtime.artifacts_dir)
    run_name = cfg.runtime.run_name or "run"
    run_id = f"{run_name}_{now_utc_compact()}"
    out = ensure_dir(base / run_id)
    # persist resolved config
    save_config(cfg, str(out / "config_resolved.yaml"))
    return out
