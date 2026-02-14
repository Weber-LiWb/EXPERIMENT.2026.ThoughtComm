from __future__ import annotations
from pathlib import Path
from typing import Any, Optional
from omegaconf import OmegaConf, DictConfig, ListConfig

def load_config(default_path: str, override_path: str) -> DictConfig:
    default = OmegaConf.load(default_path)
    override = OmegaConf.load(override_path)
    cfg = OmegaConf.merge(default, override)
    return cfg # pyright: ignore[reportReturnType]

def resolve_path(p: Optional[str]) -> str:
    if p is None:
        return None
    return str(Path(p).expanduser().resolve())

def save_config(cfg: DictConfig, out_path: str) -> None:
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, out_path)
