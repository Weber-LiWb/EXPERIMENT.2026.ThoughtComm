from __future__ import annotations
import logging
from pathlib import Path
from rich.logging import RichHandler

def setup_logger(log_path: str | None = None, level: int = logging.INFO) -> logging.Logger:
    handlers = [RichHandler(rich_tracebacks=True, show_time=True, show_level=True, show_path=False)]
    if log_path:
        Path(log_path).parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_path, encoding="utf-8")
        fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
        handlers.append(fh)

    logging.basicConfig(level=level, handlers=handlers)
    return logging.getLogger("thoughtcomm")
