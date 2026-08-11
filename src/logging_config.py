"""
logging_config.py

Central logging setup for the Health Insurance Cross-Sell system.

Why one module instead of logging.basicConfig() in each file
------------------------------------------------------------
basicConfig() only takes effect on the first call in a process; every later
call is silently ignored. If several modules each call it, whichever imports
first wins and the log format becomes dependent on import order. Configuring
the root logger once here and handing out named loggers via get_logger()
makes behaviour deterministic and lets every module inherit the same format,
levels and destinations.

Replacing print() with a logger also gives us severity. print() cannot
distinguish "loaded 381,109 rows" from "the encoder rejected this record" --
both land on stdout with equal weight and neither is retained after the
process exits.
"""

from __future__ import annotations

import logging
import logging.handlers
import sys
from pathlib import Path

LOG_DIR = Path(__file__).resolve().parent.parent / "logs"
LOG_FILE = LOG_DIR / "application.log"

CONSOLE_LEVEL = logging.INFO
FILE_LEVEL = logging.DEBUG

LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s:%(lineno)d | %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

_configured = False


def _configure_root() -> None:
    """Attach handlers to the root logger exactly once."""
    global _configured
    if _configured:
        return

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    formatter = logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT)

    root = logging.getLogger()
    root.setLevel(logging.DEBUG)

    console = logging.StreamHandler(stream=sys.stdout)
    console.setLevel(CONSOLE_LEVEL)
    console.setFormatter(formatter)
    root.addHandler(console)

    # Rotating file handler so repeated training runs cannot fill the disk.
    file_handler = logging.handlers.RotatingFileHandler(
        LOG_FILE, maxBytes=1_000_000, backupCount=3, encoding="utf-8"
    )
    file_handler.setLevel(FILE_LEVEL)
    file_handler.setFormatter(formatter)
    root.addHandler(file_handler)

    for noisy in ("matplotlib", "urllib3", "asyncio", "watchdog"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    _configured = True


def get_logger(name: str) -> logging.Logger:
    """Return a module-scoped logger. Always call as get_logger(__name__)."""
    _configure_root()
    return logging.getLogger(name)
