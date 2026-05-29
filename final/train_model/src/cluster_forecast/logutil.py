"""Lightweight stage logging for long-running train / CV jobs."""

from __future__ import annotations

import logging
import time
from contextlib import contextmanager
from typing import Iterator

LOGGER = logging.getLogger("cluster_forecast")


def configure_logging() -> None:
    """Idempotent INFO logging to stdout (call once from trainer)."""
    root = logging.getLogger("cluster_forecast")
    if root.handlers:
        return
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s | %(message)s", datefmt="%H:%M:%S"))
    root.addHandler(handler)
    root.setLevel(logging.INFO)
    root.propagate = False


@contextmanager
def stage(name: str) -> Iterator[None]:
    """Log start/finish of a stage with elapsed seconds."""
    LOGGER.info("→ %s", name)
    t0 = time.perf_counter()
    try:
        yield
    finally:
        LOGGER.info("✓ %s (%.1f s)", name, time.perf_counter() - t0)
