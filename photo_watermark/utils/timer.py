"""计时工具。"""

import time
from contextlib import contextmanager

from .logger import get_logger

_log = get_logger("photo_watermark.timer")


@contextmanager
def timeit(label="block"):
    """with timeit('embed'): ... 打印耗时。"""
    t0 = time.perf_counter()
    try:
        yield
    finally:
        _log.info("%s: %.3fs", label, time.perf_counter() - t0)
