from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass
from functools import wraps
import time

#profiling utilities for timeit decorator and report generation, used in main.py and benchmarking.py

@dataclass
class _TimerStat:
    count: int = 0
    total: float = 0.0
    min: float = float("inf")
    max: float = 0.0

    def add(self, dt):
        self.count += 1
        self.total += dt
        if dt < self.min:
            self.min = dt
        if dt > self.max:
            self.max = dt

    @property
    def avg(self):
        return self.total / self.count if self.count else 0.0


_ENABLED = False
_STATS = defaultdict(_TimerStat)


def set_enabled(enabled=True):
    global _ENABLED
    _ENABLED = bool(enabled)


def is_enabled():
    return _ENABLED


def reset_profiler():
    _STATS.clear()


def get_rows(sort_by="total", descending=True):
    rows = []
    for name, stat in _STATS.items():
        rows.append({
            "name": name,
            "count": stat.count,
            "total": stat.total,
            "avg": stat.avg,
            "min": stat.min if stat.count else 0.0,
            "max": stat.max,
        })

    if sort_by not in {"name", "count", "total", "avg", "min", "max"}:
        sort_by = "total"
    rows.sort(key=lambda r: r[sort_by], reverse=descending)
    return rows


def print_report(title=None):
    rows = get_rows(sort_by="total", descending=True)
    if title:
        print(title)
    if not rows:
        print("(no timing samples)")
        return

    print(f"{'metric':34s} {'count':>8s} {'total':>10s} {'avg':>10s} {'min':>10s} {'max':>10s}")
    for row in rows:
        print(
            f"{row['name'][:34]:34s} "
            f"{row['count']:8d} "
            f"{row['total']:10.2f} "
            f"{row['avg']:10.3f} "
            f"{row['min']:10.3f} "
            f"{row['max']:10.3f}"
        )


@contextmanager
def timed(name):
    if not _ENABLED:
        yield
        return

    t0 = time.perf_counter()
    try:
        yield
    finally:
        _STATS[name].add(time.perf_counter() - t0)


def timeit(name):
    def deco(fn):
        @wraps(fn)
        def wrapped(*args, **kwargs):
            with timed(name):
                return fn(*args, **kwargs)
        return wrapped
    return deco