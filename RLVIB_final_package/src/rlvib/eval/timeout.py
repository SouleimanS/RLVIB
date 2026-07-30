"""A per-item wall-clock guard so one pathological clip can't hang a long eval.

CMM/AVHBench/DAVE wrap model.generate() (which does the media decode *and* the
generation) in this. If a single item runs longer than the cap -- e.g. a malformed
video that makes the decoder spin -- it raises TimeoutError, which the eval loop's
existing `except Exception` turns into pred=None (a skip) and a recorded
"ERROR: item exceeded Ns", instead of stalling the whole job forever.

SIGALRM-based, so it only works on the main thread on Unix -- which is exactly how the
eval scripts run on the cluster. A no-op when seconds<=0.
"""
from __future__ import annotations

import contextlib
import signal


@contextlib.contextmanager
def time_limit(seconds: int):
    """Raise TimeoutError if the wrapped block runs longer than `seconds` (<=0 disables)."""
    if not seconds or seconds <= 0:
        yield
        return

    def _raise(signum, frame):  # noqa: ARG001 -- signal handler signature
        raise TimeoutError(f"item exceeded {seconds}s")

    old = signal.signal(signal.SIGALRM, _raise)
    signal.alarm(int(seconds))
    try:
        yield
    finally:
        signal.alarm(0)                      # always disarm, even on success/other errors
        signal.signal(signal.SIGALRM, old)
