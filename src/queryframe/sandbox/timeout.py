"""Execution timeout guard."""

from __future__ import annotations

import platform
import signal
import threading
from contextlib import contextmanager
from typing import Generator

from queryframe.utils.errors import SandboxTimeoutError


@contextmanager
def execution_timeout(seconds: int) -> Generator[None, None, None]:
    """Context manager that raises SandboxTimeoutError after `seconds`."""
    if platform.system() != "Windows" and threading.current_thread() is threading.main_thread():
        # Unix: use signal-based timeout (more reliable)
        def _handler(signum: int, frame: object) -> None:
            raise SandboxTimeoutError(
                f"Code execution exceeded {seconds}s timeout"
            )

        old_handler = signal.signal(signal.SIGALRM, _handler)
        signal.alarm(seconds)
        try:
            yield
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old_handler)
    else:
        # Windows / non-main thread: use threading-based timeout
        timer_fired = threading.Event()

        def _timeout() -> None:
            timer_fired.set()

        timer = threading.Timer(float(seconds), _timeout)
        timer.start()
        try:
            yield
            if timer_fired.is_set():
                raise SandboxTimeoutError(
                    f"Code execution exceeded {seconds}s timeout"
                )
        finally:
            timer.cancel()
