"""Non-blocking single-key stdin reads for scripts run directly in a
terminal (cbreak mode so keys land without waiting for Enter)."""

from __future__ import annotations

import select
import sys
import termios
import tty


class NonBlockingKeyReader:
    """No-op if stdin isn't a real terminal (e.g. piped/redirected/
    backgrounded), so callers can use it unconditionally."""

    def __init__(self) -> None:
        self._fd = sys.stdin.fileno() if sys.stdin.isatty() else None
        self._original = None

    def __enter__(self) -> "NonBlockingKeyReader":
        if self._fd is not None:
            self._original = termios.tcgetattr(self._fd)
            tty.setcbreak(self._fd)
        return self

    def __exit__(self, *exc) -> None:
        if self._fd is not None and self._original is not None:
            termios.tcsetattr(self._fd, termios.TCSADRAIN, self._original)

    def read_key(self) -> str | None:
        """Returns a single pressed key, or None if nothing is waiting."""
        if self._fd is None:
            return None
        if select.select([sys.stdin], [], [], 0)[0]:
            return sys.stdin.read(1)
        return None
