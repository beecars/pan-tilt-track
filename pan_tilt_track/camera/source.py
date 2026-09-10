"""Structural interface for camera sources: read() and release()."""

from __future__ import annotations

from typing import Protocol


class CameraSource(Protocol):
    def read(self):
        """Return the next frame as a BGR numpy array, or None if unavailable."""
        ...

    def release(self) -> None:
        ...
