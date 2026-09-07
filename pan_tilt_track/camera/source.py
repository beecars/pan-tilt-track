"""Minimal camera source interface, so the tracking loop doesn't care whether
frames come from nvarguscamerasrc, a file, or (later) something else."""

from __future__ import annotations

from typing import Protocol


class CameraSource(Protocol):
    def read(self):
        """Return the next frame as a BGR numpy array, or None if unavailable."""
        ...

    def release(self) -> None:
        ...
