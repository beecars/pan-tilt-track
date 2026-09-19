"""ClipRecorder: writes frames handed to it into a fixed-length MP4 clip
on the local drive, for headless recording (see --record in
scripts/run_tracker.py) where there's no RTSP viewer to capture from."""

from __future__ import annotations

import time
from pathlib import Path

import cv2


class ClipRecorder:
    def __init__(
        self,
        output_dir: Path | str,
        width: int,
        height: int,
        framerate: float,
        max_seconds: float = 10.0,
        fourcc: str = "mp4v",
    ):
        self.output_dir = Path(output_dir)
        self.width = width
        self.height = height
        self.framerate = framerate
        self.max_seconds = max_seconds
        self.fourcc = fourcc
        self._writer: cv2.VideoWriter | None = None
        self._frame_count = 0
        self._path: Path | None = None
        self._start_time: float | None = None
        self.clips_saved = 0

    @property
    def active(self) -> bool:
        return self._writer is not None

    @property
    def elapsed_seconds(self) -> float:
        return time.monotonic() - self._start_time if self._start_time is not None else 0.0

    @property
    def total_seconds(self) -> float:
        return self.max_seconds

    def start(self, framerate: float | None = None) -> Path:
        """Begin a new clip, finalizing any clip already in progress
        first. Returns the output path. `framerate` overrides the
        constructor's nominal camera framerate for this clip's declared
        fps -- pass the caller's actually-observed frame rate (e.g. a
        detection-throttled tracking loop, which runs well under the raw
        camera rate) so the clip's playback speed matches real time."""
        if self.active:
            self.stop()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._path = self.output_dir / f"clip_{time.strftime('%Y%m%d_%H%M%S')}.mp4"
        self._writer = cv2.VideoWriter(
            str(self._path),
            cv2.VideoWriter_fourcc(*self.fourcc),
            framerate or self.framerate,
            (self.width, self.height),
        )
        self._frame_count = 0
        self._start_time = time.monotonic()
        return self._path

    def write(self, frame) -> Path | None:
        """No-op unless a clip is in progress. Returns the finalized path
        once max_seconds of wall-clock time has elapsed since start()
        (auto-stopping this clip), otherwise None."""
        if not self.active:
            return None
        self._writer.write(frame)
        self._frame_count += 1
        if self.elapsed_seconds >= self.max_seconds:
            return self.stop()
        return None

    def stop(self) -> Path | None:
        """Finalize the in-progress clip, if any, and return its path."""
        if not self.active:
            return None
        self._writer.release()
        self._writer = None
        path, self._path = self._path, None
        self.clips_saved += 1
        return path
