"""Live CLI reporting for scripts/run_tracker.py: LiveDashboard renders a
rich-based, continuously-refreshed view of acquire/handoff state,
detection counts, and pan/tilt position; PlainReporter is the --plain
fallback that just prints lines like the tool always did."""

from __future__ import annotations

from collections import deque

from rich.console import Group
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from ..dynamixel.config import JointLimits, TICKS_PER_REV

DEG_PER_TICK = 360.0 / TICKS_PER_REV

# state -> (log/badge label, color)
STATE_STYLE = {
    "MANUAL": ("MANUAL", "magenta"),
    "ACQUIRE": ("ACQUIRE (wide)", "yellow"),
    "HANDOFF": ("HANDOFF (tele)", "green"),
    "STARTING": ("STARTING", "dim"),
}


class PlainReporter:
    """--plain fallback: log() is a plain print(), update() is a no-op
    since there's no live view to redraw. Also what a non-interactive
    (piped/logged) run should use -- rich's Live assumes a real terminal."""

    def __enter__(self) -> "PlainReporter":
        return self

    def __exit__(self, *exc) -> None:
        return None

    def log(self, message: str) -> None:
        print(message)

    def update(self, **_fields) -> None:
        pass


def _stage_box(name: str, value: str) -> Panel:
    return Panel(
        Text(f"{name}\n{value}", justify="center"),
        width=12,
        padding=(0, 0),
        border_style="grey50",
    )


def _pipeline_row(label: str, cam: str, timing: dict, extra: str) -> Table:
    """One block-diagram row: input -> preprocess -> inference ->
    postprocess -> tracker, each stage's most recent timing (from
    YoloDetector.last_timing) in its own box."""
    def _ms(key: str) -> str:
        value = timing.get(key)
        return f"{value:5.1f}ms" if value is not None else "   – "

    stages = [
        ("IN", cam),
        ("PRE", _ms("preprocess_ms")),
        ("INFER", _ms("nn_inference_ms")),
        ("POST", _ms("postprocess_ms")),
        ("TRACK", _ms("track_ms")),
    ]
    grid = Table.grid(padding=(0, 0))
    cells = [Text(f" {label} ", style="bold cyan")]
    for i, (name, value) in enumerate(stages):
        if i > 0:
            cells.append(Text(" → ", style="dim"))
        cells.append(_stage_box(name, value))
    cells.append(Text(f"  {extra}", style="dim"))
    for _ in cells:
        grid.add_column(vertical="middle")
    grid.add_row(*cells)
    return grid


def _gauge(
    ticks: int | None,
    limits: JointLimits,
    bounds: tuple[int, int] | None = None,
    width: int = 24,
) -> Text:
    span = limits.max_position - limits.min_position

    def frac(value: int) -> float:
        return 0.0 if span <= 0 else min(1.0, max(0.0, (value - limits.min_position) / span))

    def pos(value: int) -> int:
        return min(width - 1, int(round(frac(value) * (width - 1))))

    chars = ["─"] * width
    styles = ["grey50"] * width

    if bounds is not None:
        lo, hi = pos(bounds[0]), pos(bounds[1])
        for i in range(lo, hi + 1):
            styles[i] = "yellow"
        chars[lo] = "╎"
        chars[hi] = "╎"

    if ticks is not None:
        p = pos(ticks)
        chars[p] = "●"
        styles[p] = "bold cyan"

    bar = Text()
    for ch, style in zip(chars, styles):
        bar.append(ch, style=style)
    return bar


class LiveDashboard:
    """Rich Live view: a block-diagram of each detector's pipeline
    (input -> preprocess -> inference -> postprocess -> tracker) with
    live per-stage timings, current acquire/handoff/manual state and
    detection counts, pan/tilt position (ticks, degrees, and a bar gauge
    against the joint's configured limits), and a rolling log of state
    transitions / capture / record events underneath."""

    def __init__(
        self,
        pan_limits: JointLimits,
        tilt_limits: JointLimits,
        wide_cam: str = "wide",
        tele_cam: str = "tele",
        wide_pan_bounds: tuple[int, int] | None = None,
        wide_tilt_bounds: tuple[int, int] | None = None,
        max_log_lines: int = 10,
    ):
        self.pan_limits = pan_limits
        self.tilt_limits = tilt_limits
        self.wide_cam = wide_cam
        self.tele_cam = tele_cam
        # The pan/tilt sub-range wide's own FOV can command, per its
        # handoff calibration -- highlighted on the gauges below so it's
        # clear how much of the full joint travel wide can steer into.
        self.wide_pan_bounds = wide_pan_bounds
        self.wide_tilt_bounds = wide_tilt_bounds
        self._log: deque[str] = deque(maxlen=max_log_lines)

        self.state = "STARTING"
        self.wide_detections = 0
        self.tele_detections = 0
        self.wide_timing: dict = {}
        self.tele_timing: dict = {}
        self.tele_frame_interval_ms: float | None = None
        self.pan_ticks: int | None = None
        self.tilt_ticks: int | None = None

        self._live = Live(self._render(), refresh_per_second=8, transient=False)

    def __enter__(self) -> "LiveDashboard":
        self._live.__enter__()
        return self

    def __exit__(self, *exc) -> None:
        self._live.__exit__(*exc)

    def log(self, message: str) -> None:
        self._log.append(message)
        self._live.update(self._render())

    def update(self, **fields) -> None:
        for key, value in fields.items():
            setattr(self, key, value)
        self._live.update(self._render())

    def _render(self) -> Group:
        label, style = STATE_STYLE.get(self.state, (self.state, "white"))

        fps = f"  {1000 / self.tele_frame_interval_ms:.1f} fps" if self.tele_frame_interval_ms else ""
        pipeline = Group(
            _pipeline_row("WIDE", self.wide_cam, self.wide_timing, f"{self.wide_detections} det"),
            _pipeline_row("TELE", self.tele_cam, self.tele_timing, f"{self.tele_detections} det{fps}"),
        )

        stats = Table.grid(padding=(0, 2))
        stats.add_column(justify="right", style="dim")
        stats.add_column()
        stats.add_row("state", Text(label, style=f"bold {style}"))
        stats.add_row("pan", self._axis_row(self.pan_ticks, self.pan_limits, self.wide_pan_bounds))
        stats.add_row("tilt", self._axis_row(self.tilt_ticks, self.tilt_limits, self.wide_tilt_bounds))

        log_text = Text("\n".join(self._log) or "…", style="dim")
        return Group(
            Panel(pipeline, title="pipeline", border_style="magenta"),
            Panel(stats, title="pan-tilt-track", border_style="blue"),
            Panel(log_text, title="log", border_style="grey50"),
        )

    def _axis_row(self, ticks: int | None, limits: JointLimits, bounds: tuple[int, int] | None) -> Text:
        row = _gauge(ticks, limits, bounds)
        if ticks is not None:
            row.append(f"  {ticks:5d} ticks  {ticks * DEG_PER_TICK:6.1f}°")
        return row
