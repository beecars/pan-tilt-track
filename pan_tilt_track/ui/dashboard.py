"""Live CLI reporting for scripts/run_tracker.py: LiveDashboard renders a
rich-based, continuously-refreshed view of acquire/handoff state,
detection counts, and pan/tilt position; PlainReporter is the --plain
fallback that just prints lines like the tool always did."""

from __future__ import annotations

from collections import deque

from rich.console import Group
from rich.live import Live
from rich.panel import Panel
from rich.rule import Rule
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


def _stage_box(name: str, value: str, border_style: str, width: int = 12) -> Panel:
    return Panel(
        Text(f"{name}\n{value}", justify="center"),
        width=width,
        padding=(0, 0),
        border_style=border_style,
    )


def _branch_row(
    label: str, cam: str, timing: dict, count: int, active: bool, downsample: str | None = None
) -> Table:
    """One camera's detection branch: input -> [downsample] -> preprocess
    -> inference -> postprocess -> tracker. Each camera has its own
    detector, so both branches can run in the same frame (wide acquisition
    probing tele) or just one (tele-only once locked) -- `active` says
    whether this branch ran. The inactive row shows blank ("–") stage
    values rather than stale last-known numbers, so it reads as "not
    currently running". `downsample` (target resolution) adds a DOWN stage
    after IN when that branch's camera captures at a different resolution
    than the pipeline processes."""
    def _ms(key: str) -> str:
        if not active:
            return "   – "
        value = timing.get(key)
        return f"{value:5.1f}ms" if value is not None else "   – "

    row_style = "bold cyan" if active else "dim"
    box_style = "cyan" if active else "grey50"
    arrow = "▶" if active else " "

    stages = [("IN", cam)]
    if downsample:
        stages.append(("DOWN", downsample))
    stages += [
        ("PRE", _ms("preprocess_ms")),
        ("INFER", _ms("nn_inference_ms")),
        ("POST", _ms("postprocess_ms")),
        ("TRACK", _ms("track_ms")),
    ]
    grid = Table.grid(padding=(0, 0))
    cells = [Text(f"{arrow} {label} ", style=row_style)]
    for i, (name, value) in enumerate(stages):
        if i > 0:
            cells.append(Text(" → ", style=row_style if active else "dim"))
        box_width = 14 if name == "IN" else 12
        cells.append(_stage_box(name, value, box_style, width=box_width))
    cells.append(Text(f"  {count if active else '–'} det", style=row_style))
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


def _badge(label: str, active: bool, active_style: str) -> Text:
    return Text(f" {label} ", style=f"bold white on {active_style}" if active else "dim")


class LiveDashboard:
    """Rich Live view: a block-diagram of the two camera detection
    branches (wide, tele) -- each branch shows its own last-known stats,
    with an arrow/highlight on whichever branches ran this frame --
    current acquire/handoff/manual state, the locked track's ID/
    confidence, pan/tilt position (ticks, degrees, and a bar gauge against
    the joint's configured limits), status badges (recording/RTSP/PIP),
    the active run config, and a rolling log of state transitions /
    capture / record events underneath."""

    def __init__(
        self,
        pan_limits: JointLimits,
        tilt_limits: JointLimits,
        run_config: dict | None = None,
        wide_cam: str = "wide",
        tele_cam: str = "tele",
        wide_downsample: str | None = None,
        wide_pan_bounds: tuple[int, int] | None = None,
        wide_tilt_bounds: tuple[int, int] | None = None,
        max_log_lines: int = 10,
    ):
        self.pan_limits = pan_limits
        self.tilt_limits = tilt_limits
        self.run_config = run_config or {}
        self.wide_cam = wide_cam
        self.tele_cam = tele_cam
        self.wide_downsample = wide_downsample
        self.wide_pan_bounds = wide_pan_bounds
        self.wide_tilt_bounds = wide_tilt_bounds
        self._log: deque[str] = deque(maxlen=max_log_lines)

        self.state = "STARTING"
        self.wide_active = False
        self.tele_active = False
        self.wide_timing: dict = {}
        self.tele_timing: dict = {}
        self.wide_detections = 0
        self.tele_detections = 0
        self.loop_interval_ms: float | None = None
        self.locked_track_id: int | None = None
        self.locked_confidence: float | None = None
        self.pan_ticks: int | None = None
        self.tilt_ticks: int | None = None
        self.recording = False
        self.recording_elapsed: float | None = None
        self.recording_total: float | None = None
        self.rtsp_active = False
        self.pip_active = False
        self.rtsp_clients = 0
        self.rtsp_fps: float | None = None
        self.clips_saved = 0
        self.pip_compose_ms: float | None = None

        self._live = Live(self._render(), refresh_per_second=8, transient=False, screen=True)

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

    def _render(self) -> Panel:
        label, style = STATE_STYLE.get(self.state, (self.state, "white"))

        pipeline = Group(
            _branch_row(
                "WIDE", self.wide_cam, self.wide_timing, self.wide_detections,
                active=self.wide_active, downsample=self.wide_downsample,
            ),
            _branch_row(
                "TELE", self.tele_cam, self.tele_timing, self.tele_detections,
                active=self.tele_active,
            ),
        )

        stats = Table.grid(padding=(0, 2))
        stats.add_column(justify="right", style="dim")
        stats.add_column()
        fps = f"  ({1000 / self.loop_interval_ms:.1f} fps)" if self.loop_interval_ms else ""
        stats.add_row("state", Text.assemble((label, f"bold {style}"), (fps, "dim")))
        stats.add_row("track", self._track_row())
        stats.add_row("pan", self._axis_row(self.pan_ticks, self.pan_limits, self.wide_pan_bounds))
        stats.add_row("tilt", self._axis_row(self.tilt_ticks, self.tilt_limits, self.wide_tilt_bounds))

        badges = Table.grid(padding=(0, 1))
        for _ in range(3):
            badges.add_column()
        rec_label = "REC" if self.recording_elapsed is None else (
            f"REC {self.recording_elapsed:4.1f}/{self.recording_total:.0f}s"
        )
        if self.clips_saved:
            rec_label += f" ({self.clips_saved} saved)"

        rtsp_label = "RTSP"
        if self.rtsp_active:
            rtsp_label += f" {self.rtsp_clients}c"
            if self.rtsp_fps is not None:
                rtsp_label += f" {self.rtsp_fps:4.1f}fps"

        pip_label = "PIP"
        if self.pip_active and self.pip_compose_ms is not None:
            pip_label += f" {self.pip_compose_ms:.1f}ms"

        badges.add_row(
            _badge(rec_label, self.recording, "red"),
            _badge(rtsp_label, self.rtsp_active, "blue"),
            _badge(pip_label, self.pip_active, "blue"),
        )

        sections = []
        if self.run_config:
            line = "  ".join(f"{k}={v}" for k, v in self.run_config.items())
            sections.append(Text(line, style="dim"))
            sections.append(Rule(style="grey50"))
        sections.append(Text("pipeline (per-camera detectors)", style="dim italic"))
        sections.append(pipeline)
        sections.append(Rule(style="grey50"))
        sections.append(stats)
        sections.append(badges)
        sections.append(Rule(style="grey50"))
        sections.append(Text("\n".join(self._log) or "…", style="dim"))
        return Panel(Group(*sections), title="pan-tilt-track", border_style="blue")

    def _track_row(self) -> Text:
        if self.locked_track_id is None:
            return Text("–", style="dim")
        conf = f"  {self.locked_confidence:.0%}" if self.locked_confidence is not None else ""
        return Text(f"#{self.locked_track_id}{conf}", style="bold green")

    def _axis_row(self, ticks: int | None, limits: JointLimits, bounds: tuple[int, int] | None) -> Text:
        row = _gauge(ticks, limits, bounds)
        if ticks is not None:
            row.append(f"  {ticks:5d} ticks  {ticks * DEG_PER_TICK:6.1f}°")
        return row
