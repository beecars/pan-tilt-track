"""Draws tracking diagnostics onto a frame for RTSP viewing: every
detection box, the locked target highlighted, the frame-center crosshair,
the deadband boundary, and the current pixel-error vector."""

from __future__ import annotations

import cv2
import numpy as np

from .detector import Detection

COLOR_TARGET = (0, 255, 0)
COLOR_OTHER = (128, 128, 128)
COLOR_CROSSHAIR = (0, 165, 255)
COLOR_ERROR = (0, 0, 255)
COLOR_TEXT = (255, 255, 255)

# -- debug HUD palette, kept separate from the tracking-overlay colors above --
HUD_FONT = cv2.FONT_HERSHEY_SIMPLEX
HUD_SCALE_HEAD = 0.5
HUD_SCALE = 0.42
HUD_BG_COLOR = (30, 26, 22)
HUD_BG_ALPHA = 0.6
HUD_COLOR_HEAD = (255, 255, 255)
HUD_COLOR_TEXT = (215, 215, 215)
HUD_COLOR_DIM = (120, 120, 120)
HUD_COLOR_WARN = (0, 165, 255)  # slowest detect-side stage this frame
HUD_COLOR_ACCENT = (0, 220, 0)  # servo state -- echoes COLOR_TARGET


def draw_tracking_overlay(
    frame,
    detections: list[Detection],
    target: Detection | None,
    target_mode: str,
    frame_center: tuple[float, float],
    deadband_x: float,
    deadband_y: float,
) -> None:
    """Mutates frame in place."""
    cx, cy = int(frame_center[0]), int(frame_center[1])

    cv2.drawMarker(frame, (cx, cy), COLOR_CROSSHAIR, cv2.MARKER_CROSS, 20, 2)
    cv2.ellipse(frame, (cx, cy), (int(deadband_x), int(deadband_y)), 0, 0, 360, COLOR_CROSSHAIR, 1)

    # Detection boxes are body mode's aim reference; not relevant in head mode.
    if target_mode == "body":
        for d in detections:
            is_target = target is not None and d.track_id == target.track_id
            color = COLOR_TARGET if is_target else COLOR_OTHER
            x1, y1, x2, y2 = (int(v) for v in d.box)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2 if is_target else 1)
            cv2.putText(
                frame,
                f"id={d.track_id} {d.confidence:.2f}",
                (x1, max(0, y1 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1,
            )

    if target is None:
        cv2.putText(frame, "NO TARGET", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, COLOR_ERROR, 2)
        return

    tx, ty = target.target_point(target_mode)
    tx_i, ty_i = int(tx), int(ty)
    cv2.drawMarker(frame, (tx_i, ty_i), COLOR_TARGET, cv2.MARKER_TILTED_CROSS, 14, 2)
    cv2.line(frame, (cx, cy), (tx_i, ty_i), COLOR_ERROR, 2)

    err_x, err_y = tx - cx, ty - cy
    cv2.putText(
        frame,
        f"err=({err_x:+.0f},{err_y:+.0f})",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        COLOR_TEXT,
        2,
    )

    if target_mode == "head":
        head_locked = target.head_point is not None
        label = "HEAD" if head_locked else "BODY (no head kpts)"
        label_color = COLOR_TARGET if head_locked else COLOR_CROSSHAIR
        cv2.putText(frame, label, (10, 54), cv2.FONT_HERSHEY_SIMPLEX, 0.6, label_color, 2)


def _ms(value: float | None) -> str:
    return f"{value:5.1f}" if value is not None else "  -- "


def draw_debug_hud(frame, debug: dict) -> None:
    """Draws a per-frame timing/servo diagnostics panel in the top-right corner.

    `debug` keys (all optional -- missing/None entries render as "--"):
    frame_interval_ms, preprocess_ms, nn_inference_ms, postprocess_ms,
    track_ms, detect_total_ms, draw_ms, servo_io_ms, track_id,
    num_detections, pan_position, tilt_position, pan_delta, tilt_delta.

    track_ms is not from Ultralytics' own timers (see YoloDetector.track()) --
    it's the ByteTrack update + result-parsing overhead. The slowest
    detect-side stage each frame is highlighted as the bottleneck.
    """
    frame_interval_ms = debug.get("frame_interval_ms")
    fps = 1000.0 / frame_interval_ms if frame_interval_ms else None

    stage_labels = ("pre", "nn", "post", "track")
    stage_values = {
        "pre": debug.get("preprocess_ms"),
        "nn": debug.get("nn_inference_ms"),
        "post": debug.get("postprocess_ms"),
        "track": debug.get("track_ms"),
    }
    bottleneck = max(
        (label for label in stage_labels if stage_values[label] is not None),
        key=lambda label: stage_values[label],
        default=None,
    )

    # Each entry is (text, color, font_scale), or None for a divider rule.
    rows: list[tuple[str, tuple[int, int, int], float] | None] = []

    rows.append(
        (
            f"{fps:4.1f} fps  {frame_interval_ms:5.1f} ms/loop" if fps else "  --  fps",
            HUD_COLOR_HEAD,
            HUD_SCALE_HEAD,
        )
    )
    rows.append(None)
    for label in stage_labels:
        color = HUD_COLOR_WARN if label == bottleneck else HUD_COLOR_TEXT
        rows.append((f"{label:<6}{_ms(stage_values[label])} ms", color, HUD_SCALE))
    rows.append((f"{'=detect':<6}{_ms(debug.get('detect_total_ms'))} ms", HUD_COLOR_DIM, HUD_SCALE))
    rows.append(None)
    rows.append((f"{'draw':<6}{_ms(debug.get('draw_ms'))} ms", HUD_COLOR_TEXT, HUD_SCALE))
    rows.append((f"{'servo':<6}{_ms(debug.get('servo_io_ms'))} ms", HUD_COLOR_TEXT, HUD_SCALE))
    rows.append(None)

    track_id = debug.get("track_id")
    rows.append(
        (
            f"dets {debug.get('num_detections', 0)}   id {track_id if track_id is not None else '-'}",
            HUD_COLOR_TEXT,
            HUD_SCALE,
        )
    )

    pan_position = debug.get("pan_position")
    if pan_position is not None:
        pan_line = f"pan  {pan_position:5d}  d{debug.get('pan_delta') or 0:+4d}"
        rows.append((pan_line, HUD_COLOR_ACCENT, HUD_SCALE))
    else:
        rows.append(("pan     --", HUD_COLOR_DIM, HUD_SCALE))

    tilt_position = debug.get("tilt_position")
    if tilt_position is not None:
        tilt_line = f"tilt {tilt_position:5d}  d{debug.get('tilt_delta') or 0:+4d}"
        rows.append((tilt_line, HUD_COLOR_ACCENT, HUD_SCALE))
    else:
        rows.append(("tilt    --", HUD_COLOR_DIM, HUD_SCALE))

    _draw_hud_panel(frame, rows)


def _draw_hud_panel(
    frame, rows: list[tuple[str, tuple[int, int, int], float] | None]
) -> None:
    """Renders `rows` (text/color/font_scale triples, or None for a divider
    rule) as a translucent panel anchored to the frame's top-right corner,
    sized to fit its content."""
    height, width = frame.shape[:2]
    pad = 8
    line_h = 16
    divider_h = 8

    text_rows = [row for row in rows if row is not None]
    panel_w = pad * 2 + max(
        cv2.getTextSize(text, HUD_FONT, scale, 1)[0][0] for text, _, scale in text_rows
    )
    panel_h = pad * 2 + sum(line_h if row is not None else divider_h for row in rows)

    x0 = max(0, width - panel_w - 10)
    y0 = 10
    x1 = min(width, x0 + panel_w)
    y1 = min(height, y0 + panel_h)

    panel = frame[y0:y1, x0:x1]
    tint = np.full_like(panel, HUD_BG_COLOR)
    cv2.addWeighted(tint, HUD_BG_ALPHA, panel, 1 - HUD_BG_ALPHA, 0, dst=panel)

    text_x = x0 + pad
    y = y0 + pad + line_h - 5
    for row in rows:
        if row is None:
            rule_y = y - line_h + divider_h // 2 + 4
            cv2.line(frame, (text_x, rule_y), (x1 - pad, rule_y), HUD_COLOR_DIM, 1, cv2.LINE_AA)
            y += divider_h
            continue
        text, color, scale = row
        cv2.putText(frame, text, (text_x, y), HUD_FONT, scale, color, 1, cv2.LINE_AA)
        y += line_h
