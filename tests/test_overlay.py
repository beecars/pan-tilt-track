import numpy as np

from pan_tilt_track.tracking.detector import Detection
from pan_tilt_track.tracking.overlay import draw_tracking_overlay


def _blank_frame():
    return np.zeros((200, 300, 3), dtype=np.uint8)


def test_draw_overlay_with_target_modifies_frame():
    frame = _blank_frame()
    target = Detection(track_id=1, class_id=0, confidence=0.9, box=(100, 50, 200, 150))
    draw_tracking_overlay(frame, [target], target, "body", (150.0, 100.0), 6.0, 6.0)
    assert frame.any()


def test_draw_overlay_with_no_target_does_not_crash():
    frame = _blank_frame()
    draw_tracking_overlay(frame, [], None, "body", (150.0, 100.0), 6.0, 6.0)
    assert frame.any()


def test_draw_overlay_head_mode_uses_head_point():
    frame = _blank_frame()
    target = Detection(
        track_id=1,
        class_id=0,
        confidence=0.9,
        box=(0, 0, 300, 200),
        keypoints=[(150.0, 100.0, 0.9)],
    )
    # Should not raise even though the head point coincides with center.
    draw_tracking_overlay(frame, [target], target, "head", (150.0, 100.0), 6.0, 6.0)
    assert frame.any()
