from pan_tilt_track.tracking.detector import Detection
from pan_tilt_track.tracking.target import select_target


def _body_only(track_id, box):
    return Detection(track_id=track_id, class_id=0, confidence=0.9, box=box)


def _with_head(track_id, box, nose=(50.0, 10.0, 0.9)):
    return Detection(track_id=track_id, class_id=0, confidence=0.9, box=box, keypoints=[nose])


def test_target_point_body_mode_ignores_keypoints():
    d = _with_head(1, (0, 0, 100, 100), nose=(90.0, 5.0, 0.9))
    assert d.target_point("body") == d.center


def test_head_point_uses_confident_keypoints():
    d = Detection(
        track_id=1,
        class_id=0,
        confidence=0.9,
        box=(0, 0, 100, 100),
        keypoints=[(10.0, 20.0, 0.9), (30.0, 20.0, 0.9)],
    )
    assert d.head_point == (20.0, 20.0)


def test_head_point_ignores_low_confidence_keypoints():
    d = Detection(
        track_id=1,
        class_id=0,
        confidence=0.9,
        box=(0, 0, 100, 100),
        keypoints=[(10.0, 20.0, 0.05)],
    )
    assert d.head_point is None


def test_target_point_head_mode_falls_back_to_body_center():
    d = _body_only(1, (0, 0, 100, 100))
    assert d.target_point("head") == d.center


def test_select_target_head_mode_prefers_closest_head_point():
    near_head_far_body = Detection(
        track_id=1, class_id=0, confidence=0.9, box=(0, 0, 40, 200),
        keypoints=[(90.0, 100.0, 0.9)],
    )
    far_head_near_body = Detection(
        track_id=2, class_id=0, confidence=0.9, box=(60, 0, 140, 200),
        keypoints=[(10.0, 100.0, 0.9)],
    )
    frame_center = (100.0, 100.0)
    target = select_target([near_head_far_body, far_head_near_body], frame_center, mode="head")
    assert target.track_id == 1
