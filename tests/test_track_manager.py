from pan_tilt_track.tracking.detector import Detection
from pan_tilt_track.tracking.track_manager import TrackManager


def _at(track_id, cx, cy):
    return Detection(track_id=track_id, class_id=0, confidence=0.9, box=(cx - 5, cy - 5, cx + 5, cy + 5))


def test_locks_onto_closest_when_nothing_locked():
    manager = TrackManager()
    frame_center = (100.0, 100.0)
    target = manager.update([_at(1, 10, 10), _at(2, 100, 100)], frame_center)
    assert target.track_id == 2
    assert manager.locked_track_id == 2


def test_stays_locked_even_if_no_longer_closest():
    manager = TrackManager()
    frame_center = (100.0, 100.0)
    manager.update([_at(1, 100, 100)], frame_center)
    target = manager.update([_at(1, 10, 10), _at(2, 100, 100)], frame_center)
    assert target.track_id == 1
    assert manager.locked_track_id == 1


def test_relocks_when_locked_id_disappears():
    manager = TrackManager()
    frame_center = (100.0, 100.0)
    manager.update([_at(1, 100, 100)], frame_center)
    target = manager.update([_at(2, 100, 100)], frame_center)
    assert target.track_id == 2
    assert manager.locked_track_id == 2


def test_no_detections_clears_lock():
    manager = TrackManager()
    frame_center = (100.0, 100.0)
    manager.update([_at(1, 100, 100)], frame_center)
    target = manager.update([], frame_center)
    assert target is None
    assert manager.locked_track_id is None
