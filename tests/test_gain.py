from pan_tilt_track.control.gain import ProportionalGain


def test_within_deadband_returns_zero():
    gain = ProportionalGain(kp=1.0, deadband_px=5.0)
    assert gain.compute(4.9) == 0
    assert gain.compute(-4.9) == 0
    assert gain.compute(0) == 0


def test_at_deadband_boundary_is_active():
    gain = ProportionalGain(kp=1.0, deadband_px=5.0)
    assert gain.compute(5.0) == 5


def test_scales_with_kp():
    gain = ProportionalGain(kp=0.5, deadband_px=1.0)
    assert gain.compute(10.0) == 5
    assert gain.compute(-10.0) == -5


def test_rounds_to_nearest_tick():
    gain = ProportionalGain(kp=0.33, deadband_px=1.0)
    assert gain.compute(10.0) == round(3.3)
