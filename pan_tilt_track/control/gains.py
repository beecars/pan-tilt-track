"""Tuned pan/tilt proportional-gain constants for TrackingLoop, shared by
scripts/run_tracker.py and scripts/sign_check.py."""

DEADBAND_PX = 6.0
# Pan needs a NEGATIVE kp on this mount, tilt positive: mount-specific,
# not a copy-paste artifact. Re-verify with scripts/sign_check.py after
# any reassembly.
PAN_KP = -0.15
TILT_KP = 0.15
