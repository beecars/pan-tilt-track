"""Outer-loop proportional-gain tuning constants, empirically verified
against real hardware with scripts/sign_check.py (see that file for the
verification procedure). Shared by scripts/run_tracker.py and
scripts/sign_check.py so a re-tune only has to happen in one place.

Do not use these as a template to unify pan_kp and tilt_kp into one
value -- pan needs a NEGATIVE kp on this mount, tilt positive; that sign
difference is mount-specific, not a copy-paste artifact.
"""

DEADBAND_PX = 6.0
PAN_KP = -0.15
TILT_KP = 0.15
