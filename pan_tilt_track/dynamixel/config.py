"""Bring-up values and control-table addresses for the XL330 pan/tilt head."""

from dataclasses import dataclass

# --- X-series (Protocol 2.0) control table addresses used here ---
ADDR_TORQUE_ENABLE = 64        # 1 byte, RAM
ADDR_POSITION_P_GAIN = 84      # 2 bytes, RAM
ADDR_PROFILE_ACCELERATION = 108  # 4 bytes, RAM (resets to 0 on power-up)
ADDR_PROFILE_VELOCITY = 112    # 4 bytes, RAM (resets to 0 on power-up)
ADDR_GOAL_POSITION = 116       # 4 bytes, RAM
ADDR_PRESENT_POSITION = 132    # 4 bytes, RAM
ADDR_VELOCITY_LIMIT = 44       # 4 bytes, EEPROM
ADDR_MAX_POSITION_LIMIT = 48   # 4 bytes, EEPROM
ADDR_MIN_POSITION_LIMIT = 52   # 4 bytes, EEPROM

TORQUE_ENABLE = 1
TORQUE_DISABLE = 0

TICKS_PER_REV = 4096  # 0.088 deg/tick
RPM_PER_VELOCITY_UNIT = 0.229  # XL330 Velocity Limit / Profile Velocity unit


def rpm_to_velocity_units(rpm: float) -> int:
    return round(rpm / RPM_PER_VELOCITY_UNIT)


VELOCITY_CAP_RPM = 60
VELOCITY_LIMIT_UNITS = rpm_to_velocity_units(VELOCITY_CAP_RPM)  # ~262 units


@dataclass(frozen=True)
class JointLimits:
    min_position: int
    max_position: int
    velocity_limit: int
    profile_velocity: int
    profile_acceleration: int


@dataclass(frozen=True)
class DynamixelConfig:
    port: str = "/dev/ttyUSB0"
    baudrate: int = 57600
    protocol_version: float = 2.0

    pan_id: int = 1
    tilt_id: int = 2

    pan_limits: JointLimits = JointLimits(
        min_position=490,
        max_position=3550,
        velocity_limit=VELOCITY_LIMIT_UNITS,
        profile_velocity=200,
        profile_acceleration=30,
    )

    tilt_limits: JointLimits = JointLimits(
        min_position=2048,
        max_position=3246,
        velocity_limit=VELOCITY_LIMIT_UNITS,
        profile_velocity=200,
        profile_acceleration=30,
    )
