"""Control-table addresses and per-rig servo configuration (port, IDs,
joint limits/profile) for the XL330 pan/tilt head, the latter loaded from
config/servos.json."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

# X-series (Protocol 2.0) control table addresses used here:
ADDR_TORQUE_ENABLE = 64        # 1 byte, RAM
ADDR_POSITION_D_GAIN = 80      # 2 bytes, RAM
ADDR_POSITION_I_GAIN = 82      # 2 bytes, RAM
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

DEFAULT_SERVO_CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "servos.json"


def rpm_to_velocity_units(rpm: float) -> int:
    return round(rpm / RPM_PER_VELOCITY_UNIT)


DEFAULT_POSITION_P_GAIN = 400  # XL330 firmware default
DEFAULT_POSITION_I_GAIN = 0    # XL330 firmware default (no integral action)


@dataclass(frozen=True)
class JointLimits:
    min_position: int
    max_position: int
    velocity_limit: int
    profile_velocity: int
    profile_acceleration: int
    position_p_gain: int = DEFAULT_POSITION_P_GAIN
    position_i_gain: int = DEFAULT_POSITION_I_GAIN


def clamp_position(value: int, limits: JointLimits) -> int:
    return max(limits.min_position, min(limits.max_position, value))


@dataclass(frozen=True)
class DynamixelConfig:
    port: str
    baudrate: int
    pan_id: int
    tilt_id: int
    pan_limits: JointLimits
    tilt_limits: JointLimits
    protocol_version: float = 2.0


def _joint_limits_from(joint: dict) -> JointLimits:
    return JointLimits(
        min_position=joint["min_position"],
        max_position=joint["max_position"],
        velocity_limit=rpm_to_velocity_units(joint["velocity_limit_rpm"]),
        profile_velocity=joint["profile_velocity"],
        profile_acceleration=joint["profile_acceleration"],
        position_p_gain=joint.get("position_p_gain", DEFAULT_POSITION_P_GAIN),
        position_i_gain=joint.get("position_i_gain", DEFAULT_POSITION_I_GAIN),
    )


def load_dynamixel_config(path: Path | str = DEFAULT_SERVO_CONFIG_PATH) -> DynamixelConfig:
    path = Path(path)
    data = json.loads(path.read_text())
    try:
        pan, tilt = data["pan"], data["tilt"]
    except KeyError as e:
        raise ValueError(f"{path}: missing required joint {e}") from e
    return DynamixelConfig(
        port=data["port"],
        baudrate=data["baudrate"],
        protocol_version=data.get("protocol_version", 2.0),
        pan_id=pan["id"],
        tilt_id=tilt["id"],
        pan_limits=_joint_limits_from(pan),
        tilt_limits=_joint_limits_from(tilt),
    )
