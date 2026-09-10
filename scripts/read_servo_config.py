#!/usr/bin/env python3
"""Read back the actual EEPROM/RAM values currently on both servos, so
config.py can be populated with real numbers instead of placeholders.

Read-only: does not write anything. Safe to run any time, torque on or off.
"""

import sys

from pan_tilt_track.dynamixel.config import (
    ADDR_MAX_POSITION_LIMIT,
    ADDR_MIN_POSITION_LIMIT,
    ADDR_POSITION_P_GAIN,
    ADDR_PROFILE_ACCELERATION,
    ADDR_PROFILE_VELOCITY,
    ADDR_VELOCITY_LIMIT,
    load_dynamixel_config,
)
from pan_tilt_track.dynamixel.controller import PanTiltController

ADDR_HOMING_OFFSET = 20


def read4(controller, dxl_id, addr):
    value, result, error = controller.packet_handler.read4ByteTxRx(
        controller.port_handler, dxl_id, addr
    )
    return value


def read2(controller, dxl_id, addr):
    value, result, error = controller.packet_handler.read2ByteTxRx(
        controller.port_handler, dxl_id, addr
    )
    return value


def dump(controller, dxl_id, name):
    print(f"--- {name} (id={dxl_id}) ---")
    print(f"  Min Position Limit (52):     {read4(controller, dxl_id, ADDR_MIN_POSITION_LIMIT)}")
    print(f"  Max Position Limit (48):     {read4(controller, dxl_id, ADDR_MAX_POSITION_LIMIT)}")
    print(f"  Homing Offset (20):          {read4(controller, dxl_id, ADDR_HOMING_OFFSET)}")
    print(f"  Velocity Limit (44):         {read4(controller, dxl_id, ADDR_VELOCITY_LIMIT)}")
    print(f"  Profile Velocity (112):      {read4(controller, dxl_id, ADDR_PROFILE_VELOCITY)}")
    print(f"  Profile Acceleration (108):  {read4(controller, dxl_id, ADDR_PROFILE_ACCELERATION)}")
    print(f"  Position P Gain (84):        {read2(controller, dxl_id, ADDR_POSITION_P_GAIN)}")
    print(f"  Present Position (132):      {controller.read_present_position(dxl_id)}")


def main() -> int:
    config = load_dynamixel_config()
    controller = PanTiltController(config)
    try:
        controller.connect()
        dump(controller, config.pan_id, "pan")
        dump(controller, config.tilt_id, "tilt")
    finally:
        controller.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
