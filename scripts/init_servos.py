#!/usr/bin/env python3
"""One-time/per-boot bring-up: open the port, write EEPROM guardrails and
RAM profile (velocity/acceleration reset every power-cycle), enable torque,
and confirm both servos respond. Servo-only, no camera dependency."""

import logging
import sys

from pan_tilt_track.dynamixel import DynamixelConfig, PanTiltController

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)


def main() -> int:
    config = DynamixelConfig()
    controller = PanTiltController(config)
    try:
        controller.connect()
        controller.initialize()
        pan_pos = controller.read_present_position(config.pan_id)
        tilt_pos = controller.read_present_position(config.tilt_id)
        logger.info("Pan present position: %d", pan_pos)
        logger.info("Tilt present position: %d", tilt_pos)
    except Exception:
        logger.exception("Servo init failed")
        return 1
    finally:
        controller.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
