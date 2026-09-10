#!/usr/bin/env python3
"""One-time/per-boot servo bring-up: opens the port, writes EEPROM/RAM
config, enables torque, and reads back both present positions."""

import logging
import sys

from pan_tilt_track.dynamixel import PanTiltController, load_dynamixel_config

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)


def main() -> int:
    config = load_dynamixel_config()
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
