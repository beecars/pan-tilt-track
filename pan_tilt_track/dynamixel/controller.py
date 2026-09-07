"""Thin wrapper around dynamixel_sdk for the two-servo pan/tilt head.

Owns the port/packet handlers and the RAM writes (torque, profile
velocity/acceleration, goal position). Deliberately does not implement any
control law of its own -- the servo firmware's own position PID plus
Profile Velocity/Acceleration are the only "controller" in the loop. See
pan_tilt_track.control.gain for the outer (vision) loop that decides what
goal positions to write.
"""

from __future__ import annotations

import logging

import dynamixel_sdk as dxl

from .config import (
    ADDR_GOAL_POSITION,
    ADDR_PRESENT_POSITION,
    ADDR_PROFILE_ACCELERATION,
    ADDR_PROFILE_VELOCITY,
    ADDR_TORQUE_ENABLE,
    ADDR_VELOCITY_LIMIT,
    TORQUE_DISABLE,
    TORQUE_ENABLE,
    DynamixelConfig,
)

logger = logging.getLogger(__name__)

GOAL_POSITION_LEN = 4


class DynamixelWriteError(RuntimeError):
    """Raised when a write to a servo does not come back with success."""


class PanTiltController:
    def __init__(self, config: DynamixelConfig | None = None):
        self.config = config or DynamixelConfig()
        self.port_handler = dxl.PortHandler(self.config.port)
        self.packet_handler = dxl.PacketHandler(self.config.protocol_version)
        self._sync_write_goal = dxl.GroupSyncWrite(
            self.port_handler, self.packet_handler, ADDR_GOAL_POSITION, GOAL_POSITION_LEN
        )

    # -- lifecycle -----------------------------------------------------

    def connect(self) -> None:
        if not self.port_handler.openPort():
            raise RuntimeError(f"Failed to open port {self.config.port}")
        if not self.port_handler.setBaudRate(self.config.baudrate):
            raise RuntimeError(f"Failed to set baudrate {self.config.baudrate}")
        logger.info("Connected to %s @ %d", self.config.port, self.config.baudrate)

    def close(self) -> None:
        # dynamixel_sdk's closePort() assumes ser is set; guard against
        # closing a port that never successfully opened (e.g. connect()
        # raised before setBaudRate succeeded).
        if self.port_handler.ser is not None:
            self.port_handler.closePort()

    def __enter__(self) -> "PanTiltController":
        self.connect()
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # -- low-level writes ------------------------------------------------

    def _check(self, dxl_id: int, comm_result: int, error: int, what: str) -> None:
        if comm_result != dxl.COMM_SUCCESS:
            raise DynamixelWriteError(
                f"{what} failed for id={dxl_id}: {self.packet_handler.getTxRxResult(comm_result)}"
            )
        if error != 0:
            raise DynamixelWriteError(
                f"{what} reported hardware error for id={dxl_id}: "
                f"{self.packet_handler.getRxPacketError(error)}"
            )

    def set_torque(self, dxl_id: int, enabled: bool) -> None:
        value = TORQUE_ENABLE if enabled else TORQUE_DISABLE
        result, error = self.packet_handler.write1ByteTxRx(
            self.port_handler, dxl_id, ADDR_TORQUE_ENABLE, value
        )
        self._check(dxl_id, result, error, "set_torque")

    def write_velocity_limit(self, dxl_id: int, ticks_per_s: int) -> None:
        result, error = self.packet_handler.write4ByteTxRx(
            self.port_handler, dxl_id, ADDR_VELOCITY_LIMIT, ticks_per_s
        )
        self._check(dxl_id, result, error, "write_velocity_limit")

    def write_profile(self, dxl_id: int, velocity: int, acceleration: int) -> None:
        # Firmware ignores acceleration while velocity reads 0 -- velocity
        # must be written first.
        result, error = self.packet_handler.write4ByteTxRx(
            self.port_handler, dxl_id, ADDR_PROFILE_VELOCITY, velocity
        )
        self._check(dxl_id, result, error, "write_profile_velocity")
        result, error = self.packet_handler.write4ByteTxRx(
            self.port_handler, dxl_id, ADDR_PROFILE_ACCELERATION, acceleration
        )
        self._check(dxl_id, result, error, "write_profile_acceleration")

    def read_present_position(self, dxl_id: int) -> int:
        position, result, error = self.packet_handler.read4ByteTxRx(
            self.port_handler, dxl_id, ADDR_PRESENT_POSITION
        )
        self._check(dxl_id, result, error, "read_present_position")
        return position

    def sync_write_goal_positions(self, pan_position: int, tilt_position: int) -> None:
        """Write both goal positions in a single bus transaction."""
        self._sync_write_goal.clearParam()
        for dxl_id, position in (
            (self.config.pan_id, pan_position),
            (self.config.tilt_id, tilt_position),
        ):
            param = [
                dxl.DXL_LOBYTE(dxl.DXL_LOWORD(position)),
                dxl.DXL_HIBYTE(dxl.DXL_LOWORD(position)),
                dxl.DXL_LOBYTE(dxl.DXL_HIWORD(position)),
                dxl.DXL_HIBYTE(dxl.DXL_HIWORD(position)),
            ]
            if not self._sync_write_goal.addParam(dxl_id, param):
                raise DynamixelWriteError(f"sync_write addParam failed for id={dxl_id}")
        result = self._sync_write_goal.txPacket()
        if result != dxl.COMM_SUCCESS:
            raise DynamixelWriteError(
                f"sync_write_goal_positions failed: {self.packet_handler.getTxRxResult(result)}"
            )

    # -- bring-up ---------------------------------------------------------

    def initialize(self) -> None:
        """Apply EEPROM guardrails + RAM profile, then enable torque.

        Velocity Limit lives in EEPROM, which the firmware refuses to write
        while torque is enabled ("Writing or Reading is not available to
        target address!") -- so torque must be off first, even if a prior
        run left it on.
        """
        for dxl_id, limits in (
            (self.config.pan_id, self.config.pan_limits),
            (self.config.tilt_id, self.config.tilt_limits),
        ):
            self.set_torque(dxl_id, enabled=False)
            self.write_velocity_limit(dxl_id, limits.velocity_limit)
            self.write_profile(dxl_id, limits.profile_velocity, limits.profile_acceleration)
            self.set_torque(dxl_id, enabled=True)
        logger.info("Pan/tilt controller initialized")

    def shutdown(self) -> None:
        for dxl_id in (self.config.pan_id, self.config.tilt_id):
            self.set_torque(dxl_id, enabled=False)
