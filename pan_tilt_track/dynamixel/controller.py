"""PanTiltController: dynamixel_sdk wrapper owning the port/packet
handlers and RAM writes (torque, profile velocity/acceleration, goal
position) for the two-servo pan/tilt head."""

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
    load_dynamixel_config,
)

logger = logging.getLogger(__name__)

GOAL_POSITION_LEN = 4
COMM_MAX_ATTEMPTS = 3


class DynamixelWriteError(RuntimeError):
    """Raised when a write to a servo does not come back with success."""


class PanTiltController:
    def __init__(self, config: DynamixelConfig | None = None):
        self.config = config or load_dynamixel_config()
        self.port_handler = dxl.PortHandler(self.config.port)
        self.packet_handler = dxl.PacketHandler(self.config.protocol_version)
        self._sync_write_goal = dxl.GroupSyncWrite(
            self.port_handler, self.packet_handler, ADDR_GOAL_POSITION, GOAL_POSITION_LEN
        )

    # lifecycle

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

    # low-level writes

    def _check(self, dxl_id: int | None, comm_result: int, error: int, what: str) -> None:
        id_label = dxl_id if dxl_id is not None else "pan+tilt"
        if comm_result != dxl.COMM_SUCCESS:
            raise DynamixelWriteError(
                f"{what} failed for id={id_label}: {self.packet_handler.getTxRxResult(comm_result)}"
            )
        if error != 0:
            raise DynamixelWriteError(
                f"{what} reported hardware error for id={id_label}: "
                f"{self.packet_handler.getRxPacketError(error)}"
            )

    def _retry_comm(self, dxl_id: int | None, what: str, attempt_fn, max_attempts: int = COMM_MAX_ATTEMPTS):
        """Retries a single Dynamixel transaction up to `max_attempts` times
        before raising. `attempt_fn` takes no args and returns either
        `(result, error)` (writes) or `(value, result, error)` (reads);
        the leading value, if any, is returned on success. `dxl_id` is used
        only for logging - pass None for a multi-ID transaction (e.g. a
        sync write) that has no single target and no per-ID error byte."""
        id_label = dxl_id if dxl_id is not None else "pan+tilt"
        for attempt in range(1, max_attempts + 1):
            *value, result, error = attempt_fn()
            if result == dxl.COMM_SUCCESS and error == 0:
                return value[0] if value else None
            if attempt < max_attempts:
                logger.warning(
                    "%s transient failure for id=%s (attempt %d/%d): %s",
                    what,
                    id_label,
                    attempt,
                    max_attempts,
                    self.packet_handler.getTxRxResult(result)
                    if result != dxl.COMM_SUCCESS
                    else self.packet_handler.getRxPacketError(error),
                )
        self._check(dxl_id, result, error, what)

    def set_torque(self, dxl_id: int, enabled: bool) -> None:
        value = TORQUE_ENABLE if enabled else TORQUE_DISABLE
        self._retry_comm(
            dxl_id, "set_torque", lambda: self.packet_handler.write1ByteTxRx(
                self.port_handler, dxl_id, ADDR_TORQUE_ENABLE, value
            )
        )

    def write_velocity_limit(self, dxl_id: int, ticks_per_s: int) -> None:
        self._retry_comm(
            dxl_id, "write_velocity_limit", lambda: self.packet_handler.write4ByteTxRx(
                self.port_handler, dxl_id, ADDR_VELOCITY_LIMIT, ticks_per_s
            )
        )

    def write_profile(self, dxl_id: int, velocity: int, acceleration: int) -> None:
        # Firmware ignores acceleration while velocity reads 0: velocity
        # must be written first.
        self._retry_comm(
            dxl_id, "write_profile_velocity", lambda: self.packet_handler.write4ByteTxRx(
                self.port_handler, dxl_id, ADDR_PROFILE_VELOCITY, velocity
            )
        )
        self._retry_comm(
            dxl_id, "write_profile_acceleration", lambda: self.packet_handler.write4ByteTxRx(
                self.port_handler, dxl_id, ADDR_PROFILE_ACCELERATION, acceleration
            )
        )

    def read_present_position(self, dxl_id: int) -> int:
        return self._retry_comm(
            dxl_id, "read_present_position", lambda: self.packet_handler.read4ByteTxRx(
                self.port_handler, dxl_id, ADDR_PRESENT_POSITION
            )
        )

    def sync_write_goal_positions(self, pan_position: int, tilt_position: int) -> None:
        """Write both goal positions in a single bus transaction."""

        def attempt():
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
            # No per-ID error byte for a sync write - synthesize error=0 so
            # _retry_comm's (result, error) contract still applies.
            return self._sync_write_goal.txPacket(), 0

        self._retry_comm(None, "sync_write_goal_positions", attempt)

    # bring-up

    def initialize(self) -> None:
        """Apply EEPROM guardrails + RAM profile, then enable torque.

        Velocity Limit lives in EEPROM, which the firmware refuses to write
        while torque is enabled ("Writing or Reading is not available to
        target address!"), so torque must be off first, even if a prior
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
        """Disable torque on both joints. Each joint is attempted
        independently: a failure on one (even after retries) must not
        prevent trying to disable the other."""
        errors = []
        for dxl_id in (self.config.pan_id, self.config.tilt_id):
            try:
                self.set_torque(dxl_id, enabled=False)
            except DynamixelWriteError as e:
                logger.error("Failed to disable torque for id=%d: %s", dxl_id, e)
                errors.append(e)
        if errors:
            raise DynamixelWriteError(f"shutdown incomplete: {len(errors)} joint(s) failed to disable torque")
