"""Physical camera characteristics and mount orientation, loaded from a JSON
file on disk rather than hardcoded -- these are per-rig calibration facts
(which physical sensor is which lens, how it's mounted) that change on
reassembly, not fixed hardware specs. Same pattern the wide-camera handoff
plan uses for the pixel->angle calibration file."""

from __future__ import annotations

import json
from dataclasses import dataclass, fields
from pathlib import Path

DEFAULT_CAMERA_CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "cameras.json"


@dataclass(frozen=True)
class CameraConfig:
    sensor_id: int
    capture_width: int = 1920
    capture_height: int = 1080
    framerate: int = 30
    flip_method: int = 0  # nvvidconv enum: 0=none, 2=180 deg. Free (VIC HW).


@dataclass(frozen=True)
class CamerasConfig:
    wide: CameraConfig
    telephoto: CameraConfig


def load_cameras_config(path: Path | str = DEFAULT_CAMERA_CONFIG_PATH) -> CamerasConfig:
    path = Path(path)
    data = json.loads(path.read_text())
    known_fields = {f.name for f in fields(CameraConfig)}
    try:
        roles = {"wide": data["wide"], "telephoto": data["telephoto"]}
    except KeyError as e:
        raise ValueError(f"{path}: missing required camera role {e}") from e
    return CamerasConfig(
        **{
            role: CameraConfig(**{k: v for k, v in role_data.items() if k in known_fields})
            for role, role_data in roles.items()
        }
    )
