"""YoloDetector: thin wrapper around Ultralytics YOLO's track() with
ByteTrack, returning a list of Detection objects. Accepts either a
detection model (boxes only) or a `*-pose.pt` model (boxes + COCO
keypoints)."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

import torch
from ultralytics import YOLO

logger = logging.getLogger(__name__)

# COCO pose keypoint indices that make up the head.
HEAD_KEYPOINT_INDICES = (0, 1, 2, 3, 4)  # nose, left_eye, right_eye, left_ear, right_ear
HEAD_KEYPOINT_CONF_THRESHOLD = 0.3

COCO_CLASS_BIRD = 14
COCO_CLASS_CAT = 15
COCO_CLASS_DOG = 16
ANIMAL_CLASS_IDS = (COCO_CLASS_BIRD, COCO_CLASS_CAT, COCO_CLASS_DOG)

TRACKER_CONFIG = "bytetrack.yaml"


@dataclass
class Detection:
    track_id: int | None
    class_id: int
    confidence: float
    # (x1, y1, x2, y2) in pixel coordinates
    box: tuple[float, float, float, float]
    # (x, y, conf) per COCO keypoint, or None for a boxes-only model
    keypoints: list[tuple[float, float, float]] | None = None

    @property
    def center(self) -> tuple[float, float]:
        x1, y1, x2, y2 = self.box
        return (x1 + x2) / 2, (y1 + y2) / 2

    @property
    def head_point(self) -> tuple[float, float] | None:
        """Centroid of the visible head keypoints, or None if this
        detection has no keypoints or none are confident enough (e.g. the
        head is turned away or occluded)."""
        if not self.keypoints:
            return None
        visible = [
            (x, y)
            for i, (x, y, conf) in enumerate(self.keypoints)
            if i in HEAD_KEYPOINT_INDICES and conf >= HEAD_KEYPOINT_CONF_THRESHOLD
        ]
        if not visible:
            return None
        xs, ys = zip(*visible)
        return sum(xs) / len(xs), sum(ys) / len(ys)

    def target_point(self, mode: str = "body") -> tuple[float, float]:
        """Point to aim the pan/tilt loop at. mode="head" falls back to the
        body center when no head keypoints are visible this frame."""
        if mode == "head":
            head = self.head_point
            if head is not None:
                return head
        return self.center


class YoloDetector:
    def __init__(
        self,
        model_path: str = "yolo26n.pt",
        classes: list[int] | None = None,
        imgsz: int | None = None,
        conf: float | None = None,
    ):
        self.model = YOLO(model_path)
        cuda_available = torch.cuda.is_available()
        is_pytorch_model = model_path.endswith(".pt")
        if cuda_available and is_pytorch_model:
            self.model.to("cuda")
            device_name = torch.cuda.get_device_name(0)
            logger.info("CUDA available: %s -- inference on GPU", device_name)
        elif not cuda_available:
            logger.warning(
                "CUDA NOT available (torch.cuda.is_available() is False) -- "
                "%s will run on CPU, which is much slower than GPU inference",
                model_path,
            )
        logger.info(
            "Loaded %s (torch=%s, cuda_build=%s)",
            model_path,
            torch.__version__,
            torch.version.cuda,
        )
        self.classes = classes
        if imgsz is not None and not is_pytorch_model:
            logger.warning(
                "imgsz=%s ignored for %s -- exported formats use the imgsz baked in at export time",
                imgsz,
                model_path,
            )
            imgsz = None
        self.imgsz = imgsz
        self.conf = conf
        self.last_timing: dict[str, float] = {}

    def track(self, frame, reset: bool = False) -> list[Detection]:
        """`reset=True` drops ByteTrack's internal track/Kalman state before
        this call -- required the first time a call is fed frames from a
        different camera than the previous call, since track_ids aren't
        namespaced per source and motion state from one camera is invalid
        for another."""
        t_start = time.perf_counter()
        extra_kwargs = {}
        if self.imgsz is not None:
            extra_kwargs["imgsz"] = self.imgsz
        if self.conf is not None:
            extra_kwargs["conf"] = self.conf
        results = self.model.track(
            frame,
            persist=not reset,
            classes=self.classes,
            tracker=TRACKER_CONFIG,
            verbose=False,
            **extra_kwargs,
        )
        detections: list[Detection] = []
        result = results[0] if results else None
        boxes = result.boxes if result is not None else None

        if boxes is not None:
            ids = boxes.id.int().tolist() if boxes.id is not None else [None] * len(boxes)

            keypoints_per_box: list[list[tuple[float, float, float]] | None] = [None] * len(boxes)
            if result.keypoints is not None and result.keypoints.conf is not None:
                kp_xy = result.keypoints.xy.tolist()
                kp_conf = result.keypoints.conf.tolist()
                keypoints_per_box = [
                    [(x, y, c) for (x, y), c in zip(xy, conf)] for xy, conf in zip(kp_xy, kp_conf)
                ]

            for box_xyxy, cls, conf, track_id, keypoints in zip(
                boxes.xyxy.tolist(), boxes.cls.tolist(), boxes.conf.tolist(), ids, keypoints_per_box
            ):
                detections.append(
                    Detection(
                        track_id=track_id,
                        class_id=int(cls),
                        confidence=float(conf),
                        box=tuple(box_xyxy),
                        keypoints=keypoints,
                    )
                )

        total_ms = (time.perf_counter() - t_start) * 1000
        speed = getattr(result, "speed", None) or {}
        preprocess_ms = speed.get("preprocess", 0.0)
        nn_inference_ms = speed.get("inference", 0.0)
        postprocess_ms = speed.get("postprocess", 0.0)
        self.last_timing = {
            "preprocess_ms": preprocess_ms,
            "nn_inference_ms": nn_inference_ms,
            "postprocess_ms": postprocess_ms,
            # Remainder: ByteTrack update + this method's own parsing.
            "track_ms": max(0.0, total_ms - (preprocess_ms + nn_inference_ms + postprocess_ms)),
            "total_ms": total_ms,
        }
        return detections
