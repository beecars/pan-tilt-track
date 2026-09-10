"""Picture-in-picture frame compositing: burn a smaller inset frame onto a
larger main frame. Pure OpenCV/numpy logic, no camera or GStreamer
dependency, so it's usable with any pair of same-shaped-dtype BGR frames."""

from __future__ import annotations

import cv2
import numpy as np


def compose_pip(
    main_frame: np.ndarray,
    pip_frame: np.ndarray,
    scale: float = 0.25,
    margin: int = 16,
    border: int = 2,
    border_color: tuple[int, int, int] = (255, 255, 255),
) -> np.ndarray:
    """Return a copy of `main_frame` with `pip_frame` scaled down and
    composited into the bottom-left corner.

    `scale` is the inset's width as a fraction of the main frame's width;
    the inset's height follows `pip_frame`'s own aspect ratio.
    """
    main_h, main_w = main_frame.shape[:2]
    pip_h, pip_w = pip_frame.shape[:2]

    inset_w = max(1, int(main_w * scale))
    inset_h = max(1, int(pip_h * (inset_w / pip_w)))
    inset = cv2.resize(pip_frame, (inset_w, inset_h))

    if border > 0:
        inset = cv2.copyMakeBorder(
            inset, border, border, border, border, cv2.BORDER_CONSTANT, value=border_color
        )

    out = main_frame.copy()
    x0 = margin
    y0 = main_h - margin - inset.shape[0]
    x1, y1 = x0 + inset.shape[1], y0 + inset.shape[0]
    out[y0:y1, x0:x1] = inset
    return out
