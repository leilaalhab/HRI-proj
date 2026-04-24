import numpy as np
from dataclasses import dataclass
from typing import Tuple


@dataclass
class Target:
    name: str
    position: np.ndarray        # [cx, cy] centre in pixel coords
    region: Tuple[int, int, int, int]  # (x_min, y_min, x_max, y_max)
    color_rgb: Tuple[float, float, float]  # 0-1 floats for matplotlib
    color_bgr: Tuple[int, int, int]        # 0-255 ints for OpenCV
    label: str


# Canvas size matches typical webcam resolution so Stage 8 needs no remapping.
CANVAS_W = 640
CANVAS_H = 480

# Hand starting position (bottom-centre of the canvas)
HAND_START = np.array([320, 420], dtype=float)

_HALF = 50  # half-width/height of each target rectangle


def get_targets() -> list[Target]:
    return [
        Target(
            name="red",
            position=np.array([120, 180], dtype=float),
            region=(120 - _HALF, 180 - _HALF, 120 + _HALF, 180 + _HALF),
            color_rgb=(0.85, 0.15, 0.15),
            color_bgr=(30, 30, 200),
            label="Red Block",
        ),
        Target(
            name="blue",
            position=np.array([320, 180], dtype=float),
            region=(320 - _HALF, 180 - _HALF, 320 + _HALF, 180 + _HALF),
            color_rgb=(0.15, 0.35, 0.85),
            color_bgr=(200, 80, 30),
            label="Blue Block",
        ),
        Target(
            name="green",
            position=np.array([520, 180], dtype=float),
            region=(520 - _HALF, 180 - _HALF, 520 + _HALF, 180 + _HALF),
            color_rgb=(0.1, 0.7, 0.2),
            color_bgr=(50, 180, 50),
            label="Green Block",
        ),
    ]
