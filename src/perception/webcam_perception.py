"""
Webcam-based hand tracking using MediaPipe Hands.

Provides WebcamObservation objects with the same interface as
SimulatedObservation, so BayesianGoalInference.update() accepts both.
"""
import time

import cv2
import mediapipe as mp
import numpy as np

import config
from src.scene.targets import CANVAS_W, CANVAS_H

# Wrist (0) is more stable than fingertip (8) under partial occlusion.
_WRIST = 0


class WebcamObservation:
    """Hand observation from webcam — duck-type compatible with SimulatedObservation."""
    __slots__ = ("position", "velocity", "timestamp")

    def __init__(self, position: np.ndarray, velocity: np.ndarray, timestamp: float):
        self.position  = position   # np.ndarray [x, y] pixel coords
        self.velocity  = velocity   # np.ndarray [vx, vy] px/s
        self.timestamp = timestamp  # seconds from trial start


class WebcamPerception:
    """
    Captures webcam frames, runs MediaPipe Hands, smooths position, estimates velocity.

    Usage
    -----
    perception = WebcamPerception()
    obs = perception.update()   # None when no hand visible
    bgr = perception.get_frame_bgr()
    perception.reset_trial()    # call at the start of each new trial
    perception.release()        # call on shutdown
    """

    def __init__(self, camera_index: int = config.WEBCAM_INDEX):
        self._cap = cv2.VideoCapture(camera_index)
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH,  CANVAS_W)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CANVAS_H)

        self._hands = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.5,
            model_complexity=0,     # fastest; sufficient for single-hand tracking
        )

        self._smooth_pos: np.ndarray | None = None
        self._prev_smooth: np.ndarray | None = None
        self._prev_time:   float | None = None
        self._trial_start: float = time.time()

        self._last_bgr:     np.ndarray | None = None
        self._last_results = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(self) -> WebcamObservation | None:
        """
        Capture one frame, detect wrist, apply smoothing, estimate velocity.
        Returns WebcamObservation or None when no hand is visible.

        The frame is flipped horizontally (mirror) before processing so
        landmark x-coordinates match the visually displayed image.
        """
        ret, bgr = self._cap.read()
        if not ret:
            return None

        bgr = cv2.flip(bgr, 1)
        bgr = cv2.resize(bgr, (CANVAS_W, CANVAS_H))
        self._last_bgr = bgr

        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        self._last_results = self._hands.process(rgb)

        now = time.time()
        t   = now - self._trial_start

        if not self._last_results.multi_hand_landmarks:
            return None

        lm  = self._last_results.multi_hand_landmarks[0].landmark[_WRIST]
        raw = np.array([lm.x * CANVAS_W, lm.y * CANVAS_H], dtype=float)

        # Exponential smoothing: new_pos = α·raw + (1-α)·prev
        if self._smooth_pos is None:
            self._smooth_pos = raw.copy()
        else:
            a = config.SMOOTHING_ALPHA
            self._smooth_pos = a * raw + (1.0 - a) * self._smooth_pos

        # Velocity from last two smoothed positions
        if self._prev_smooth is not None and self._prev_time is not None:
            dt  = max(now - self._prev_time, 1e-6)
            vel = (self._smooth_pos - self._prev_smooth) / dt
        else:
            vel = np.zeros(2)

        self._prev_smooth = self._smooth_pos.copy()
        self._prev_time   = now

        return WebcamObservation(
            position  = self._smooth_pos.copy(),
            velocity  = vel.copy(),
            timestamp = round(t, 4),
        )

    def get_frame_bgr(self) -> np.ndarray | None:
        return self._last_bgr

    def get_landmark_results(self):
        """Return raw MediaPipe results for custom overlay drawing."""
        return self._last_results

    def reset_trial(self) -> None:
        """Clear smoothing state at the start of each new trial."""
        self._smooth_pos  = None
        self._prev_smooth = None
        self._prev_time   = None
        self._trial_start = time.time()

    def release(self) -> None:
        self._cap.release()
        self._hands.close()
