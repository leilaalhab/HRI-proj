"""
Flask-based MJPEG stream server.

Architecture
------------
A SharedState object holds the latest composite frame as JPEG bytes,
protected by a threading.Lock.  The Flask /video_feed endpoint reads
from it continuously.  Any thread (webcam, inference, simulation) calls
update_frame() to push a new frame.

Usage
-----
    from src.server.stream_server import StreamServer
    server = StreamServer()
    server.start()                     # daemon thread; non-blocking
    server.update_frame(composite_rgb) # call from your main loop
"""

import os
import time
import threading

import cv2
import numpy as np
from flask import Flask, Response, send_from_directory

import config

_STATIC_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "static",
)


class SharedState:
    """Thread-safe latest-frame buffer."""

    def __init__(self):
        self._jpeg: bytes = b""
        self._lock = threading.Lock()

    def write(self, jpeg_bytes: bytes) -> None:
        with self._lock:
            self._jpeg = jpeg_bytes

    def read(self) -> bytes:
        with self._lock:
            return self._jpeg


def _encode_jpeg(frame_rgb: np.ndarray) -> bytes:
    frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    _, buf = cv2.imencode(".jpg", frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return buf.tobytes()


def build_composite(
    scene_frame: np.ndarray,
    robot_frame: np.ndarray,
    posteriors: dict,
    targets: list,
    status_lines: list[str] | None = None,
) -> np.ndarray:
    """
    Assemble browser composite:

        [ scene_frame (webcam + overlays) | robot_frame | prob bars ]

    All panels are top-aligned on a dark background.

    Parameters
    ----------
    scene_frame  : RGB numpy array (H x W x 3) — webcam with overlays
    robot_frame  : RGB numpy array (H x W x 3) — PyBullet render
    posteriors   : {target_name: probability}
    targets      : list of Target objects (for colours and names)
    status_lines : optional list of strings drawn at bottom-left
    """
    h_s, w_s = scene_frame.shape[:2]
    h_r, w_r = robot_frame.shape[:2]

    bar_panel_w = 130
    total_h = max(h_s, h_r)
    total_w = w_s + w_r + bar_panel_w

    composite = np.full((total_h, total_w, 3), 25, dtype=np.uint8)  # dark bg

    # Scene panel
    composite[:h_s, :w_s] = scene_frame

    # Robot render panel
    y_off = (total_h - h_r) // 2
    composite[y_off : y_off + h_r, w_s : w_s + w_r] = robot_frame

    # Divider lines
    composite[:, w_s - 1 : w_s + 1] = 60
    composite[:, w_s + w_r - 1 : w_s + w_r + 1] = 60

    # Probability bars panel
    _draw_prob_bars(composite, posteriors, targets,
                    x_start=w_s + w_r, panel_w=bar_panel_w, panel_h=total_h)

    # Status text
    if status_lines:
        for i, line in enumerate(status_lines):
            cv2.putText(
                composite, line,
                (8, total_h - 12 - i * 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                (200, 200, 200), 1, cv2.LINE_AA,
            )

    return composite


def _draw_prob_bars(
    canvas: np.ndarray,
    posteriors: dict,
    targets: list,
    x_start: int,
    panel_w: int,
    panel_h: int,
) -> None:
    """Draw vertical probability bars for each target on the right panel."""
    n = len(targets)
    bar_w = 28
    bar_max_h = panel_h - 80
    spacing = panel_w // n

    # Panel label
    cv2.putText(canvas, "P(G|O)", (x_start + 20, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (160, 160, 160), 1)

    for i, t in enumerate(targets):
        prob = float(posteriors.get(t.name, 1.0 / n))
        bar_h = max(1, int(prob * bar_max_h))

        bx = x_start + i * spacing + (spacing - bar_w) // 2
        by_bottom = panel_h - 45

        color_bgr = (
            int(t.color_rgb[2] * 255),
            int(t.color_rgb[1] * 255),
            int(t.color_rgb[0] * 255),
        )

        # Background track
        cv2.rectangle(canvas,
                      (bx, panel_h - 45 - bar_max_h),
                      (bx + bar_w, by_bottom),
                      (50, 50, 50), -1)

        # Filled bar
        cv2.rectangle(canvas,
                      (bx, by_bottom - bar_h),
                      (bx + bar_w, by_bottom),
                      color_bgr, -1)

        # Target initial letter
        cv2.putText(canvas, t.name[0].upper(),
                    (bx + 9, by_bottom + 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_bgr, 2)

        # Probability value
        cv2.putText(canvas, f"{prob:.2f}",
                    (bx + 2, by_bottom - bar_h - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200, 200, 200), 1)


class StreamServer:
    """Wraps Flask and the shared frame buffer."""

    def __init__(self, port: int | None = None):
        self._port = port or config.FLASK_PORT
        self._state = SharedState()
        self._app = self._build_app()

    def _build_app(self) -> Flask:
        app = Flask(__name__, static_folder=None)

        state = self._state

        @app.route("/")
        def index():
            return send_from_directory(_STATIC_DIR, "index.html")

        @app.route("/video_feed")
        def video_feed():
            def generate():
                while True:
                    jpeg = state.read()
                    if jpeg:
                        yield (
                            b"--frame\r\n"
                            b"Content-Type: image/jpeg\r\n\r\n"
                            + jpeg
                            + b"\r\n"
                        )
                    time.sleep(1 / 30)

            return Response(
                generate(),
                mimetype="multipart/x-mixed-replace; boundary=frame",
            )

        return app

    def update_frame(self, frame_rgb: np.ndarray) -> None:
        """Push a new composite frame (RGB numpy array) to the stream."""
        self._state.write(_encode_jpeg(frame_rgb))

    def start(self) -> threading.Thread:
        """Start Flask in a daemon thread. Returns the thread."""
        thread = threading.Thread(
            target=lambda: self._app.run(
                host="0.0.0.0",
                port=self._port,
                debug=False,
                use_reloader=False,
            ),
            daemon=True,
            name="flask-server",
        )
        thread.start()
        print(f"  → Open http://localhost:{self._port} in your browser")
        return thread
