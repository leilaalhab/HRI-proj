import numpy as np
from dataclasses import dataclass

from src.scene.targets import Target, HAND_START


@dataclass
class SimulatedObservation:
    position: np.ndarray   # [x, y] in pixel coords
    velocity: np.ndarray   # [vx, vy] pixels/second
    timestamp: float       # seconds from trial start


def _minimum_jerk_pos(x0, xf, tau):
    """Minimum-jerk position profile. tau = t/D in [0, 1]."""
    s = 10 * tau**3 - 15 * tau**4 + 6 * tau**5
    return x0 + (xf - x0) * s


def generate_trajectory(
    target: Target,
    start: np.ndarray = HAND_START,
    duration: float = 2.0,
    dt: float = 0.05,
    noise_std: float = 0.0,
) -> list[SimulatedObservation]:
    """
    Generate a smooth simulated hand trajectory from start toward target
    using the minimum-jerk profile.  noise_std=0 keeps it clean for Stage 3.
    """
    x0 = start.copy().astype(float)
    xf = target.position.copy().astype(float)

    timestamps = np.arange(0.0, duration + dt, dt)
    observations = []
    prev_pos = x0.copy()

    for i, t in enumerate(timestamps):
        tau = min(t / duration, 1.0)
        pos = _minimum_jerk_pos(x0, xf, tau)

        if noise_std > 0.0:
            pos = pos + np.random.normal(0, noise_std, size=2)

        if i == 0:
            vel = np.zeros(2)
        else:
            vel = (pos - prev_pos) / dt

        observations.append(SimulatedObservation(
            position=pos.copy(),
            velocity=vel.copy(),
            timestamp=round(float(t), 6),
        ))
        prev_pos = pos.copy()

    return observations
