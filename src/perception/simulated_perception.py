import numpy as np
from dataclasses import dataclass

from src.scene.targets import Target, HAND_START
from src.prediction.minimum_jerk import minimum_jerk_trajectory


@dataclass
class SimulatedObservation:
    position: np.ndarray   # [x, y] in pixel coords
    velocity: np.ndarray   # [vx, vy] pixels/second
    timestamp: float       # seconds from trial start


def generate_trajectory(
    target: Target,
    start: np.ndarray = HAND_START,
    duration: float = 2.0,
    dt: float = 0.05,
    noise_std: float = 0.0,
) -> list[SimulatedObservation]:
    """
    Generate a smooth simulated hand trajectory from start toward target
    using the minimum-jerk profile.  noise_std=0 keeps it clean for testing.
    """
    x0 = start.copy().astype(float)
    xf = target.position.copy().astype(float)

    positions = minimum_jerk_trajectory(x0, xf, duration, dt)
    timestamps = np.arange(0.0, duration + dt, dt)[: len(positions)]

    observations = []
    for i, (pos, t) in enumerate(zip(positions, timestamps)):
        if noise_std > 0.0:
            pos = pos + np.random.normal(0, noise_std, size=2)

        vel = np.zeros(2) if i == 0 else (pos - positions[i - 1]) / dt

        observations.append(SimulatedObservation(
            position=pos.copy(),
            velocity=vel.copy(),
            timestamp=round(float(t), 6),
        ))

    return observations
