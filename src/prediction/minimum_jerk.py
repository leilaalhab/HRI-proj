import numpy as np

# Speed below this (px/s) triggers the fallback D estimator.
_MIN_SPEED_PX = 5.0
# Fallback assumed speed when hand is nearly stationary at lock.
_FALLBACK_SPEED_PX = 60.0


def _s(tau: float) -> float:
    """Minimum-jerk position profile scalar. tau in [0, 1]."""
    return 10 * tau**3 - 15 * tau**4 + 6 * tau**5


def _g(tau: float) -> float:
    """Minimum-jerk normalized speed profile. tau in [0, 1]."""
    return 30 * tau**2 - 60 * tau**3 + 30 * tau**4


def _solve_tau(progress: float) -> float:
    """
    Solve s(tau) = progress for tau in [0, 1].

    s(tau) = 6tau^5 - 15tau^4 + 10tau^3 - progress = 0

    s is monotonically increasing on [0,1] so there is exactly one real
    root in that interval for any progress in [0, 1].
    """
    if progress <= 0.0:
        return 0.0
    if progress >= 1.0:
        return 1.0

    # Polynomial coefficients: highest degree first
    coeffs = [6.0, -15.0, 10.0, 0.0, 0.0, -progress]
    roots = np.roots(coeffs)

    # Keep only real roots in [0, 1]
    real_roots = [
        r.real for r in roots
        if abs(r.imag) < 1e-6 and -1e-6 <= r.real <= 1.0 + 1e-6
    ]
    if not real_roots:
        # Numerical fallback: linear approximation
        return float(np.clip(progress, 0.0, 1.0))

    tau = float(np.clip(min(real_roots, key=lambda r: abs(r - progress ** (1 / 3))), 0.0, 1.0))
    return tau


def minimum_jerk_trajectory(
    x0: np.ndarray,
    xf: np.ndarray,
    D: float,
    dt: float,
) -> list[np.ndarray]:
    """
    Generate a smooth minimum-jerk trajectory from x0 to xf.

    x(t) = x0 + (xf - x0) * s(tau),   tau = t / D

    Returns a list of 2-D position arrays, one per timestep.
    """
    if D <= 0:
        return [xf.copy()]

    timestamps = np.arange(0.0, D + dt, dt)
    positions = []
    for t in timestamps:
        tau = min(t / D, 1.0)
        pos = x0 + (xf - x0) * _s(tau)
        positions.append(pos.copy())
    return positions


def estimate_duration(
    x0: np.ndarray,
    x_current: np.ndarray,
    xf: np.ndarray,
    current_speed: float,
) -> float:
    """
    Estimate remaining movement duration using the minimum-jerk profile.

    Method
    ------
    1. Compute progress = |x_current - x0| / |xf - x0|
    2. Solve s(tau) = progress to find where we are in the profile.
    3. D_total = total_dist * g(tau) / current_speed
    4. D_remaining = D_total * (1 - tau)

    Falls back to remaining_dist / _FALLBACK_SPEED_PX when speed is too
    low for a reliable estimate (e.g. hand barely moving at lock time).
    """
    total_dist = float(np.linalg.norm(xf - x0))
    remaining_dist = float(np.linalg.norm(xf - x_current))

    if total_dist < 1e-6:
        return 0.1

    traveled_dist = total_dist - remaining_dist
    progress = float(np.clip(traveled_dist / total_dist, 0.0, 1.0))

    if current_speed < _MIN_SPEED_PX:
        # Hand barely moving — use distance / fallback speed
        D_remaining = remaining_dist / _FALLBACK_SPEED_PX
        return float(max(D_remaining, 0.1))

    tau = _solve_tau(progress)
    g_tau = _g(tau)

    if g_tau < 1e-4:
        # Near start or end of profile — g is ~0, formula breaks down
        D_remaining = remaining_dist / _FALLBACK_SPEED_PX
        return float(max(D_remaining, 0.1))

    D_total = total_dist * g_tau / current_speed
    D_remaining = D_total * (1.0 - tau)
    return float(max(D_remaining, 0.1))
