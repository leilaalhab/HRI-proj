"""
Evaluation metrics, trial logging, and duration adaptation for Stage 7.

Three responsibilities:
  - Pure computation: prediction error, target correctness
  - Stateful adaptation: DurationAdapter tracks correction factor across trials
  - Logging: TrialLogger appends rows to a CSV
"""
import csv
import math
import os
from dataclasses import asdict, dataclass, fields
from datetime import datetime
from typing import Optional

import numpy as np

import config


# ---------------------------------------------------------------------------
# Data record
# ---------------------------------------------------------------------------

@dataclass
class TrialResult:
    trial_id: int
    mode: str                     # "simulation" | "webcam"
    started_at: str               # ISO-8601 timestamp
    ground_truth_target: str
    predicted_target: str         # "none" if no lock
    target_correct: bool
    lock_time: float              # seconds; NaN if no lock
    lock_confidence: float        # NaN if no lock
    xf_predicted_x: float         # pixel coords; NaN if no lock
    xf_predicted_y: float
    xf_actual_x: float
    xf_actual_y: float
    prediction_error_x: float     # xf_predicted − xf_actual; NaN if no lock
    prediction_error_y: float
    prediction_error_norm: float  # Euclidean norm; NaN if no lock
    D_estimated: float            # seconds; NaN if no lock
    D_actual: float               # seconds from lock to hand stop; NaN if no lock
    D_adapted: float              # D_estimated × correction_factor; NaN if no lock
    num_frames: int
    notes: str


# ---------------------------------------------------------------------------
# Pure computation
# ---------------------------------------------------------------------------

def compute_prediction_error(
    xf_predicted: Optional[np.ndarray],
    xf_actual: np.ndarray,
) -> tuple[np.ndarray, float]:
    """
    Return (error_vector, error_norm).
    error_vector = xf_predicted − xf_actual.
    Returns (zeros, 0.0) when xf_predicted is None.
    """
    if xf_predicted is None:
        return np.zeros(2), 0.0
    vec = np.asarray(xf_predicted, dtype=float) - np.asarray(xf_actual, dtype=float)
    return vec, float(np.linalg.norm(vec))


def compute_target_correct(
    predicted_target: Optional[str],
    ground_truth_target: str,
) -> bool:
    if predicted_target is None:
        return False
    return predicted_target == ground_truth_target


# ---------------------------------------------------------------------------
# Duration adaptation
# ---------------------------------------------------------------------------

class DurationAdapter:
    """
    Tracks a correction factor for the robot arm move duration across trials.

    At lock time we estimate D_remaining from the minimum-jerk tau
    back-calculation. That estimate has a consistent bias (over- or
    under-estimates due to velocity discretisation). After each trial ends
    we know the true remaining duration (D_actual) and can correct.

    Update rule (exponential moving average):
        correction_factor += gain × (D_actual/D_estimated − correction_factor)

    Applied next trial:
        D_robot = D_estimated × correction_factor × ROBOT_D_SCALE

    With gain = 0.1 the factor converges slowly, averaging over noise, so
    a single outlier trial does not destabilise the estimate.
    """

    def __init__(self, gain: float = config.ADAPTATION_GAIN):
        self.correction_factor: float = 1.0
        self._gain = gain

    def adapt(self, D_estimated: float, D_actual: float) -> None:
        """Update after a completed trial. No-op if either value is invalid."""
        if math.isnan(D_estimated) or math.isnan(D_actual) or D_estimated < 1e-6:
            return
        ratio = D_actual / D_estimated
        self.correction_factor += self._gain * (ratio - self.correction_factor)

    def apply(self, D_estimated: float) -> float:
        """Return corrected duration for use in the next trial."""
        if math.isnan(D_estimated):
            return float("nan")
        return D_estimated * self.correction_factor


# ---------------------------------------------------------------------------
# Trial logging
# ---------------------------------------------------------------------------

class TrialLogger:
    """Appends TrialResult rows to a CSV; creates the file with header on first use."""

    DEFAULT_PATH = "results/logs/trials.csv"

    def __init__(self, log_path: str = DEFAULT_PATH):
        self.log_path = log_path
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        self._fieldnames = [f.name for f in fields(TrialResult)]
        if not os.path.exists(log_path):
            with open(log_path, "w", newline="") as fh:
                csv.DictWriter(fh, fieldnames=self._fieldnames).writeheader()

    def log(self, result: TrialResult) -> None:
        with open(self.log_path, "a", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=self._fieldnames)
            writer.writerow(asdict(result))

    def load_all(self) -> list[dict]:
        """Return all rows as a list of dicts (string values — cast as needed)."""
        if not os.path.exists(self.log_path):
            return []
        with open(self.log_path, newline="") as fh:
            return list(csv.DictReader(fh))


# ---------------------------------------------------------------------------
# Helper: build a TrialResult from pipeline outputs
# ---------------------------------------------------------------------------

def build_trial_result(
    trial_id: int,
    mode: str,
    ground_truth_target: str,
    predicted_target: Optional[str],
    lock_time: Optional[float],
    lock_confidence: Optional[float],
    xf_predicted: Optional[np.ndarray],
    xf_actual: np.ndarray,
    D_estimated: Optional[float],
    D_actual: Optional[float],
    D_adapted: Optional[float],
    num_frames: int,
    notes: str = "",
) -> TrialResult:
    """
    Assemble a TrialResult from the pipeline's raw outputs.
    Fills NaN for every field that is undefined when there was no target lock.
    """
    nan = float("nan")
    target_correct = compute_target_correct(predicted_target, ground_truth_target)
    err_vec, err_norm = compute_prediction_error(xf_predicted, xf_actual)

    return TrialResult(
        trial_id=trial_id,
        mode=mode,
        started_at=datetime.now().isoformat(timespec="seconds"),
        ground_truth_target=ground_truth_target,
        predicted_target=predicted_target or "none",
        target_correct=target_correct,
        lock_time=lock_time if lock_time is not None else nan,
        lock_confidence=lock_confidence if lock_confidence is not None else nan,
        xf_predicted_x=float(xf_predicted[0]) if xf_predicted is not None else nan,
        xf_predicted_y=float(xf_predicted[1]) if xf_predicted is not None else nan,
        xf_actual_x=float(xf_actual[0]),
        xf_actual_y=float(xf_actual[1]),
        prediction_error_x=float(err_vec[0]) if xf_predicted is not None else nan,
        prediction_error_y=float(err_vec[1]) if xf_predicted is not None else nan,
        prediction_error_norm=err_norm if xf_predicted is not None else nan,
        D_estimated=D_estimated if D_estimated is not None else nan,
        D_actual=D_actual if D_actual is not None else nan,
        D_adapted=D_adapted if D_adapted is not None else nan,
        num_frames=num_frames,
        notes=notes,
    )
