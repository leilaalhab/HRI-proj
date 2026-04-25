"""Tests for src/evaluation/metrics.py"""
import math
import os
import sys
import tempfile

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.evaluation.metrics import (
    DurationAdapter,
    TrialLogger,
    build_trial_result,
    compute_prediction_error,
    compute_target_correct,
)


# ---------------------------------------------------------------------------
# compute_prediction_error
# ---------------------------------------------------------------------------

def test_prediction_error_zero_when_equal():
    pt = np.array([320.0, 180.0])
    vec, norm = compute_prediction_error(pt, pt)
    assert norm == 0.0
    assert np.allclose(vec, [0.0, 0.0])


def test_prediction_error_nonzero():
    pred = np.array([100.0, 200.0])
    actual = np.array([110.0, 200.0])
    vec, norm = compute_prediction_error(pred, actual)
    assert abs(norm - 10.0) < 1e-9
    assert abs(vec[0] - (-10.0)) < 1e-9


def test_prediction_error_2d_norm():
    pred   = np.array([0.0, 0.0])
    actual = np.array([3.0, 4.0])
    _, norm = compute_prediction_error(pred, actual)
    assert abs(norm - 5.0) < 1e-9


def test_prediction_error_none_returns_zero():
    actual = np.array([320.0, 180.0])
    vec, norm = compute_prediction_error(None, actual)
    assert norm == 0.0
    assert np.allclose(vec, [0.0, 0.0])


# ---------------------------------------------------------------------------
# compute_target_correct
# ---------------------------------------------------------------------------

def test_target_correct_match():
    assert compute_target_correct("blue", "blue") is True


def test_target_correct_mismatch():
    assert compute_target_correct("red", "blue") is False


def test_target_correct_none():
    assert compute_target_correct(None, "blue") is False


# ---------------------------------------------------------------------------
# DurationAdapter
# ---------------------------------------------------------------------------

def test_adapter_starts_at_one():
    adapter = DurationAdapter(gain=0.1)
    assert adapter.correction_factor == 1.0


def test_adapter_apply_identity():
    adapter = DurationAdapter(gain=0.1)
    assert abs(adapter.apply(2.0) - 2.0) < 1e-9


def test_adapter_moves_toward_ratio():
    adapter = DurationAdapter(gain=0.1)
    # ratio = 1.5 / 2.0 = 0.75  →  correction = 1.0 + 0.1*(0.75-1.0) = 0.975
    adapter.adapt(D_estimated=2.0, D_actual=1.5)
    assert abs(adapter.correction_factor - 0.975) < 1e-9


def test_adapter_converges():
    adapter = DurationAdapter(gain=0.1)
    for _ in range(200):
        adapter.adapt(D_estimated=2.0, D_actual=1.6)
    # Should converge to 1.6/2.0 = 0.8
    assert abs(adapter.correction_factor - 0.8) < 0.01


def test_adapter_changes_across_trials():
    adapter = DurationAdapter(gain=0.1)
    prev = adapter.correction_factor
    adapter.adapt(D_estimated=2.0, D_actual=1.5)
    assert adapter.correction_factor != prev


def test_adapter_nan_guard():
    adapter = DurationAdapter(gain=0.1)
    adapter.adapt(D_estimated=0.0, D_actual=1.5)
    assert adapter.correction_factor == 1.0   # unchanged — bad D_estimated ignored


# ---------------------------------------------------------------------------
# TrialLogger
# ---------------------------------------------------------------------------

def _make_result(trial_id=1):
    return build_trial_result(
        trial_id=trial_id,
        mode="simulation",
        ground_truth_target="blue",
        predicted_target="blue",
        lock_time=0.35,
        lock_confidence=0.87,
        xf_predicted=np.array([320.0, 180.0]),
        xf_actual=np.array([321.0, 181.0]),
        D_estimated=1.85,
        D_actual=1.65,
        D_adapted=1.80,
        num_frames=41,
        notes="test",
    )


def test_logger_creates_file():
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "logs", "trials.csv")
        logger = TrialLogger(log_path=path)
        assert os.path.exists(path)


def test_logger_writes_header():
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "logs", "trials.csv")
        TrialLogger(log_path=path)
        with open(path) as fh:
            header = fh.readline().strip()
        assert "trial_id" in header
        assert "prediction_error_norm" in header
        assert "D_adapted" in header


def test_logger_round_trip():
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "logs", "trials.csv")
        logger = TrialLogger(log_path=path)
        r = _make_result(trial_id=42)
        logger.log(r)
        rows = logger.load_all()
        assert len(rows) == 1
        assert rows[0]["trial_id"] == "42"
        assert rows[0]["ground_truth_target"] == "blue"
        assert rows[0]["target_correct"] == "True"


def test_logger_appends_multiple():
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "logs", "trials.csv")
        logger = TrialLogger(log_path=path)
        logger.log(_make_result(1))
        logger.log(_make_result(2))
        logger.log(_make_result(3))
        assert len(logger.load_all()) == 3


# ---------------------------------------------------------------------------
# build_trial_result
# ---------------------------------------------------------------------------

def test_build_no_lock():
    result = build_trial_result(
        trial_id=1,
        mode="simulation",
        ground_truth_target="red",
        predicted_target=None,
        lock_time=None,
        lock_confidence=None,
        xf_predicted=None,
        xf_actual=np.array([120.0, 180.0]),
        D_estimated=None,
        D_actual=None,
        D_adapted=None,
        num_frames=20,
    )
    assert result.predicted_target == "none"
    assert result.target_correct is False
    assert math.isnan(result.lock_time)
    assert math.isnan(result.prediction_error_norm)
    assert math.isnan(result.D_estimated)


def test_build_correct_lock_zero_error():
    result = build_trial_result(
        trial_id=1,
        mode="simulation",
        ground_truth_target="green",
        predicted_target="green",
        lock_time=0.4,
        lock_confidence=0.9,
        xf_predicted=np.array([520.0, 180.0]),
        xf_actual=np.array([520.0, 180.0]),
        D_estimated=1.6,
        D_actual=1.6,
        D_adapted=1.6,
        num_frames=41,
    )
    assert result.target_correct is True
    assert result.prediction_error_norm == 0.0


def test_build_wrong_lock():
    result = build_trial_result(
        trial_id=1,
        mode="simulation",
        ground_truth_target="red",
        predicted_target="blue",
        lock_time=0.5,
        lock_confidence=0.82,
        xf_predicted=np.array([320.0, 180.0]),
        xf_actual=np.array([120.0, 180.0]),
        D_estimated=1.5,
        D_actual=1.5,
        D_adapted=1.5,
        num_frames=41,
    )
    assert result.target_correct is False
    assert result.prediction_error_norm > 0.0
