"""
Multi-trial evaluation runner — Stage 7.

Runs N simulation trials with noise, adapts robot arm timing across trials,
logs every trial to results/logs/trials.csv, and saves summary figures.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from src.scene.targets import get_targets, HAND_START
from src.perception.simulated_perception import generate_trajectory
from src.inference.bayesian_goal_inference import BayesianGoalInference
from src.prediction.minimum_jerk import estimate_duration
from src.evaluation.metrics import (
    DurationAdapter,
    TrialLogger,
    build_trial_result,
)
from src.visualization.plots import plot_prediction_error, plot_summary_metrics

# ---- Trial plan -------------------------------------------------------
# 9 trials: 3 per target, interleaved so adaptation sees variety.
TRIAL_PLAN = [
    "red", "blue", "green",
    "blue", "green", "red",
    "green", "red", "blue",
]
TRIAL_DURATION = 2.0   # seconds per simulated trajectory
NOISE_STD      = 1.5   # px — adds realistic variability; 3.0+ causes early false locks
                       #      because finite-diff velocity noise (≈42px/s) > early true speed


def _run_one_trial(
    targets: list,
    target_name: str,
    trial_id: int,
    adapter: DurationAdapter,
    noise_std: float,
) -> tuple:
    """
    Run one complete trial and return (TrialResult, D_estimated, D_actual).
    D_estimated and D_actual are returned so the caller can update the adapter.
    """
    target_map   = {t.name: t for t in targets}
    ground_truth = target_map[target_name]

    observations = generate_trajectory(
        target=ground_truth,
        duration=TRIAL_DURATION,
        dt=config.SIMULATION_DT,
        noise_std=noise_std,
    )

    inference = BayesianGoalInference(targets)
    lock_obs  = None
    for obs in observations:
        inference.update(obs)
        if inference.locked_target is not None and obs.timestamp == inference.lock_time:
            lock_obs = obs

    locked      = inference.locked_target
    xf_predicted = None
    D_estimated  = None
    D_actual     = None
    D_adapted    = None

    if locked is not None and lock_obs is not None:
        xf_predicted = locked.position.copy()
        speed        = float(np.linalg.norm(lock_obs.velocity))
        D_estimated  = estimate_duration(
            x0=HAND_START,
            x_current=lock_obs.position,
            xf=xf_predicted,
            current_speed=speed,
        )
        D_actual  = observations[-1].timestamp - inference.lock_time
        D_adapted = adapter.apply(D_estimated)

    result = build_trial_result(
        trial_id=trial_id,
        mode="simulation",
        ground_truth_target=target_name,
        predicted_target=locked.name if locked else None,
        lock_time=inference.lock_time,
        lock_confidence=inference.lock_confidence,
        xf_predicted=xf_predicted,
        xf_actual=observations[-1].position,
        D_estimated=D_estimated,
        D_actual=D_actual,
        D_adapted=D_adapted,
        num_frames=len(observations),
        notes=f"noise_std={noise_std}",
    )

    return result, D_estimated, D_actual


def main():
    np.random.seed(config.RANDOM_SEED)

    os.makedirs("results/figures", exist_ok=True)
    os.makedirs("results/logs", exist_ok=True)

    targets = get_targets()
    adapter = DurationAdapter(gain=config.ADAPTATION_GAIN)
    logger  = TrialLogger()

    print("=== Multi-Trial Evaluation ===")
    print(f"Trials    : {len(TRIAL_PLAN)}")
    print(f"noise_std : {NOISE_STD} px")
    print(f"gain      : {config.ADAPTATION_GAIN}")
    print()
    print(f"{'#':>3}  {'target':>6}  {'locked':>6}  {'ok':>3}  "
          f"{'lock_t':>7}  {'D_est':>6}  {'D_act':>6}  {'ratio':>6}  "
          f"{'err_px':>7}  corr_factor")
    print("-" * 90)

    all_results = []

    for i, target_name in enumerate(TRIAL_PLAN):
        trial_id    = i + 1
        corr_before = adapter.correction_factor

        result, D_est, D_act = _run_one_trial(
            targets, target_name, trial_id, adapter, NOISE_STD
        )
        logger.log(result)
        all_results.append(result)

        if D_est is not None and D_act is not None:
            adapter.adapt(D_estimated=D_est, D_actual=D_act)
        corr_after = adapter.correction_factor

        locked_str = result.predicted_target
        ok_str     = "ok" if result.target_correct else "X"
        lock_t_str = f"{result.lock_time:.2f}s" if result.lock_time == result.lock_time else "  --  "
        d_est_str  = f"{D_est:.3f}" if D_est is not None else "  --  "
        d_act_str  = f"{D_act:.3f}" if D_act is not None else "  --  "
        ratio_str  = f"{D_act/D_est:.3f}" if (D_est and D_act) else "  --  "
        err_str    = (f"{result.prediction_error_norm:.1f}px"
                      if result.prediction_error_norm == result.prediction_error_norm
                      else "  --  ")

        print(
            f"{trial_id:>3}  {target_name:>6}  {locked_str:>6}  {ok_str:>3}  "
            f"{lock_t_str:>7}  {d_est_str:>6}  {d_act_str:>6}  {ratio_str:>6}  "
            f"{err_str:>7}  {corr_before:.4f} -> {corr_after:.4f}"
        )

    print("-" * 90)

    n_correct  = sum(r.target_correct for r in all_results)
    n_total    = len(all_results)
    errors     = [r.prediction_error_norm for r in all_results
                  if r.prediction_error_norm == r.prediction_error_norm]
    lock_times = [r.lock_time for r in all_results
                  if r.lock_time == r.lock_time]

    print()
    print("=== Summary ===")
    print(f"  Target accuracy    : {n_correct}/{n_total}  ({100*n_correct/n_total:.0f}%)")
    print(f"  Mean pred error    : {np.mean(errors):.2f} px  (std={np.std(errors):.2f})")
    print(f"  Mean lock time     : {np.mean(lock_times):.3f}s")
    print(f"  Final D_correction : {adapter.correction_factor:.4f}")

    plot_prediction_error(
        results=all_results,
        save_path="results/figures/prediction_error.png",
    )
    plot_summary_metrics(
        results=all_results,
        save_path="results/figures/summary_metrics.png",
    )

    print("\nStage 7 complete.")
    print(f"  Log  : results/logs/trials.csv")
    print(f"  Figs : results/figures/prediction_error.png")
    print(f"         results/figures/summary_metrics.png")


if __name__ == "__main__":
    main()
