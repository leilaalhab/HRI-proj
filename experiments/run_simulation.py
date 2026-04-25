"""Simulation demo — development/debug tool only. Not the final project output."""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from src.scene.targets import get_targets, HAND_START
from src.perception.simulated_perception import generate_trajectory
from src.inference.bayesian_goal_inference import BayesianGoalInference
from src.prediction.minimum_jerk import estimate_duration, minimum_jerk_trajectory
from src.visualization.plots import plot_scene_trajectory, plot_posterior_probabilities

# Change to "red", "blue", or "green" to test different targets.
GROUND_TRUTH = "blue"


def main():
    np.random.seed(config.RANDOM_SEED)
    os.makedirs("results/figures", exist_ok=True)

    targets = get_targets()
    target_map = {t.name: t for t in targets}

    if GROUND_TRUTH not in target_map:
        raise ValueError(f"Unknown target '{GROUND_TRUTH}'. Choose from {list(target_map)}")

    ground_truth_target = target_map[GROUND_TRUTH]

    # ------------------------------------------------------------------
    # Stage 3: generate trajectory
    # ------------------------------------------------------------------
    print("=== Simulation Demo ===")
    print(f"Ground truth target : {ground_truth_target.name} — {ground_truth_target.label}")

    observations = generate_trajectory(
        target=ground_truth_target,
        duration=2.0,
        dt=config.SIMULATION_DT,
        noise_std=0.0,
    )
    print(f"Observations        : {len(observations)} frames  "
          f"(duration={observations[-1].timestamp:.2f}s, dt={config.SIMULATION_DT}s)")

    # ------------------------------------------------------------------
    # Stage 4: Bayesian goal inference
    # ------------------------------------------------------------------
    print("\n=== Bayesian Goal Inference ===")
    print(f"{'Frame':>5}  {'Time':>6}  {'red':>7}  {'blue':>7}  {'green':>7}  {'Status'}")
    print("-" * 62)

    inference = BayesianGoalInference(targets)
    lock_obs = None

    for i, obs in enumerate(observations):
        posterior = inference.update(obs)

        is_lock_frame = (
            inference.locked_target is not None
            and obs.timestamp == inference.lock_time
        )
        if is_lock_frame:
            lock_obs = obs

        if i % 5 == 0 or is_lock_frame:
            status = f"<-- LOCKED ({inference.locked_target.name})" if is_lock_frame else ""
            print(
                f"{i:>5}  {obs.timestamp:>6.2f}s  "
                f"{posterior['red']:>7.3f}  {posterior['blue']:>7.3f}  "
                f"{posterior['green']:>7.3f}  {status}"
            )

    print("-" * 62)

    locked = inference.locked_target
    correct = locked is not None and locked.name == GROUND_TRUTH

    if locked:
        print(f"\nLocked target   : {locked.name}  "
              f"{'✓ correct' if correct else '✗ wrong (expected ' + GROUND_TRUTH + ')'}")
        print(f"Lock time       : {inference.lock_time:.2f}s  "
              f"(out of {observations[-1].timestamp:.2f}s total)")
        print(f"Lock confidence : {inference.lock_confidence:.3f}")
    else:
        print(f"\nNo target locked — max posterior never exceeded {config.CONFIDENCE_THRESHOLD}")

    plot_posterior_probabilities(
        history=inference.history,
        targets=targets,
        locked_target=inference.locked_target,
        lock_time=inference.lock_time,
        save_path="results/figures/posterior_probabilities.png",
    )

    # ------------------------------------------------------------------
    # Stage 5: minimum-jerk prediction
    # ------------------------------------------------------------------
    print("\n=== Minimum-Jerk Prediction ===")

    predicted_trajectory = None
    xf_predicted = None

    if locked is None or lock_obs is None:
        print("No target locked — skipping prediction.")
    else:
        xf_predicted = locked.position.copy()
        speed_at_lock = float(np.linalg.norm(lock_obs.velocity))

        D_remaining = estimate_duration(
            x0=HAND_START,
            x_current=lock_obs.position,
            xf=xf_predicted,
            current_speed=speed_at_lock,
        )
        D_actual_remaining = observations[-1].timestamp - inference.lock_time

        predicted_trajectory = minimum_jerk_trajectory(
            x0=lock_obs.position,
            xf=xf_predicted,
            D=D_remaining,
            dt=config.SIMULATION_DT,
        )

        print(f"xf_predicted       : {xf_predicted.tolist()}")
        print(f"xf_actual          : {[round(v, 2) for v in observations[-1].position.tolist()]}")
        print(f"Speed at lock      : {speed_at_lock:.1f} px/s")
        print(f"D_remaining est.   : {D_remaining:.3f}s")
        print(f"D_remaining actual : {D_actual_remaining:.3f}s")
        print(f"D estimation error : {abs(D_remaining - D_actual_remaining):.3f}s "
              f"({100 * abs(D_remaining - D_actual_remaining) / D_actual_remaining:.1f}%)")
        print(f"Predicted traj pts : {len(predicted_trajectory)}")

    plot_scene_trajectory(
        targets=targets,
        observations=observations,
        ground_truth_target=ground_truth_target,
        save_path="results/figures/scene_trajectory.png",
        predicted_trajectory=predicted_trajectory,
        xf_predicted=xf_predicted,
        lock_time=inference.lock_time,
    )


if __name__ == "__main__":
    main()
