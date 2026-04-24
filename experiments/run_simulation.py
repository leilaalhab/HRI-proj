"""Simulation demo — development/debug tool only. Not the final project output."""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from src.scene.targets import get_targets
from src.perception.simulated_perception import generate_trajectory
from src.visualization.plots import plot_scene_trajectory

# Change this to "red", "blue", or "green" to test different targets.
GROUND_TRUTH = "green"


def main():
    np.random.seed(config.RANDOM_SEED)

    targets = get_targets()
    target_map = {t.name: t for t in targets}

    if GROUND_TRUTH not in target_map:
        raise ValueError(f"Unknown target '{GROUND_TRUTH}'. Choose from {list(target_map)}")

    ground_truth_target = target_map[GROUND_TRUTH]

    print("=== Simulation Demo (Stage 3) ===")
    print(f"Ground truth target : {ground_truth_target.name} — {ground_truth_target.label}")
    print(f"Target centre       : {ground_truth_target.position.tolist()}")

    observations = generate_trajectory(
        target=ground_truth_target,
        duration=2.0,
        dt=config.SIMULATION_DT,
        noise_std=0.0,
    )

    print(f"Observations        : {len(observations)} frames "
          f"(duration={observations[-1].timestamp:.2f}s, dt={config.SIMULATION_DT}s)")
    print(f"Start position      : {observations[0].position.tolist()}")
    print(f"Start velocity      : {observations[0].velocity.tolist()}")
    print(f"End position        : {[round(v, 2) for v in observations[-1].position.tolist()]}")
    print(f"End velocity        : {[round(v, 2) for v in observations[-1].velocity.tolist()]}")

    os.makedirs("results/figures", exist_ok=True)
    plot_scene_trajectory(
        targets=targets,
        observations=observations,
        ground_truth_target=ground_truth_target,
        save_path="results/figures/scene_trajectory.png",
    )


if __name__ == "__main__":
    main()
