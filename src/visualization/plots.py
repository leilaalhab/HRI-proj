import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

import config
from src.scene.targets import CANVAS_W, CANVAS_H, HAND_START


def plot_static_scene(targets: list, save_path: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))

    # Workspace background
    workspace = patches.Rectangle(
        (0, 0), CANVAS_W, CANVAS_H,
        linewidth=2, edgecolor="#555555", facecolor="#f0f0f0"
    )
    ax.add_patch(workspace)

    for target in targets:
        x_min, y_min, x_max, y_max = target.region
        w, h = x_max - x_min, y_max - y_min

        rect = patches.Rectangle(
            (x_min, y_min), w, h,
            linewidth=2,
            edgecolor=target.color_rgb,
            facecolor=(*target.color_rgb, 0.3),  # semi-transparent fill
        )
        ax.add_patch(rect)

        ax.text(
            target.position[0], y_min - 15,
            target.label,
            ha="center", va="top",
            fontsize=11, fontweight="bold",
            color=target.color_rgb,
        )

    # Hand start marker
    ax.plot(*HAND_START, "o", color="#333333", markersize=12, zorder=5)
    ax.text(
        HAND_START[0], HAND_START[1] + 18,
        "Hand start",
        ha="center", va="bottom",
        fontsize=10, color="#333333",
    )

    ax.set_xlim(0, CANVAS_W)
    ax.set_ylim(0, CANVAS_H)
    ax.invert_yaxis()   # pixel coords: y=0 at top
    ax.set_aspect("equal")
    ax.set_xlabel("X (pixels)")
    ax.set_ylabel("Y (pixels)")
    ax.set_title("HRI Scene — Candidate Target Objects", fontsize=13)
    ax.grid(True, linestyle="--", alpha=0.4)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.tight_layout()
    fig.savefig(save_path, dpi=120)
    plt.close(fig)
    print(f"Static scene saved → {save_path}")


def _draw_scene_base(ax, targets):
    """Shared helper: draw workspace + target boxes onto an existing axes."""
    workspace = patches.Rectangle(
        (0, 0), CANVAS_W, CANVAS_H,
        linewidth=2, edgecolor="#555555", facecolor="#f0f0f0"
    )
    ax.add_patch(workspace)

    for target in targets:
        x_min, y_min, x_max, y_max = target.region
        w, h = x_max - x_min, y_max - y_min
        rect = patches.Rectangle(
            (x_min, y_min), w, h,
            linewidth=2,
            edgecolor=target.color_rgb,
            facecolor=(*target.color_rgb, 0.3),
        )
        ax.add_patch(rect)
        ax.text(
            target.position[0], y_min - 15,
            target.label,
            ha="center", va="top",
            fontsize=11, fontweight="bold",
            color=target.color_rgb,
        )

    ax.plot(*HAND_START, "o", color="#333333", markersize=12, zorder=5)
    ax.text(
        HAND_START[0], HAND_START[1] + 18,
        "Hand start",
        ha="center", va="bottom",
        fontsize=10, color="#333333",
    )


def plot_scene_trajectory(targets: list, observations: list, ground_truth_target,
                          save_path: str) -> None:
    fig, (ax_scene, ax_vel) = plt.subplots(1, 2, figsize=(14, 6))

    # --- Left: spatial trajectory ---
    _draw_scene_base(ax_scene, targets)

    xs = [obs.position[0] for obs in observations]
    ys = [obs.position[1] for obs in observations]

    ax_scene.plot(xs, ys, "-", color=ground_truth_target.color_rgb,
                  linewidth=2, label="Observed trajectory", zorder=3)
    ax_scene.plot(xs[::2], ys[::2], ".", color=ground_truth_target.color_rgb,
                  markersize=5, alpha=0.6, zorder=4)
    ax_scene.plot(xs[-1], ys[-1], "x", color=ground_truth_target.color_rgb,
                  markersize=14, markeredgewidth=3,
                  label=f"Endpoint ({ground_truth_target.name})", zorder=6)

    ax_scene.set_xlim(0, CANVAS_W)
    ax_scene.set_ylim(0, CANVAS_H)
    ax_scene.invert_yaxis()
    ax_scene.set_aspect("equal")
    ax_scene.set_xlabel("X (pixels)")
    ax_scene.set_ylabel("Y (pixels)")
    ax_scene.set_title(f"Simulated Hand Trajectory → {ground_truth_target.label}",
                       fontsize=13)
    ax_scene.legend(loc="lower right", fontsize=9)
    ax_scene.grid(True, linestyle="--", alpha=0.4)

    # --- Right: velocity magnitude over time ---
    timestamps = [obs.timestamp for obs in observations]
    speeds = [float(np.linalg.norm(obs.velocity)) for obs in observations]

    ax_vel.plot(timestamps, speeds, ":", color=ground_truth_target.color_rgb,
                linewidth=2.5, label="Speed (px/s)")
    ax_vel.plot(timestamps, speeds, ".", color=ground_truth_target.color_rgb,
                markersize=6)

    # Mark peak speed
    peak_idx = int(np.argmax(speeds))
    ax_vel.annotate(
        f"Peak\n{speeds[peak_idx]:.1f} px/s",
        xy=(timestamps[peak_idx], speeds[peak_idx]),
        xytext=(timestamps[peak_idx] + 0.15, speeds[peak_idx] * 0.85),
        fontsize=9,
        arrowprops=dict(arrowstyle="->", color="#555555"),
        color="#555555",
    )

    ax_vel.set_xlabel("Time (s)")
    ax_vel.set_ylabel("Speed (pixels/s)")
    ax_vel.set_title("Velocity Profile — Minimum-Jerk\n(slow → fast → slow)", fontsize=13)
    ax_vel.legend(fontsize=9)
    ax_vel.grid(True, linestyle="--", alpha=0.4)
    ax_vel.set_xlim(left=0)
    ax_vel.set_ylim(bottom=0)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.tight_layout()
    fig.savefig(save_path, dpi=120)
    plt.close(fig)
    print(f"Trajectory figure saved → {save_path}")


def plot_posterior_probabilities(history: dict, targets: list, locked_target,
                                  lock_time: float, save_path: str) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))

    timestamps = history["timestamps"]

    for target in targets:
        ax.plot(
            timestamps, history[target.name],
            color=target.color_rgb, linewidth=2.5, label=target.label,
        )

    # Confidence threshold
    ax.axhline(
        y=config.CONFIDENCE_THRESHOLD,
        color="#333333", linestyle="--", linewidth=1.5,
        label=f"Lock threshold  (P = {config.CONFIDENCE_THRESHOLD})",
    )

    # Lock event
    if lock_time is not None and locked_target is not None:
        ax.axvline(x=lock_time, color="#666666", linestyle=":", linewidth=2)
        lock_prob = history[locked_target.name][
            history["timestamps"].index(lock_time)
        ]
        ax.annotate(
            f"TARGET LOCKED\n→ {locked_target.label}\n  t = {lock_time:.2f}s",
            xy=(lock_time, lock_prob),
            xytext=(lock_time + 0.12, lock_prob - 0.15),
            fontsize=9,
            color="#333333",
            arrowprops=dict(arrowstyle="->", color="#555555"),
        )

    ax.set_xlabel("Time (s)", fontsize=11)
    ax.set_ylabel("Posterior  P(G | observations)", fontsize=11)
    ax.set_title(
        "Bayesian Goal Inference — Posterior Probabilities Over Time", fontsize=13
    )
    ax.set_ylim(0, 1.05)
    ax.set_xlim(left=0)
    ax.legend(fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.4)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.tight_layout()
    fig.savefig(save_path, dpi=120)
    plt.close(fig)
    print(f"Posterior probabilities saved → {save_path}")
