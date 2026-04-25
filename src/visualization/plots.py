import os
import numpy as np
import cv2
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
                          save_path: str,
                          predicted_trajectory: list = None,
                          xf_predicted: np.ndarray = None,
                          lock_time: float = None) -> None:
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
                  label=f"Actual endpoint ({ground_truth_target.name})", zorder=6)

    # Predicted trajectory (dashed, drawn from lock point onward)
    if predicted_trajectory is not None and len(predicted_trajectory) > 1:
        pxs = [p[0] for p in predicted_trajectory]
        pys = [p[1] for p in predicted_trajectory]
        ax_scene.plot(pxs, pys, "--", color="#888888",
                      linewidth=2, label="Predicted trajectory", zorder=3)

    # Predicted endpoint star
    if xf_predicted is not None:
        ax_scene.plot(xf_predicted[0], xf_predicted[1], "*",
                      color="#333333", markersize=18,
                      label="xf_predicted", zorder=7)

    # Lock point marker
    if lock_time is not None:
        lock_obs = next((o for o in observations if o.timestamp >= lock_time), None)
        if lock_obs is not None:
            ax_scene.plot(lock_obs.position[0], lock_obs.position[1],
                          "D", color="#FFaa00", markersize=10,
                          label=f"Lock point  (t={lock_time:.2f}s)", zorder=8)

    ax_scene.set_xlim(0, CANVAS_W)
    ax_scene.set_ylim(0, CANVAS_H)
    ax_scene.invert_yaxis()
    ax_scene.set_aspect("equal")
    ax_scene.set_xlabel("X (pixels)")
    ax_scene.set_ylabel("Y (pixels)")
    ax_scene.set_title(f"Hand Trajectory + Prediction → {ground_truth_target.label}",
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

    if lock_time is not None:
        ax_vel.axvline(x=lock_time, color="#FFaa00", linestyle="--",
                       linewidth=1.5, label=f"Lock  (t={lock_time:.2f}s)")

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


def plot_interception_prediction(
    targets: list,
    observations: list,
    ground_truth_target,
    predicted_trajectory: list,
    xf_predicted: np.ndarray,
    lock_time: float,
    robot_frame: np.ndarray,
    locked_target,
    save_path: str,
) -> None:
    """
    Two-panel figure:
      Left  — scene with observed trajectory, predicted trajectory, lock point, xf_predicted
      Right — PyBullet robot arm render at its IK-computed pose
    """
    fig, (ax_scene, ax_robot) = plt.subplots(1, 2, figsize=(14, 6))

    # --- Left: scene ---
    _draw_scene_base(ax_scene, targets)

    xs = [o.position[0] for o in observations]
    ys = [o.position[1] for o in observations]
    ax_scene.plot(xs, ys, "-", color=ground_truth_target.color_rgb,
                  linewidth=2, label="Observed trajectory", zorder=3)
    ax_scene.plot(xs[::2], ys[::2], ".", color=ground_truth_target.color_rgb,
                  markersize=5, alpha=0.6, zorder=4)
    ax_scene.plot(xs[-1], ys[-1], "x", color=ground_truth_target.color_rgb,
                  markersize=14, markeredgewidth=3,
                  label=f"Actual endpoint ({ground_truth_target.name})", zorder=6)

    if predicted_trajectory:
        pxs = [p[0] for p in predicted_trajectory]
        pys = [p[1] for p in predicted_trajectory]
        ax_scene.plot(pxs, pys, "--", color="#888888",
                      linewidth=2, label="Predicted trajectory", zorder=3)

    if xf_predicted is not None:
        ax_scene.plot(xf_predicted[0], xf_predicted[1], "*",
                      color="#111111", markersize=18,
                      label="xf_predicted", zorder=7)

    if lock_time is not None:
        lock_obs = next((o for o in observations if o.timestamp >= lock_time), None)
        if lock_obs is not None:
            ax_scene.plot(lock_obs.position[0], lock_obs.position[1],
                          "D", color="#FFaa00", markersize=10,
                          label=f"Lock  (t={lock_time:.2f}s)", zorder=8)

    ax_scene.set_xlim(0, CANVAS_W)
    ax_scene.set_ylim(0, CANVAS_H)
    ax_scene.invert_yaxis()
    ax_scene.set_aspect("equal")
    ax_scene.set_xlabel("X (pixels)")
    ax_scene.set_ylabel("Y (pixels)")
    ax_scene.set_title(f"Scene + Prediction → {ground_truth_target.label}", fontsize=13)
    ax_scene.legend(loc="lower right", fontsize=9)
    ax_scene.grid(True, linestyle="--", alpha=0.4)

    # --- Right: PyBullet render ---
    ax_robot.imshow(robot_frame)
    ax_robot.axis("off")
    locked_name = locked_target.label if locked_target else "none"
    ax_robot.set_title(f"PyBullet Robot Arm — IK target: {locked_name}", fontsize=13)
    if locked_target is not None:
        ax_robot.text(
            0.5, 0.03,
            f"End effector positioned at → {locked_target.label}",
            transform=ax_robot.transAxes, ha="center", fontsize=10,
            color="white",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#222", alpha=0.8),
        )

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.tight_layout()
    fig.savefig(save_path, dpi=120)
    plt.close(fig)
    print(f"Interception prediction saved → {save_path}")


def render_scene_frame(
    targets: list,
    observations: list,
    ground_truth_target,
    predicted_trajectory: list = None,
    xf_predicted: np.ndarray = None,
    lock_time: float = None,
    locked_target=None,
    status_lines: list = None,
) -> np.ndarray:
    """
    Render the simulation scene onto a CANVAS_H × CANVAS_W OpenCV numpy array (RGB).
    Used to build the browser composite frame.
    """
    canvas = np.full((CANVAS_H, CANVAS_W, 3), 30, dtype=np.uint8)

    # Target rectangles
    for t in targets:
        x_min, y_min, x_max, y_max = t.region
        color = (
            int(t.color_rgb[2] * 255),
            int(t.color_rgb[1] * 255),
            int(t.color_rgb[0] * 255),
        )
        overlay = canvas.copy()
        cv2.rectangle(overlay, (x_min, y_min), (x_max, y_max), color, -1)
        cv2.addWeighted(overlay, 0.25, canvas, 0.75, 0, canvas)
        cv2.rectangle(canvas, (x_min, y_min), (x_max, y_max), color, 2)
        cv2.putText(canvas, t.label,
                    (int(t.position[0]) - 30, y_min - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

    # Hand start marker
    hs = (int(HAND_START[0]), int(HAND_START[1]))
    cv2.circle(canvas, hs, 8, (80, 80, 80), -1)
    cv2.putText(canvas, "start", (hs[0] - 16, hs[1] - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (120, 120, 120), 1)

    # Observed trajectory
    gt_color = (
        int(ground_truth_target.color_rgb[2] * 255),
        int(ground_truth_target.color_rgb[1] * 255),
        int(ground_truth_target.color_rgb[0] * 255),
    )
    pts = [(int(o.position[0]), int(o.position[1])) for o in observations]
    for i in range(1, len(pts)):
        cv2.line(canvas, pts[i - 1], pts[i], gt_color, 2, cv2.LINE_AA)

    # Lock point marker (orange diamond)
    if lock_time is not None:
        lock_obs = next((o for o in observations if o.timestamp >= lock_time), None)
        if lock_obs is not None:
            lp = (int(lock_obs.position[0]), int(lock_obs.position[1]))
            size = 8
            diamond = np.array([
                [lp[0], lp[1] - size],
                [lp[0] + size, lp[1]],
                [lp[0], lp[1] + size],
                [lp[0] - size, lp[1]],
            ])
            cv2.fillPoly(canvas, [diamond], (0, 170, 255))
            cv2.putText(canvas, f"lock t={lock_time:.2f}s",
                        (lp[0] + 10, lp[1]),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 170, 255), 1)

    # Predicted trajectory (dashed)
    if predicted_trajectory and len(predicted_trajectory) > 1:
        pp = [(int(p[0]), int(p[1])) for p in predicted_trajectory]
        for i in range(0, len(pp) - 1, 2):
            cv2.line(canvas, pp[i], pp[min(i + 1, len(pp) - 1)],
                     (160, 160, 160), 1, cv2.LINE_AA)

    # xf_predicted star (drawn as two overlapping triangles)
    if xf_predicted is not None:
        sp = (int(xf_predicted[0]), int(xf_predicted[1]))
        for angle in range(0, 360, 60):
            rad = np.radians(angle)
            outer = (int(sp[0] + 12 * np.cos(rad)), int(sp[1] + 12 * np.sin(rad)))
            rad2 = np.radians(angle + 30)
            inner = (int(sp[0] + 5 * np.cos(rad2)), int(sp[1] + 5 * np.sin(rad2)))
            cv2.line(canvas, inner, outer, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(canvas, "xf_pred",
                    (sp[0] + 10, sp[1] - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200, 200, 200), 1)

    # Status text (top-left)
    lines = status_lines or []
    for i, line in enumerate(lines):
        cv2.putText(canvas, line, (8, 20 + i * 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)

    return canvas


# ---------------------------------------------------------------------------
# Stage 7 summary plots
# ---------------------------------------------------------------------------

def plot_prediction_error(results: list, save_path: str) -> None:
    """
    Bar chart of prediction error norm per trial, coloured by target.
    Twin Y-axis shows D_correction_factor evolution.

    Parameters
    ----------
    results : list of TrialResult dataclasses
    """
    import math

    _TARGET_COLORS = {"red": "#CC3333", "blue": "#3355CC", "green": "#22AA44"}

    trial_ids   = [r.trial_id for r in results]
    errors      = [r.prediction_error_norm if not math.isnan(r.prediction_error_norm) else 0.0
                   for r in results]
    colors      = [_TARGET_COLORS.get(r.ground_truth_target, "#888888") for r in results]
    corrections = [r.D_adapted / r.D_estimated
                   if not math.isnan(r.D_adapted) and not math.isnan(r.D_estimated)
                   and r.D_estimated > 0 else float("nan")
                   for r in results]

    fig, ax1 = plt.subplots(figsize=(10, 5))

    ax1.bar(trial_ids, errors, color=colors, alpha=0.8, zorder=2)
    ax1.set_xlabel("Trial", fontsize=11)
    ax1.set_ylabel("Prediction Error  ‖xf_pred − xf_actual‖  (px)", fontsize=11)
    ax1.set_xticks(trial_ids)
    ax1.grid(True, linestyle="--", alpha=0.4, axis="y")

    ax2 = ax1.twinx()
    valid = [(tid, c) for tid, c in zip(trial_ids, corrections) if not math.isnan(c)]
    if valid:
        vt, vc = zip(*valid)
        ax2.plot(vt, vc, "o--", color="#FF8800", linewidth=2, markersize=7,
                 label="D_correction_factor")
        ax2.set_ylabel("D_correction_factor  (orange)", fontsize=11, color="#FF8800")
        ax2.tick_params(axis="y", labelcolor="#FF8800")
        ax2.axhline(y=1.0, color="#FF8800", linestyle=":", alpha=0.4)

    # Legend patches for target colours
    from matplotlib.patches import Patch
    legend_els = [Patch(color=c, label=name.capitalize())
                  for name, c in _TARGET_COLORS.items()]
    if valid:
        from matplotlib.lines import Line2D
        legend_els.append(Line2D([0], [0], color="#FF8800", marker="o",
                                 linestyle="--", label="D_correction_factor"))
    ax1.legend(handles=legend_els, loc="upper left", fontsize=9)

    ax1.set_title("Prediction Error per Trial  +  Duration Adaptation", fontsize=13)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.tight_layout()
    fig.savefig(save_path, dpi=120)
    plt.close(fig)
    print(f"Prediction error figure saved → {save_path}")


def plot_summary_metrics(results: list, save_path: str) -> None:
    """
    Three-panel summary figure:
      Left   — target prediction accuracy (bar per target)
      Centre — prediction error distribution (histogram)
      Right  — D_correction_factor trajectory across trials
    """
    import math

    _TARGET_COLORS = {"red": "#CC3333", "blue": "#3355CC", "green": "#22AA44"}

    fig, (ax_acc, ax_err, ax_corr) = plt.subplots(1, 3, figsize=(15, 5))

    # --- Accuracy ---
    targets_seen = sorted({r.ground_truth_target for r in results})
    acc_data = {t: {"total": 0, "correct": 0} for t in targets_seen}
    for r in results:
        acc_data[r.ground_truth_target]["total"] += 1
        if r.target_correct:
            acc_data[r.ground_truth_target]["correct"] += 1

    names = list(acc_data)
    totals  = [acc_data[t]["total"]   for t in names]
    correct = [acc_data[t]["correct"] for t in names]
    wrong   = [acc_data[t]["total"] - acc_data[t]["correct"] for t in names]
    cols    = [_TARGET_COLORS.get(t, "#888888") for t in names]
    x = range(len(names))

    ax_acc.bar(x, correct, color=cols, alpha=0.85, label="Correct")
    ax_acc.bar(x, wrong, bottom=correct, color="gray", alpha=0.5, label="Wrong")
    ax_acc.set_xticks(list(x))
    ax_acc.set_xticklabels([n.capitalize() for n in names])
    ax_acc.set_ylabel("Trial count")
    ax_acc.set_title("Target Accuracy per Class", fontsize=12)
    ax_acc.legend(fontsize=9)
    ax_acc.grid(True, axis="y", linestyle="--", alpha=0.4)
    total_correct = sum(correct)
    total_all = len(results)
    ax_acc.set_title(
        f"Target Accuracy  ({total_correct}/{total_all} = "
        f"{100*total_correct/total_all:.0f}%)",
        fontsize=12,
    )

    # --- Error histogram ---
    errors = [r.prediction_error_norm for r in results if not math.isnan(r.prediction_error_norm)]
    if errors:
        ax_err.hist(errors, bins=max(5, len(errors) // 2), color="#4488CC",
                    edgecolor="white", alpha=0.85)
        ax_err.axvline(x=float(np.mean(errors)), color="#FF4444", linestyle="--",
                       linewidth=2, label=f"Mean = {np.mean(errors):.1f} px")
        ax_err.legend(fontsize=9)
    ax_err.set_xlabel("Prediction error (px)")
    ax_err.set_ylabel("Count")
    ax_err.set_title("Prediction Error Distribution", fontsize=12)
    ax_err.grid(True, linestyle="--", alpha=0.4)

    # --- D_correction trajectory ---
    trial_ids = [r.trial_id for r in results]
    corrections = []
    for r in results:
        if not math.isnan(r.D_adapted) and not math.isnan(r.D_estimated) and r.D_estimated > 0:
            corrections.append(r.D_adapted / r.D_estimated)
        else:
            corrections.append(float("nan"))

    valid_pairs = [(t, c) for t, c in zip(trial_ids, corrections) if not math.isnan(c)]
    if valid_pairs:
        vt, vc = zip(*valid_pairs)
        ax_corr.plot(vt, vc, "o-", color="#FF8800", linewidth=2.5, markersize=8)
        ax_corr.axhline(y=1.0, color="#888888", linestyle=":", linewidth=1.5,
                        label="No correction (1.0)")
        ax_corr.set_xlabel("Trial")
        ax_corr.set_ylabel("D_correction_factor")
        ax_corr.set_title("Duration Adaptation Across Trials", fontsize=12)
        ax_corr.legend(fontsize=9)
        ax_corr.grid(True, linestyle="--", alpha=0.4)
        ax_corr.set_xticks(list(vt))

    fig.suptitle(
        f"Stage 7 Summary — {total_all} trials  |  "
        f"accuracy {100*total_correct/total_all:.0f}%  |  "
        f"mean error {np.mean(errors):.1f} px" if errors else "Stage 7 Summary",
        fontsize=14,
        fontweight="bold",
    )

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.tight_layout()
    fig.savefig(save_path, dpi=120)
    plt.close(fig)
    print(f"Summary metrics figure saved → {save_path}")
