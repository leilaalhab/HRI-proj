import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

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
