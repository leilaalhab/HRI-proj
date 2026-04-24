import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

RESULT_DIRS = [
    "results/figures",
    "results/screenshots",
    "results/animations",
    "results/logs",
]


def create_result_dirs():
    for d in RESULT_DIRS:
        os.makedirs(d, exist_ok=True)
    print("Result directories ready.")


def main():
    print("=== Real-Time Human Intent Recognition for Proactive Robot Handover ===")
    create_result_dirs()

    from src.scene.targets import get_targets
    from src.visualization.plots import plot_static_scene

    targets = get_targets()
    print(f"Loaded {len(targets)} targets: {[t.name for t in targets]}")
    for t in targets:
        print(f"  {t.label}: centre={t.position.tolist()}, region={t.region}")

    plot_static_scene(targets, "results/figures/static_scene.png")

    print("Run 'python experiments/run_simulation.py' for the simulation demo.")
    print("Run 'python experiments/run_webcam_demo.py' for the live webcam demo.")


if __name__ == "__main__":
    main()
