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
    print("Stage 1: Project scaffold initialised.")
    create_result_dirs()
    print("Run 'python experiments/run_simulation.py' for the simulation demo.")
    print("Run 'python experiments/run_webcam_demo.py' for the live webcam demo.")


if __name__ == "__main__":
    main()
