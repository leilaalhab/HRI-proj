"""
Live webcam HRI demo — Stage 8 / 9.

Runs a guided 9-trial plan (3 per target, interleaved). The overlay
shows which target the user should reach next; ground truth is therefore
known automatically and accuracy can be computed.

Four threads
------------
  webcam     : MediaPipe hand tracking → shared state
  inference  : Bayesian update + minimum-jerk prediction → shared state
  pybullet   : Robot arm stepping + rendering → shared state
  composite  : Builds browser frame from shared state → Flask MJPEG buffer

Flask runs as a fifth daemon thread.
Open http://localhost:5000 in your browser after launch.

Controls (browser buttons or terminal commands + Enter)
-------------------------------------------------------
  r / Reset      — complete current trial and advance to next
  s / Screenshot — save the composite frame to results/screenshots/
  q / Quit       — clean shutdown + print summary if plan is complete
"""

import os
import sys
import threading
import time

import cv2
import numpy as np
from flask import Flask, Response, jsonify, send_from_directory

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from src.scene.targets import get_targets, HAND_START, CANVAS_W, CANVAS_H
from src.perception.webcam_perception import WebcamPerception, WebcamObservation
from src.inference.bayesian_goal_inference import BayesianGoalInference
from src.prediction.minimum_jerk import estimate_duration, minimum_jerk_trajectory
from src.robot.pybullet_robot import PybulletRobot
from src.server.stream_server import build_composite, SharedState, _encode_jpeg
from src.evaluation.metrics import DurationAdapter, TrialLogger, build_trial_result
from src.visualization.plots import plot_prediction_error, plot_summary_metrics

# 9-trial guided plan — 3 per target, interleaved so the arm sweeps to
# different world positions each trial (better for the demo video).
TRIAL_PLAN = [
    "red", "blue", "green",
    "blue", "green", "red",
    "green", "red", "blue",
]

_STATIC_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "static"
)


# ---------------------------------------------------------------------------
# Shared demo state
# ---------------------------------------------------------------------------

class DemoState:
    """
    All state shared across threads, protected by a single Lock.

    reset_count is incremented by the main/Flask thread; each worker
    thread stores its own last-seen value and resets when it detects a change.
    """

    def __init__(self, targets):
        self._lock = threading.Lock()
        self.targets = targets

        # Webcam thread writes
        self.raw_frame_bgr: np.ndarray | None = None
        self.hand_pos:  np.ndarray | None = None
        self.hand_vel:  np.ndarray = np.zeros(2)
        self.latest_obs: WebcamObservation | None = None

        # Inference thread writes
        self.posteriors: dict = {t.name: 1.0 / len(targets) for t in targets}
        self.locked_target = None
        self.lock_time: float | None = None
        self.lock_confidence: float | None = None
        self.xf_predicted: np.ndarray | None = None
        self.predicted_trajectory: list = []
        self.D_remaining: float | None = None
        self.D_adapted:   float | None = None
        self.trial_x0: np.ndarray | None = None   # first obs position this trial

        # PyBullet thread writes
        self.robot_frame: np.ndarray = np.zeros(
            (config.PYBULLET_RENDER_H, config.PYBULLET_RENDER_W, 3), dtype=np.uint8
        )
        self.robot_state_str: str = "IDLE"
        self.activate_robot:  bool = False         # inference → pybullet signal

        # Trial accounting
        self.trial_id:    int = 1
        self.reset_count: int = 0                  # increment to trigger reset
        self.xf_actual:   np.ndarray | None = None
        self.prediction_error_norm: float | None = None
        self.num_obs_this_trial: int = 0

        # Guided trial plan (Stage 9)
        self.trial_plan:    list[str] = TRIAL_PLAN
        self.plan_idx:      int = 0                # which trial in plan we're on
        self.plan_complete: bool = False
        self.all_results:   list = []              # TrialResult objects, one per trial

        # Control flags
        self.quit_flag:       bool = False
        self.save_flag:       bool = False
        self.last_screenshot: str = ""

    # Convenience accessors that lock internally
    def request_reset(self):
        with self._lock:
            self.reset_count += 1

    def request_save(self):
        with self._lock:
            self.save_flag = True

    @property
    def lock(self):
        return self._lock


# ---------------------------------------------------------------------------
# Overlay drawing helpers
# ---------------------------------------------------------------------------

def _draw_overlays(
    frame_bgr: np.ndarray,
    state: DemoState,
) -> np.ndarray:
    """Draw all visual overlays on a copy of the webcam frame (BGR)."""
    out = frame_bgr.copy()

    # Target rectangles
    for t in state.targets:
        x0, y0, x1, y1 = t.region
        cv2.rectangle(out, (x0, y0), (x1, y1), t.color_bgr, 2)
        cv2.putText(out, t.label, (x0, y0 - 7),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, t.color_bgr, 1, cv2.LINE_AA)

    # Start-zone guide
    hs = (int(HAND_START[0]), int(HAND_START[1]))
    cv2.circle(out, hs, 22, (70, 70, 70), 2)
    cv2.putText(out, "START", (hs[0] - 22, hs[1] + 38),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (90, 90, 90), 1)

    # Read all shared state in one lock acquisition
    with state.lock:
        hand_pos      = state.hand_pos
        locked_target = state.locked_target
        xf_predicted  = state.xf_predicted
        pred_traj     = list(state.predicted_trajectory)
        lock_time     = state.lock_time
        lock_conf     = state.lock_confidence
        D_rem         = state.D_remaining
        err_norm      = state.prediction_error_norm
        rob_state     = state.robot_state_str
        plan_idx      = state.plan_idx
        trial_plan    = state.trial_plan
        plan_complete = state.plan_complete

    # Guided trial instruction banner (top-centre)
    if trial_plan:
        if not plan_complete and plan_idx < len(trial_plan):
            tname = trial_plan[plan_idx]
            tobj  = next(t for t in state.targets if t.name == tname)
            txt   = f"Trial {plan_idx + 1}/{len(trial_plan)}   REACH FOR: {tobj.label.upper()}"
            (tw, _), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.62, 2)
            cx = CANVAS_W // 2
            cv2.rectangle(out, (cx - tw // 2 - 10, 6), (cx + tw // 2 + 10, 38), (25, 25, 25), -1)
            cv2.putText(out, txt, (cx - tw // 2, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.62, tobj.color_bgr, 2, cv2.LINE_AA)
        else:
            txt   = "PLAN COMPLETE  |  press q to quit"
            (tw, _), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.62, 2)
            cx = CANVAS_W // 2
            cv2.rectangle(out, (cx - tw // 2 - 10, 6), (cx + tw // 2 + 10, 38), (20, 60, 20), -1)
            cv2.putText(out, txt, (cx - tw // 2, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.62, (100, 255, 100), 2, cv2.LINE_AA)

    if hand_pos is not None:
        hp = (int(hand_pos[0]), int(hand_pos[1]))
        color = (0, 200, 255) if locked_target is None else (0, 255, 100)
        cv2.circle(out, hp, 14, color, 2, cv2.LINE_AA)
        cv2.circle(out, hp, 3,  color, -1)

    # Predicted trajectory (dashed grey)
    if pred_traj:
        pts = [(int(p[0]), int(p[1])) for p in pred_traj]
        for i in range(0, len(pts) - 1, 2):
            cv2.line(out, pts[i], pts[min(i + 1, len(pts) - 1)],
                     (140, 140, 140), 1, cv2.LINE_AA)

    # xf_predicted star
    if xf_predicted is not None:
        sp = (int(xf_predicted[0]), int(xf_predicted[1]))
        for ang in range(0, 360, 60):
            rad = np.radians(ang)
            outer = (int(sp[0] + 13 * np.cos(rad)), int(sp[1] + 13 * np.sin(rad)))
            rad2  = np.radians(ang + 30)
            inner = (int(sp[0] + 5  * np.cos(rad2)), int(sp[1] + 5 * np.sin(rad2)))
            cv2.line(out, inner, outer, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(out, "xf_pred", (sp[0] + 14, sp[1] - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (210, 210, 210), 1)

    # Status text (top-left)
    lines = ["mode: WEBCAM"]
    if locked_target is not None:
        lines.append(f"LOCKED: {locked_target.name}  conf={lock_conf:.2f}")
        lines.append(f"lock_t: {lock_time:.2f}s")
        lines.append(f"D_rem : {D_rem:.2f}s" if D_rem else "D_rem : --")
        lines.append(f"robot : {rob_state}")
        if err_norm is not None:
            lines.append(f"err   : {err_norm:.1f}px")
    else:
        lines.append("waiting for lock (P>0.80)")

    for i, line in enumerate(lines):
        cv2.putText(out, line, (8, 22 + i * 19),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.46, (200, 200, 200), 1, cv2.LINE_AA)

    return out


# ---------------------------------------------------------------------------
# Thread functions
# ---------------------------------------------------------------------------

def webcam_thread_fn(perception: WebcamPerception, state: DemoState):
    my_reset = 0
    while not state.quit_flag:
        obs = perception.update()
        bgr = perception.get_frame_bgr()

        with state.lock:
            rc = state.reset_count

        if rc > my_reset:
            my_reset = rc
            perception.reset_trial()

        with state.lock:
            state.raw_frame_bgr = bgr
            if obs is not None:
                state.hand_pos   = obs.position.copy()
                state.hand_vel   = obs.velocity.copy()
                state.latest_obs = obs
                state.num_obs_this_trial += 1
            else:
                state.hand_pos   = None
                state.hand_vel   = np.zeros(2)
                state.latest_obs = None


def inference_thread_fn(inference: BayesianGoalInference, state: DemoState,
                        adapter: DurationAdapter):
    my_reset = 0
    waiting_for_first = True   # need first obs to anchor start_pos
    last_obs_ts: float | None = None  # skip observations already processed

    while not state.quit_flag:
        with state.lock:
            rc  = state.reset_count
            obs = state.latest_obs

        # --- Handle reset ---
        if rc > my_reset:
            my_reset = rc
            waiting_for_first = True
            last_obs_ts = None
            inference.reset()                 # soft reset; goal_dirs updated on first obs
            with state.lock:
                state.locked_target       = None
                state.lock_time           = None
                state.lock_confidence     = None
                state.xf_predicted        = None
                state.predicted_trajectory = []
                state.D_remaining         = None
                state.D_adapted           = None
                state.trial_x0            = None
                state.xf_actual           = None
                state.prediction_error_norm = None
                state.num_obs_this_trial  = 0
                state.activate_robot      = False

        # Don't process the same observation twice — the webcam thread pushes
        # ~30 frames/s; without this guard the tight loop would apply the same
        # likelihood dozens of times before the next frame arrives, causing
        # near-instant probability collapse.
        if obs is None or obs.timestamp == last_obs_ts:
            time.sleep(0.005)
            continue
        last_obs_ts = obs.timestamp

        # Anchor inference start at first observation of this trial
        if waiting_for_first:
            inference.reset(start_pos=obs.position)
            with state.lock:
                state.trial_x0 = obs.position.copy()
            waiting_for_first = False

        posterior = inference.update(obs)

        with state.lock:
            state.posteriors      = posterior
            state.locked_target   = inference.locked_target
            state.lock_time       = inference.lock_time
            state.lock_confidence = inference.lock_confidence

        # One-shot prediction at first lock
        if inference.locked_target is not None:
            with state.lock:
                already_predicted = state.xf_predicted is not None
                trial_x0          = state.trial_x0

            if not already_predicted and trial_x0 is not None:
                xf_pred = inference.locked_target.position.copy()
                speed   = float(np.linalg.norm(obs.velocity))
                D_rem   = estimate_duration(
                    x0=trial_x0,
                    x_current=obs.position,
                    xf=xf_pred,
                    current_speed=speed,
                )
                D_ada   = adapter.apply(D_rem)
                pred_traj = minimum_jerk_trajectory(
                    obs.position, xf_pred, D_rem, config.SIMULATION_DT
                )
                with state.lock:
                    state.xf_predicted        = xf_pred
                    state.D_remaining         = D_rem
                    state.D_adapted           = D_ada
                    state.predicted_trajectory = pred_traj
                    state.activate_robot      = True     # signal PyBullet thread


def pybullet_thread_fn(robot: PybulletRobot, state: DemoState):
    my_reset = 0
    while not state.quit_flag:
        with state.lock:
            rc = state.reset_count

        if rc > my_reset:
            my_reset = rc
            robot.reset()

        with state.lock:
            activate = state.activate_robot
            locked   = state.locked_target
            D_ada    = state.D_adapted

        if activate and locked is not None:
            D_robot = (D_ada * config.ROBOT_D_SCALE) if D_ada else 1.0
            robot.activate(locked.name, D_robot)
            with state.lock:
                state.activate_robot = False

        robot.step()
        frame = robot.render()

        with state.lock:
            state.robot_frame    = frame
            state.robot_state_str = robot.state

        time.sleep(config.SIMULATION_DT)


def composite_thread_fn(state: DemoState, shared_jpeg: SharedState):
    while not state.quit_flag:
        with state.lock:
            bgr          = state.raw_frame_bgr
            posteriors   = dict(state.posteriors)
            robot_frame  = state.robot_frame.copy()
            save_flag    = state.save_flag

        if bgr is None:
            # No webcam frame yet — push a blank placeholder
            blank = np.zeros((CANVAS_H, CANVAS_W, 3), dtype=np.uint8)
            cv2.putText(blank, "Waiting for webcam...", (160, 240),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 100, 100), 2)
            composite = build_composite(
                scene_frame=blank,
                robot_frame=robot_frame,
                posteriors=posteriors,
                targets=state.targets,
            )
            shared_jpeg.write(_encode_jpeg(composite))
            time.sleep(1 / 15)
            continue

        overlaid_bgr = _draw_overlays(bgr, state)
        scene_rgb    = cv2.cvtColor(overlaid_bgr, cv2.COLOR_BGR2RGB)

        composite = build_composite(
            scene_frame=scene_rgb,
            robot_frame=robot_frame,
            posteriors=posteriors,
            targets=state.targets,
        )

        shared_jpeg.write(_encode_jpeg(composite))

        # Screenshot
        if save_flag:
            os.makedirs("results/screenshots", exist_ok=True)
            fname = f"results/screenshots/webcam_{int(time.time())}.png"
            cv2.imwrite(fname, cv2.cvtColor(composite, cv2.COLOR_RGB2BGR))
            with state.lock:
                state.save_flag       = False
                state.last_screenshot = fname
            print(f"  Screenshot saved → {fname}")

        time.sleep(1 / 30)


def stdin_thread_fn(state: DemoState, logger: TrialLogger, adapter: DurationAdapter):
    """Block on input() so user can type r/s/q + Enter in the terminal."""
    print("  Terminal: type  r + Enter = reset  |  s + Enter = screenshot  |  q + Enter = quit")
    while not state.quit_flag:
        try:
            cmd = input().strip().lower()
        except (EOFError, KeyboardInterrupt):
            state.quit_flag = True
            break
        if cmd == "r":
            _do_reset(state, logger, adapter)
        elif cmd == "s":
            state.request_save()
        elif cmd == "q":
            state.quit_flag = True


# ---------------------------------------------------------------------------
# Trial logging helper
# ---------------------------------------------------------------------------

def _do_reset(state: DemoState, logger: TrialLogger, adapter: DurationAdapter):
    """Log the completed trial, adapt D correction, advance plan, signal reset."""
    with state.lock:
        plan_idx   = state.plan_idx
        trial_plan = state.trial_plan
        predicted  = state.locked_target.name if state.locked_target else None
        lock_time  = state.lock_time
        lock_conf  = state.lock_confidence
        xf_pred    = state.xf_predicted.copy() if state.xf_predicted is not None else None
        xf_act     = state.hand_pos.copy() if state.hand_pos is not None else None
        D_est      = state.D_remaining
        D_ada      = state.D_adapted
        D_act      = None
        if lock_time is not None and state.latest_obs is not None:
            D_act = state.latest_obs.timestamp - lock_time
        n_frames   = state.num_obs_this_trial
        trial_id   = state.trial_id

    # Ground truth: from the guided plan when available, else fall back to prediction
    gt = trial_plan[plan_idx] if trial_plan and plan_idx < len(trial_plan) else (predicted or "unknown")
    plan_label = f"{plan_idx + 1}/{len(trial_plan)}" if trial_plan else "--"

    if xf_act is not None:
        result = build_trial_result(
            trial_id=trial_id,
            mode="webcam",
            ground_truth_target=gt,
            predicted_target=predicted,
            lock_time=lock_time,
            lock_confidence=lock_conf,
            xf_predicted=xf_pred,
            xf_actual=xf_act,
            D_estimated=D_est,
            D_actual=D_act,
            D_adapted=D_ada,
            num_frames=n_frames,
            notes=f"webcam guided {plan_label}",
        )
        logger.log(result)
        if D_est and D_act:
            adapter.adapt(D_est, D_act)

        correct_str = "ok" if result.target_correct else "WRONG"
        print(f"  [{plan_label}] gt={gt:5s}  locked={str(predicted):5s}  {correct_str:5s} | "
              f"err={result.prediction_error_norm:.1f}px | "
              f"D_corr={adapter.correction_factor:.3f}")

        with state.lock:
            state.all_results.append(result)

    # Advance plan index; mark complete when all trials done
    plan_just_finished = False
    with state.lock:
        if trial_plan and state.plan_idx < len(trial_plan):
            state.plan_idx += 1
            if state.plan_idx >= len(trial_plan):
                state.plan_complete = True
                plan_just_finished = True
        state.trial_id    += 1
        state.reset_count += 1

    if plan_just_finished:
        _finish_plan(state, adapter)


# ---------------------------------------------------------------------------
# Plan completion summary
# ---------------------------------------------------------------------------

def _finish_plan(state: DemoState, adapter: DurationAdapter):
    """Print summary table and save figures once all plan trials are done."""
    with state.lock:
        results = list(state.all_results)

    n_total   = len(results)
    n_correct = sum(r.target_correct for r in results)
    errors    = [r.prediction_error_norm for r in results
                 if r.prediction_error_norm == r.prediction_error_norm]
    lock_times = [r.lock_time for r in results
                  if r.lock_time is not None and r.lock_time == r.lock_time]

    print()
    print("=" * 52)
    print(f"  Stage 9 Summary — {n_total} webcam trials")
    print("=" * 52)
    print(f"  Accuracy    : {n_correct}/{n_total}  ({100 * n_correct / n_total:.0f}%)")
    if errors:
        print(f"  Mean error  : {np.mean(errors):.1f} px  (std={np.std(errors):.1f})")
    if lock_times:
        print(f"  Mean lock   : {np.mean(lock_times):.2f} s")
    print(f"  D_correct   : {adapter.correction_factor:.3f}")
    print("=" * 52)

    # Per-trial breakdown
    print()
    print(f"  {'#':>2}  {'gt':>5}  {'pred':>5}  {'ok':>5}  {'lock_t':>7}  {'err_px':>7}")
    print("  " + "-" * 42)
    for r in results:
        ok_str = "ok" if r.target_correct else "WRONG"
        lt_str = f"{r.lock_time:.2f}s" if r.lock_time == r.lock_time else "  --  "
        ep_str = f"{r.prediction_error_norm:.1f}" if r.prediction_error_norm == r.prediction_error_norm else " --"
        print(f"  {r.trial_id:>2}  {r.ground_truth_target:>5}  {str(r.predicted_target):>5}  "
              f"{ok_str:>5}  {lt_str:>7}  {ep_str:>7}")

    print()
    os.makedirs("results/figures", exist_ok=True)
    plot_prediction_error(results, save_path="results/figures/prediction_error_webcam.png")
    plot_summary_metrics(results,  save_path="results/figures/summary_metrics_webcam.png")
    print("  Figures → results/figures/prediction_error_webcam.png")
    print("            results/figures/summary_metrics_webcam.png")
    print("  Press q (or browser Quit) to exit.\n")

    # Auto-save a final screenshot
    state.request_save()


# ---------------------------------------------------------------------------
# Flask app
# ---------------------------------------------------------------------------

def build_flask_app(state: DemoState, shared_jpeg: SharedState,
                    logger: TrialLogger, adapter: DurationAdapter) -> Flask:
    app = Flask(__name__, static_folder=None)

    @app.route("/")
    def index():
        return send_from_directory(_STATIC_DIR, "webcam.html")

    @app.route("/video_feed")
    def video_feed():
        def generate():
            while True:
                jpeg = shared_jpeg.read()
                if jpeg:
                    yield (
                        b"--frame\r\n"
                        b"Content-Type: image/jpeg\r\n\r\n"
                        + jpeg + b"\r\n"
                    )
                time.sleep(1 / 30)
        return Response(generate(),
                        mimetype="multipart/x-mixed-replace; boundary=frame")

    @app.route("/reset", methods=["POST"])
    def reset():
        _do_reset(state, logger, adapter)
        return jsonify({"status": "Trial reset"})

    @app.route("/save", methods=["POST"])
    def save():
        state.request_save()
        return jsonify({"status": "Screenshot queued"})

    @app.route("/quit", methods=["POST"])
    def quit_demo():
        state.quit_flag = True
        return jsonify({"status": "Shutting down..."})

    return app


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    os.makedirs("results/screenshots", exist_ok=True)
    os.makedirs("results/logs", exist_ok=True)

    targets    = get_targets()
    perception = WebcamPerception()
    robot      = PybulletRobot()
    inference  = BayesianGoalInference(targets)
    adapter    = DurationAdapter()
    logger     = TrialLogger()

    state       = DemoState(targets)
    shared_jpeg = SharedState()

    # Build and start Flask
    app = build_flask_app(state, shared_jpeg, logger, adapter)
    flask_thread = threading.Thread(
        target=lambda: app.run(
            host="0.0.0.0", port=config.FLASK_PORT,
            debug=False, use_reloader=False,
        ),
        daemon=True, name="flask",
    )
    flask_thread.start()
    print(f"\n  HRI Webcam Demo — Stage 9 Guided Trials")
    print(f"  → Open http://localhost:{config.FLASK_PORT} in your browser")
    print(f"  Plan: {' → '.join(TRIAL_PLAN)}")
    print(f"  Press r (Reset) after each trial to advance to the next target.\n")

    # Worker threads
    threads = [
        threading.Thread(target=webcam_thread_fn,
                         args=(perception, state), daemon=True, name="webcam"),
        threading.Thread(target=inference_thread_fn,
                         args=(inference, state, adapter), daemon=True, name="inference"),
        threading.Thread(target=pybullet_thread_fn,
                         args=(robot, state), daemon=True, name="pybullet"),
        threading.Thread(target=composite_thread_fn,
                         args=(state, shared_jpeg), daemon=True, name="composite"),
    ]
    for t in threads:
        t.start()

    # Stdin keyboard handler (daemon so it doesn't block clean exit)
    kb_thread = threading.Thread(
        target=stdin_thread_fn, args=(state, logger, adapter),
        daemon=True, name="stdin",
    )
    kb_thread.start()

    # Main thread: wait for quit signal
    try:
        while not state.quit_flag:
            time.sleep(0.2)
    except KeyboardInterrupt:
        state.quit_flag = True

    print("\n  Shutting down...")
    perception.release()
    robot.disconnect()
    print("  Done.")


if __name__ == "__main__":
    main()
