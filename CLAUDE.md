# CLAUDE.md

This file is the implementation guide for Claude Code. The repository starts empty, so build the project from scratch in small, working stages. Each stage must leave the project runnable.

Project title: **Real-Time Human Intent Recognition for Proactive Robot Handover**

The final project must be a **real-time perception HRI project** using a **webcam**. This matters because the assignment states that real-time perception projects can receive the full project mark, while simulated perception projects are capped lower. Simulation may be used only as an early development/debugging tool. The final demo, report, video, and interview must clearly show real webcam-based perception driving the HRI loop.

The goal is to build a Python-only Human-Robot Interaction demo. The system observes a human hand or wrist moving toward one of several visible candidate objects, infers the intended object using Bayesian goal recognition, predicts the final hand/interception point using a minimum-jerk trajectory model, and visualizes a proactive virtual robot/gripper response.

The final project should match the assignment concept:

> The robot must first figure out which object the human wants, then calculate where and when the human hand will stop so the robot can proactively place or prepare the object at the predicted interception point.

Do not implement everything at once. Build from a minimal runnable project to a complete webcam-based demo.

**Python interpreter:** Always run this project with `$HOME/miniforge3/envs/hri/bin/python`. Do not use bare `python` or `python3`.

---

## 1. Assignment Requirements That Must Be Explicitly Shown

The final result must visibly and explainably implement the assignment requirements below. Do not hide these ideas inside code only; they must also appear in the README, report material, and demo explanation.

### 1.1 Project Route and Marking Rule

This project is the **real-time perception route**, not the simulated perception route.

Required final perception mode:

- Webcam-based live hand/wrist/fingertip tracking.
- Live observations must feed the inference system.
- The robot/visual response must depend on the live webcam observation, not only on pre-generated trajectories.

Allowed development support:

- Simulated trajectories may be implemented first to test the math.
- Simulation must not be presented as the final project outcome.
- The final demo should clearly run in webcam mode.

### 1.2 Main Technical Concept

Implement a **Proactive Target & Interception System** with two main steps.

#### Step 1: Bayesian Inference

Estimate the probability of each possible goal/object given the current hand observation:

```text
P(G | O) = P(O | G) * P(G) / sum_i(P(O | G_i) * P(G_i))
```

The implementation must continuously update the probability of each target.

#### Step 2: Minimum-Jerk Trajectory Prediction

Once the target probability is high enough, predict the final/interception point using the minimum-jerk trajectory idea:

```text
x(t) = x0 + (xf - x0) * (10*tau^3 - 15*tau^4 + 6*tau^5)
tau = t / D
```

The final project should use this model to generate a predicted remaining hand/robot trajectory or smooth virtual robot movement.

### 1.3 Required Confidence Rule

Continuously update target probabilities and lock/confirm the target when:

```text
P(G | O) > 0.8
```

The demo must show this threshold being used, for example through a printed status, live probability bars, or a lock indicator.

### 1.4 Required HRI Loop

The implementation must clearly show this HRI loop:

1. **Perception** — webcam observes the human hand/wrist/fingertip in real time.
2. **Inference / Intention Recognition** — Bayesian target probability update for all candidate targets.
3. **Decision / Planning** — select the most likely target after confidence passes `0.8`.
4. **Robot Response / Action** — show a virtual robot/gripper/marker moving or preparing toward the predicted interception/target point.
5. **Feedback** — compare the predicted final point with the actual final hand stopping point.
6. **Adaptation** — update a simple timing/duration parameter for later trials.

### 1.5 Required Error Measurement

At the end of each trial, compute and log:

```text
error = xf_predicted - xf_actual
```

Also log the Euclidean norm of this error as the scalar prediction error.

### 1.6 Required Candidate Objects

Use three visible candidate targets, matching the assignment idea of multiple blocks/tools:

- red block
- blue block
- green block

In webcam mode, these can be represented as fixed target regions drawn in the application window, physical colored paper/blocks on the desk, or manually calibrated screen/table coordinates.

---

## 2. Marking Scheme Checklist

The implementation should be planned around the assessment, not only around code completion.

### 2.1 Report: 10 Marks

The repository should generate enough material for the report sections below:

- **Problem definition, objectives, and HRI relevance** — explain the handover/intention-recognition problem and why proactive robot response is useful.
- **System design and technical explanation** — explain webcam perception, Bayesian inference, target lock, minimum-jerk prediction, robot response, feedback, and adaptation.
- **Implementation methodology and project development** — describe the staged development from basic scene, to tested inference, to webcam integration.
- **Results, discussion, and evaluation** — include target accuracy, lock time, prediction error, screenshots/figures, and limitations.
- **Clarity, structure, figures, and presentation** — save clean figures and logs for direct use.

### 2.2 Video: 10 Marks

The demo should support a 10-minute video with:

- Clear explanation of the problem and objective.
- Short system design explanation.
- A high-quality demonstration of the webcam-based system.
- Clear flow: perception → inference → decision → action → feedback/adaptation.
- Professional timing and completeness.

### 2.3 Interview / Viva: 50 Marks

Prioritize explainable implementation decisions. The student must be able to answer:

- What is the full project workflow?
- How does webcam perception become a usable observation?
- How are Bayesian probabilities computed and normalized?
- Why is `P(G | O) > 0.8` used as the target lock rule?
- How does the minimum-jerk equation create a smooth predicted trajectory?
- How is prediction error measured?
- What HRI concepts are demonstrated?
- What are the limitations and possible improvements?

---

## 3. Technology Decision

Use **only the Python implementation path**.

Recommended libraries:

- `numpy` for math and arrays
- `opencv-python` for webcam capture and frame processing
- `mediapipe` for webcam hand/wrist/fingertip tracking
- `matplotlib` for saved offline figures and plots
- `pandas` for saving trial logs
- `pytest` for lightweight tests
- `pybullet` for physics-based robot arm simulation and inverse kinematics
- `flask` for the browser-based GUI server (MJPEG stream endpoint)
- `pillow` for JPEG frame encoding in the MJPEG stream

Do not create a Unity version. Do not create an AR version. Keep the whole project understandable as a Python course demo. The browser GUI uses plain Flask with a single HTML page — no JavaScript framework, no frontend build step.

---

## 4. Repository Architecture From Scratch

Because the repository starts empty, create the project structure gradually. Do not assume any existing files.

Target structure:

```text
project-root/
│
├── CLAUDE.md
├── README.md
├── requirements.txt
├── config.py
│
├── src/
│   ├── main.py
│   │
│   ├── scene/
│   │   ├── __init__.py
│   │   └── targets.py
│   │
│   ├── perception/
│   │   ├── __init__.py
│   │   ├── webcam_perception.py
│   │   └── simulated_perception.py
│   │
│   ├── inference/
│   │   ├── __init__.py
│   │   └── bayesian_goal_inference.py
│   │
│   ├── prediction/
│   │   ├── __init__.py
│   │   └── minimum_jerk.py
│   │
│   ├── robot/
│   │   ├── __init__.py
│   │   └── pybullet_robot.py      ← PyBullet arm setup, IK, joint stepping
│   │
│   ├── server/
│   │   ├── __init__.py
│   │   └── stream_server.py       ← Flask MJPEG server, composite frame builder
│   │
│   ├── visualization/
│   │   ├── __init__.py
│   │   └── plots.py               ← offline matplotlib figures only
│   │
│   └── evaluation/
│       ├── __init__.py
│       └── metrics.py
│
├── static/
│   └── index.html                 ← single-page browser UI (one <img> tag)
│
├── experiments/
│   ├── run_simulation.py
│   ├── run_webcam_demo.py         ← starts Flask server; user opens localhost:5000
│   └── run_trials.py
│
├── results/
│   ├── figures/
│   ├── screenshots/
│   ├── animations/
│   └── logs/
│
└── tests/
    ├── test_bayesian_goal_inference.py
    ├── test_minimum_jerk.py
    └── test_metrics.py
```

This structure is a guide, not a strict interface contract. Keep modules separated by responsibility, but let Claude choose the exact functions/classes naturally while implementing.

---

## 5. Implementation Rules

- Build the project stage by stage.
- After every stage, the project must still run.
- Make webcam perception the final required mode.
- Keep simulation only as an early debugging tool and fallback for testing.
- Keep the math visible and explainable.
- Keep webcam perception and simulated perception separate.
- The robot is simulated with PyBullet. Keep the arm model simple (2-DOF or 3-DOF planar). Do not implement full dynamics or collision — kinematics and IK are enough.
- Keep configuration values in `config.py`.
- Save generated plots, screenshots, logs, and optional animations in `results/`.
- Use readable names such as `posterior`, `goal`, `observation`, `target`, `trajectory`, `xf_predicted`, `xf_actual`, `D`, `tau`, and `prediction_error`.
- Avoid hard-coding a single target.
- Normalize posterior probabilities after every Bayesian update.
- Prefer simple working code over complex abstractions.
- Do not give the final user only a simulated demo, because that would not satisfy the intended real-time perception route.

---

## 6. GUI and Visualization Decisions

### 6.1 Browser-Based Real-Time Display (Stage 8 onward)

The live demo is served through a **Flask web server** and viewed in a browser at `localhost:5000`. There is no OpenCV display window in the final demo. The browser shows a single composite image streamed as an **MJPEG feed**.

**Composite image layout (assembled server-side as a numpy array):**

```
┌─────────────────────────────────┬──────────────────┐
│                                 │                  │
│   Webcam feed (640×480)         │  PyBullet robot  │
│   + hand circle overlay         │  arm render      │
│   + target region boxes         │  (400×400)       │
│   + predicted trajectory        │                  │
│   + xf_predicted star           ├──────────────────┤
│   + status text overlay         │  Probability     │
│                                 │  bars panel      │
│                                 │  (R / B / G)     │
└─────────────────────────────────┴──────────────────┘
```

**MJPEG streaming architecture:**

- Flask serves `GET /video_feed` with `Content-Type: multipart/x-mixed-replace`
- Each frame is JPEG-encoded with Pillow and pushed as a multipart boundary
- The browser displays it via a plain `<img src="/video_feed">` tag — no JavaScript needed
- `static/index.html` contains only this one tag plus minimal CSS for layout

**Threading model — four concurrent threads:**

| Thread | Responsibility |
|--------|---------------|
| Webcam thread | Captures frames, runs MediaPipe, smooths position, estimates velocity |
| PyBullet thread | Steps the physics simulation, updates joint positions toward IK target |
| Inference thread | Runs Bayesian update and prediction on each new observation |
| Flask thread | Composites the latest frame + robot render + bars, serves MJPEG |

A `threading.Lock` protects the shared composite frame buffer. Each thread writes its output to a shared state object; the Flask thread reads from it to build each MJPEG frame.

**Overlay conventions (drawn on composite numpy array with OpenCV):**

- Circle: detected hand/wrist/fingertip position
- Rectangles: red, blue, green target regions in their respective colours
- Star (`*`): `xf_predicted` marker
- Dashed line: predicted remaining hand trajectory
- Status text (top-left): mode, locked target, confidence, lock time, estimated D, prediction error
- Probability bars (right panel): filled rectangles scaled by posterior value, one per target in target colour

**Screenshots for the report:**
- Save the full browser composite frame (not just the webcam crop) using `cv2.imwrite` on the shared numpy array
- Save key frames: at lock moment, at hand final stop, after robot settled

### 6.2 Offline Figures (matplotlib only)

Use `matplotlib` only for saved figures — never for the live demo:

- `results/figures/static_scene.png`
- `results/figures/scene_trajectory.png`
- `results/figures/posterior_probabilities.png`
- `results/figures/interception_prediction.png`
- `results/figures/prediction_error.png`
- `results/figures/summary_metrics.png`

---

## 7. Core System Components

### 7.1 Scene and Candidate Goals

The scene should contain three candidate objects on a 2D coordinate system. A 2D implementation is enough for the course demo and easier to visualize.

Each goal should have:

- name
- position or rectangular region
- color
- display label

Suggested layout in the webcam/demo window:

```text
red block:   left side
blue block:  middle
green block: right side
```

For webcam use, allow a simple calibration mode or fixed screen coordinates. The targets do not need to be physically detected by color if that makes the project fragile. It is acceptable to draw fixed target regions and ask the user to move their hand toward those regions.

### 7.2 Webcam Perception

This is the final required perception mode.

For webcam mode:

- Capture frames using OpenCV.
- Track wrist or fingertip position using MediaPipe or a similar hand tracker.
- Convert image coordinates into the same coordinate system used by the target regions.
- Smooth noisy observations with a simple moving average or exponential smoothing.
- Estimate velocity from recent positions and timestamps.
- Feed each observation into the same Bayesian inference and prediction logic.

The webcam demo should gracefully handle:

- no hand detected
- temporary tracking loss
- low confidence frames
- user pressing a key to reset the trial
- user pressing a key to save a screenshot/log

### 7.3 Simulated Perception for Early Testing Only

Before webcam tracking is stable, generate simulated hand/wrist observations to test the math.

Each observation should contain:

- current position
- velocity
- timestamp

The simulated hand should move from a start position toward one selected ground-truth target. Add small noise only after the clean version works.

Do not present simulation as the final route. It is only a development and testing tool.

### 7.4 Bayesian Goal Inference

At every time step:

1. Compute a likelihood for each goal.
2. Multiply the likelihood by the previous belief/prior.
3. Normalize all probabilities.
4. Store the posterior history.
5. Lock the target when the maximum posterior is greater than `0.8`.

Use two complementary cues:

- **Off-axis distance** — how far the hand's current position deviates from the straight-line path between `HAND_START` and goal G
- **Direction** — cosine similarity between the hand's velocity vector and the direction toward goal G

```text
likelihood = off_axis_likelihood * direction_likelihood
```

where:

```text
off_axis_distance    = perpendicular distance from hand to the START→G line
off_axis_likelihood  = exp(-off_axis_normalized / DISTANCE_SCALE)

direction_likelihood = exp(DIRECTION_WEIGHT * cosine_similarity(velocity, direction_to_G))
```

Coordinates are normalized to [0, 1] before distance computation so `DISTANCE_SCALE` is dimensionless. Use `DISTANCE_SCALE = 0.15` (a 15%-of-canvas deviation gives likelihood ≈ 0.37). Set `direction_likelihood = 1.0` (neutral) when speed is below 5 px/s.

**Why off-axis instead of raw distance to goal:** raw distance only becomes informative when the hand is nearly at the goal. Off-axis distance discriminates from the first frame of movement — a hand moving toward green immediately has near-zero off-axis deviation for the green path and large deviation for red and blue paths.

The implementation must remain easy to explain in the interview.

### 7.5 Minimum-Jerk Prediction

Use the minimum-jerk equation to generate smooth movement from the current hand/robot position to the predicted final point.

**`xf_predicted`:** set to `locked_target.position` (the centre of the locked target region). Simple, defensible, and correct when the hand heads directly toward the target centre.

**D estimation using tau back-calculation:**

Naive `D = remaining_distance / current_speed` fails badly when the lock happens early (the hand is slow at the start of minimum-jerk motion and the estimate can be 4–5× too large). Instead:

1. Compute `progress = |x_current − x0| / |xf − x0|`
2. Solve `s(τ) = progress` for τ, where `s(τ) = 10τ³ − 15τ⁴ + 6τ⁵` (polynomial root via `np.roots`)
3. Evaluate the normalized speed profile: `g(τ) = 30τ² − 60τ³ + 30τ⁴`
4. Back-calculate total duration: `D_total = total_dist * g(τ) / current_speed`
5. Remaining duration: `D_remaining = D_total * (1 − τ)`

This uses the minimum-jerk model to figure out where in the movement we are, then derives remaining time from that. Falls back to `remaining_dist / 60.0` when speed < 5 px/s.

**Robot arm duration:** `D_robot = D_remaining × ROBOT_D_SCALE` (default 0.85) so the arm arrives slightly before the hand, making proactiveness visible.

The prediction module (`src/prediction/minimum_jerk.py`) is shared by both simulation (trajectory generation) and webcam demo (real-time prediction). No duplication of the formula.

### 7.6 Robot Action (PyBullet)

The robot is simulated using **PyBullet** in `DIRECT` mode (no PyBullet GUI window — rendering is done via `getCameraImage`).

**Arm model:** a simple 2-DOF or 3-DOF planar revolute-joint arm defined programmatically or loaded from a minimal URDF. Keep it simple enough to explain the kinematics in the viva.

**Robot base position:** placed at the lower-right area of the scene (e.g. pixel coords `(600, 380)`) so the arm sweeps upward and leftward toward whichever target is locked. This creates a clear visual of the robot reaching across the workspace.

**Workflow after target lock:**

1. Convert `xf_predicted` (pixel coords) to PyBullet world coordinates.
2. Call PyBullet IK (`calculateInverseKinematics`) to get joint angles for that end-effector position.
3. Interpolate from current joint angles to target joint angles using a minimum-jerk profile over `D_robot` seconds.
4. Step the simulation at each frame, updating joint positions.

**Rendering and compositing:**

```python
width, height, rgba, depth, seg = pybullet.getCameraImage(
    width=PYBULLET_RENDER_W,
    height=PYBULLET_RENDER_H,
    renderer=pybullet.ER_TINY_RENDERER,
)
robot_frame = np.array(rgba, dtype=np.uint8)[:, :, :3]  # drop alpha
```

This numpy array is composited into the right panel of the browser frame.

**State machine:**
- `IDLE`: before lock, arm rests at home position
- `MOVING`: after lock, arm interpolates to IK solution over `D_robot` seconds
- `HOLDING`: arm held at target, waiting for trial reset

The action must visually demonstrate that the robot starts moving **before** the hand fully reaches the object — this is the core proactiveness claim of the project.

### 7.7 Feedback and Adaptation

At the end of every trial:

- compare predicted target with true/selected target
- compare `xf_predicted` with `xf_actual`
- compute prediction error
- log trial results
- update a simple duration/timing parameter for the next trial

Keep adaptation simple and explainable. For example, adjust the estimated duration `D` based on the previous timing error or prediction error trend.

---

## 8. Staged Implementation Plan

Follow this order. Do not skip to later stages until the previous stage runs correctly.

### Stage 1: Empty Repository Setup

Goal: create a runnable Python project from nothing.

Tasks:

- Create the folder structure.
- Add `requirements.txt`.
- Add `README.md`.
- Add `config.py`.
- Add `src/main.py`.
- Add basic result-folder creation.

Suggested initial config values:

```python
CONFIDENCE_THRESHOLD = 0.8
NUM_OBJECTS = 3
RANDOM_SEED = 42
SIMULATION_DT = 0.05
DISTANCE_SCALE = 0.15      # normalized [0,1]; 15% off-axis deviation → likelihood ≈ 0.37
DIRECTION_WEIGHT = 2.0
ADAPTATION_GAIN = 0.1
WEBCAM_INDEX = 0
SMOOTHING_ALPHA = 0.35
FLASK_PORT = 5000
PYBULLET_RENDER_W = 400    # width of PyBullet camera render in pixels
PYBULLET_RENDER_H = 400    # height of PyBullet camera render in pixels
ROBOT_D_SCALE = 0.85       # robot arm arrives at D_remaining * ROBOT_D_SCALE seconds
```

Acceptance criteria:

- `python src/main.py` runs.
- The program prints a startup message.
- The program creates `results/figures`, `results/screenshots`, `results/animations`, and `results/logs` if they do not exist.

### Stage 2: Static Scene and Targets

Goal: create the target layout.

Tasks:

- Represent red, blue, and green goals.
- Assign fixed 2D positions or rectangular regions.
- Plot/save a static scene.

Acceptance criteria:

- The figure clearly shows the three blocks.
- Each block has a label.
- A static scene figure is saved to `results/figures/static_scene.png`.

### Stage 3: Simulated Motion Smoke Test

Goal: test the HRI math before webcam integration.

Tasks:

- Generate a smooth trajectory from a start point to a chosen ground-truth target.
- Store position, velocity, and timestamp.
- Plot the trajectory on the scene.

Acceptance criteria:

- The simulated hand trajectory reaches the chosen target.
- The ground-truth target is known and printed.
- A trajectory figure is saved.

### Stage 4: Bayesian Goal Recognition

Goal: infer the intended target from observations.

Tasks:

- Implement Bayesian posterior updates for all three goals.
- Use distance and velocity direction in the likelihood.
- Normalize posterior probabilities at every step.
- Save posterior history.
- Lock target when confidence is greater than `0.8`.

Acceptance criteria:

- Posterior probabilities are printed or plotted.
- The correct target probability usually increases during clean motion.
- The target lock frame/time is printed.
- A probability plot is saved to `results/figures/posterior_probabilities.png`.

### Stage 5: Minimum-Jerk Prediction

Goal: predict the final hand/interception point.

Tasks:

- Implement the minimum-jerk trajectory equation.
- After target lock, estimate `xf_predicted`.
- Estimate duration `D`.
- Generate the predicted remaining trajectory.

Acceptance criteria:

- `xf_predicted` and `D` are printed.
- The predicted endpoint is shown in the scene.
- The predicted trajectory is visually distinct from the observed trajectory.

### Stage 6: PyBullet Robot Action + Browser Composite

Goal: simulate a robot arm with PyBullet, composite its render into a browser-served MJPEG stream alongside the main scene, and demonstrate proactive robot response.

Tasks:

- Create `src/robot/pybullet_robot.py`:
  - Launch PyBullet in `DIRECT` mode (no PyBullet GUI window).
  - Define a simple 2-DOF or 3-DOF planar revolute arm programmatically or from a minimal URDF.
  - Implement `activate(xf_predicted, D_robot)` — computes IK target joint angles.
  - Implement `step(dt)` — advances joint positions toward target using minimum-jerk interpolation.
  - Implement `render()` — calls `pybullet.getCameraImage()` and returns an RGB numpy array.
- Create `src/server/stream_server.py`:
  - Flask app with a single `/video_feed` MJPEG endpoint.
  - Composite thread: combines scene frame + PyBullet render + probability bars into one numpy array.
  - Serves `static/index.html` at `/`.
  - Uses `threading.Lock` to protect the shared composite frame.
- Create `static/index.html` — one `<img src="/video_feed">` tag.
- Update `experiments/run_simulation.py` to activate the robot after lock and save the static interception figure.

Acceptance criteria:

- Running `python experiments/run_simulation.py` saves `results/figures/interception_prediction.png` showing the robot arm at `xf_predicted`.
- The robot arm is positioned by IK, not manually placed.
- The arm starts moving after target lock, not after the hand stops.
- PyBullet render is a proper camera image, not a matplotlib line drawing.

### Stage 7: Feedback, Adaptation, and Trial Logging

Goal: evaluate each trial and update a simple parameter.

Tasks:

- Compute `xf_predicted - xf_actual`.
- Compute scalar prediction error.
- Record target correctness.
- Record lock time.
- Update a simple duration/timing parameter for the next trial.
- Save logs to `results/logs/`.

Acceptance criteria:

- A CSV or JSON log is saved.
- Error values are printed.
- Adaptation values change across trials.

### Stage 8: Webcam Perception Integration + Live Browser Demo

Goal: make the project a real-time perception project with a single browser window showing the full HRI loop.

Tasks:

- Create `src/perception/webcam_perception.py`:
  - Capture frames with OpenCV (`VideoCapture`).
  - Run MediaPipe Hands to extract wrist/fingertip pixel position.
  - Apply exponential smoothing (`SMOOTHING_ALPHA`) to position.
  - Estimate velocity from last two positions and timestamps.
  - Package as an observation compatible with `BayesianGoalInference.update()`.
- Wire the four threads in `experiments/run_webcam_demo.py`:
  1. **Webcam thread** — captures frames, runs MediaPipe, pushes observations to a queue.
  2. **Inference + PyBullet thread** — consumes observations, updates posteriors, runs prediction after lock, steps PyBullet arm.
  3. **Composite thread** — builds the browser frame (webcam + overlays + PyBullet render + probability bars) and writes to shared buffer.
  4. **Flask thread** — serves MJPEG from shared buffer.
- Draw overlays on the webcam frame (OpenCV):
  - Circle at hand position
  - Target region rectangles (red, blue, green)
  - Predicted trajectory (dashed line after lock)
  - `xf_predicted` star marker
  - Status text: mode, locked target, confidence, lock time, D estimate, prediction error
- Composite the PyBullet render (right panel) and probability bars alongside the webcam frame.
- Handle graceful cases: no hand detected, tracking loss, trial reset (`r` key), screenshot save (`s` key), quit (`q` key).

Acceptance criteria:

- `python experiments/run_webcam_demo.py` starts the Flask server and prints `→ Open http://localhost:5000 in your browser`.
- Opening that URL shows the full composite: webcam feed + PyBullet arm + probability bars in one window.
- The tracked hand position updates in real time.
- Target probabilities change visibly as the hand moves.
- A target locks when a probability passes `0.8` — shown in the status overlay.
- The PyBullet arm begins moving after target lock, before the hand stops.
- Pressing `r` resets the trial. Pressing `s` saves a screenshot. Pressing `q` shuts down cleanly.

### Stage 9: Multiple Real-Time Trials and Summary Results

Goal: produce material suitable for the report, video, and interview.

Tasks:

- Run multiple webcam trials toward red, blue, and green targets.
- Record target-selection accuracy.
- Record mean prediction error.
- Record average lock time.
- Save summary plots.

Acceptance criteria:

- At least one results summary is printed.
- Summary plots are saved.
- Logs and figures can be used directly in the report.
- The final evidence is based on webcam trials, not only simulation.

### Stage 10: Report/Video/Interview Readiness Pass

Goal: make sure the assessment requirements are easy to demonstrate.

Tasks:

- Update README with how to run the webcam demo.
- Add screenshots showing perception, inference, target lock, robot action, and final error.
- Add a short `docs/demo_script.md` or README section for the 10-minute video.
- Add a short `docs/interview_notes.md` explaining the workflow and key implementation decisions.

Acceptance criteria:

- The report marking items are explicitly covered.
- The video marking items are explicitly supported.
- The interview/viva questions are easy to answer from the code and notes.

---

## 9. Testing Requirements

Add lightweight tests as the project grows.

### Bayesian Tests

- Posterior probabilities sum to 1.
- The correct target probability increases for a clean trajectory toward that target.
- Target lock happens when a posterior exceeds the threshold.

### Minimum-Jerk Tests

- Generated trajectory starts at `x0`.
- Generated trajectory ends at `xf`.
- No NaN values are produced.
- The generated trajectory has the expected number of points.

### Metrics Tests

- Prediction error is zero when predicted and actual endpoints are equal.
- Prediction error is positive when endpoints differ.
- Target correctness is computed correctly.

### Webcam Robustness Checks

These can be manual checks if automated testing is difficult:

- No crash when no hand is detected.
- No crash when the hand temporarily leaves the frame.
- Reset key starts a new trial.
- Save key writes a screenshot/log.
- Demo can close cleanly.

---

## 10. Logging Requirements

Each trial should save:

```text
trial_id
mode
started_at
ground_truth_target_or_user_selected_target
predicted_target
target_correct
lock_time
lock_confidence
xf_predicted
xf_actual
prediction_error_vector
prediction_error_norm
D_estimated
D_adapted
num_frames
notes
```

Save logs as CSV or JSON under:

```text
results/logs/
```

For webcam trials, allow the user to manually indicate the intended target if needed. This makes evaluation possible without physically detecting object colors.

---

## 11. Required Output Files

By the final stage, the project should generate some or all of these:

```text
results/figures/static_scene.png
results/figures/scene_trajectory.png
results/figures/posterior_probabilities.png
results/figures/interception_prediction.png       ← includes PyBullet arm render
results/figures/prediction_error.png
results/figures/summary_metrics.png
results/screenshots/browser_demo_lock.png         ← full browser composite at lock moment
results/screenshots/browser_demo_final.png        ← full browser composite at trial end
results/animations/hri_loop_animation.gif         ← optional
results/logs/trials.csv
```

Screenshots must capture the full browser composite frame (webcam + PyBullet + bars), not just the webcam crop. Webcam screenshots and trial logs are required; the animation is optional.

---

## 12. README Requirements

The README must explain:

- project goal
- real-time perception route and why webcam mode is required for the final project
- assignment requirements
- HRI loop
- Bayesian inference step
- minimum-jerk prediction step
- target lock threshold of `0.8`
- how to run the webcam demo
- how to run the simulation/debug mode
- where outputs are saved
- current limitations
- what to discuss in the report/video/interview

Include simple commands such as:

```bash
pip install -r requirements.txt

# Live webcam demo (opens browser at localhost:5000)
python experiments/run_webcam_demo.py
# then open http://localhost:5000 in your browser

# Simulation / debug mode (no webcam needed)
python experiments/run_simulation.py

# Multi-trial evaluation runner
python experiments/run_trials.py
```

---

## 13. Final Definition of Done

The project is complete when:

- The repository can be created from scratch and run with clear commands.
- The final demo uses webcam-based real-time hand/wrist/fingertip tracking.
- Three target regions/objects are visible: red, blue, and green.
- The system continuously updates Bayesian probabilities for red, blue, and green targets.
- The target locks when `P(G | O) > 0.8`.
- The system predicts an interception/final point using the minimum-jerk model.
- A PyBullet-simulated robot arm reacts proactively after target lock, positioned by inverse kinematics.
- The system computes and logs prediction error.
- The system runs multiple real-time trials and reports accuracy, prediction error, and lock time.
- The project saves clean webcam screenshots, figures, and logs for presentation.
- Simulation exists only as a development/debug mode, not as the final evidence.
- README/report notes explicitly map the implementation to the assignment requirements and marking scheme.

---

## 14. Common Mistakes to Avoid

- Do not finish with only simulated perception.
- Do not mention or implement an AR version.
- Do not skip Bayesian normalization.
- Do not hard-code only one object.
- Do not hide the assignment equations from the final code/report.
- Do not over-complicate the PyBullet arm. A 2-DOF planar arm is sufficient and fully explainable.
- Do not mix all logic into one giant script. Keep webcam, inference, robot, and server threads in separate modules.
- Do not make overlays unreadable with too many lines or labels.
- Do not forget thread safety. Always use `threading.Lock` when writing or reading the shared composite frame buffer.
- Do not rely on advanced tools without being able to explain them in the viva. Be ready to explain PyBullet IK, the MJPEG stream, and the threading model.

---

## 15. Implementation Style

When editing code:

- Make small focused changes.
- Keep each stage runnable.
- Add only useful comments.
- Make outputs easy to explain in a course presentation.
- Keep file names and module names simple.
- Let implementation details evolve naturally instead of forcing rigid interfaces too early.
- Prefer understandable code that can be defended in an interview over clever code.
