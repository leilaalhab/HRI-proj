# CLAUDE.md

This file is the implementation guide for Claude Code. The repository starts empty, so build the project from scratch in small, working stages. Each stage must leave the project runnable.

Project title: **Real-Time Human Intent Recognition for Proactive Robot Handover**

The final project must be a **real-time perception HRI project** using a **webcam**. This matters because the assignment states that real-time perception projects can receive the full project mark, while simulated perception projects are capped lower. Simulation may be used only as an early development/debugging tool. The final demo, report, video, and interview must clearly show real webcam-based perception driving the HRI loop.

The goal is to build a Python-only Human-Robot Interaction demo. The system observes a human hand or wrist moving toward one of several visible candidate objects, infers the intended object using Bayesian goal recognition, predicts the final hand/interception point using a minimum-jerk trajectory model, and visualizes a proactive virtual robot/gripper response.

The final project should match the assignment concept:

> The robot must first figure out which object the human wants, then calculate where and when the human hand will stop so the robot can proactively place or prepare the object at the predicted interception point.

Do not implement everything at once. Build from a minimal runnable project to a complete webcam-based demo.

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
- `opencv-python` for webcam capture, drawing, and display windows
- `mediapipe` for webcam hand/wrist/fingertip tracking
- `matplotlib` for saved plots and optional offline figures
- `pandas` for saving trial logs
- `pytest` for lightweight tests

Do not create a Unity version. Do not create a web/JavaScript version. Do not mention or implement an AR version. Keep the whole project understandable as a Python course demo.

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
│   │   └── virtual_robot.py
│   │
│   ├── visualization/
│   │   ├── __init__.py
│   │   ├── live_dashboard.py
│   │   └── plots.py
│   │
│   └── evaluation/
│       ├── __init__.py
│       └── metrics.py
│
├── experiments/
│   ├── run_simulation.py
│   ├── run_webcam_demo.py
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
- Do not over-engineer the robot. Use a virtual robot marker, gripper icon, or simple arm line.
- Keep configuration values in `config.py`.
- Save generated plots, screenshots, logs, and optional animations in `results/`.
- Use readable names such as `posterior`, `goal`, `observation`, `target`, `trajectory`, `xf_predicted`, `xf_actual`, `D`, `tau`, and `prediction_error`.
- Avoid hard-coding a single target.
- Normalize posterior probabilities after every Bayesian update.
- Prefer simple working code over complex abstractions.
- Do not give the final user only a simulated demo, because that would not satisfy the intended real-time perception route.

---

## 6. Simple GUI / Visualization Decisions

Start with simple visual output, then build a clean webcam demo dashboard.

Recommended GUI decisions:

- Use an OpenCV live window for the main real-time demo.
- Draw the webcam feed as the background.
- Overlay the tracked wrist/fingertip position.
- Draw three target regions for red, blue, and green.
- Draw live probability bars for the three targets on one side of the frame.
- Display status text: current mode, locked target, confidence, lock time, estimated `D`, and prediction error.
- Use consistent colors: red target is red, blue target is blue, green target is green.
- Use simple symbols:
  - circle for the detected hand/wrist/fingertip
  - square/rectangle for target regions
  - star/cross for `xf_predicted`
  - dashed or lighter path for predicted trajectory if easy to draw
  - solid path for observed trajectory
  - arrow/line for virtual robot action
- Keep labels large enough to be visible in the project video.
- Avoid clutter. The viewer should immediately understand: hand moves, probabilities change, target locks, robot reacts.
- Save screenshots from the live demo for the report.
- Use `matplotlib` only for offline plots such as probability history, prediction error, and summary metrics.

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

Use simple cues:

- distance from hand to goal
- direction of velocity toward goal

A good simple likelihood is:

```text
likelihood = distance_likelihood * direction_likelihood
```

where:

```text
distance_likelihood = exp(-distance / distance_scale)
direction_likelihood = exp(direction_weight * cosine_similarity)
```

The exact implementation can evolve naturally, but it must remain easy to explain in the interview.

### 7.5 Minimum-Jerk Prediction

Use the minimum-jerk equation to generate smooth movement from the current hand/robot position to the predicted final point.

For the first working version:

- After target lock, set `xf_predicted` near the locked target position or target region center.
- Estimate duration `D` from remaining distance and current hand speed.
- Generate a smooth predicted trajectory using the minimum-jerk profile.

Later, improve this by fitting recent observations if needed. Keep the final explanation simple.

### 7.6 Virtual Robot Action

The virtual robot does not need physical robot kinematics.

A good first version is:

- a fixed robot base point drawn in the window
- a moving gripper marker
- a line or arrow from the robot base to the predicted interception point
- smooth movement after target lock

The action should visually show that the robot reacts before the human fully reaches the object.

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
DISTANCE_SCALE = 1.0
DIRECTION_WEIGHT = 2.0
ADAPTATION_GAIN = 0.1
WEBCAM_INDEX = 0
SMOOTHING_ALPHA = 0.35
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

### Stage 6: Virtual Robot Action

Goal: visualize proactive robot behavior using a PyBullet-simulated robot arm.

Tasks:

- Set up a PyBullet simulation with a simple robot arm (URDF or programmatically defined).
- After target lock, compute inverse kinematics to position the end effector at `xf_predicted`.
- Command the robot arm to move smoothly to the computed joint configuration.
- Render the PyBullet simulation alongside the main demo (separate window or composited into the OpenCV frame).
- Use minimum-jerk joint interpolation for smooth arm movement after lock.
- Save a clear final plot or animation showing the robot response.

Acceptance criteria:

- A PyBullet simulation window shows the robot arm responding after target lock.
- The robot begins moving after target lock, not only after the hand fully stops.
- Inverse kinematics positions the end effector at or near `xf_predicted`.
- The robot response is understandable for the video and interview.

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

### Stage 8: Webcam Perception Integration

Goal: make the project a real-time perception project.

Tasks:

- Add webcam frame capture.
- Add hand/wrist/fingertip tracking.
- Convert tracked image coordinates into scene coordinates.
- Estimate velocity from recent webcam positions.
- Feed the live observation stream into Bayesian inference.
- Overlay targets, probabilities, target lock, prediction, and robot response on the live frame.

Acceptance criteria:

- `python experiments/run_webcam_demo.py` opens the webcam demo.
- The webcam hand/wrist/fingertip position is visible.
- Target probabilities change in real time.
- A target locks when a probability passes `0.8`.
- The virtual robot response is shown after target lock.
- The demo can reset trials and save logs/screenshots.

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
results/figures/interception_prediction.png
results/figures/prediction_error.png
results/figures/summary_metrics.png
results/screenshots/webcam_demo_lock.png
results/screenshots/webcam_demo_final.png
results/animations/hri_loop_animation.gif
results/logs/trials.csv
```

The animation is optional, but webcam screenshots, figures, and logs should exist.

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
python experiments/run_webcam_demo.py
python experiments/run_simulation.py
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
- A virtual robot/gripper/marker reacts proactively after target lock.
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
- Do not overcomplicate the robot model.
- Do not mix all logic into one giant script.
- Do not make plots or overlays unreadable with too many lines or labels.
- Do not claim physical robot control; this is a visualization/demo unless real hardware is explicitly added later.
- Do not rely on advanced tools without being able to explain them in the viva.

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
