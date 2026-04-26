# Real-Time Human Intent Recognition for Proactive Robot Handover

A Python-only Human-Robot Interaction demo that observes a human hand moving toward one of three coloured target blocks, infers the intended target using Bayesian goal recognition, predicts the final hand position using a minimum-jerk trajectory model, and shows a proactive virtual robot arm response — all in real time via webcam.

---

## Real-Time Perception Route

This project follows the **real-time perception route**. The final demo uses live webcam-based hand/wrist tracking via MediaPipe. Simulation is implemented only as a development and debugging tool and is not presented as the final outcome.

The final evidence for the report, video, and interview must come from webcam trials run with `run_webcam_demo.py`.

---

## Assignment Requirements

| Requirement | Where implemented |
|---|---|
| Webcam-based real-time perception | `src/perception/webcam_perception.py` |
| Bayesian intent inference | `src/inference/bayesian_goal_inference.py` |
| Target lock at P > 0.8 | `config.CONFIDENCE_THRESHOLD`, `BayesianGoalInference.update()` |
| Minimum-jerk trajectory prediction | `src/prediction/minimum_jerk.py` |
| Proactive robot arm response | `src/robot/pybullet_robot.py` (PyBullet, IK, joint stepping) |
| Prediction error measurement | `src/evaluation/metrics.py`, `build_trial_result()` |
| Feedback and adaptation | `DurationAdapter` in `src/evaluation/metrics.py` |
| Trial logging | `results/logs/trials.csv` |
| Summary figures | `results/figures/` |

---

## HRI Loop

```
Perception → Inference → Decision → Robot Action → Feedback → Adaptation
```

1. **Perception** — webcam frame captured by OpenCV, MediaPipe Hands extracts wrist pixel position, exponential smoothing applied, velocity estimated from consecutive positions
2. **Inference** — Bayesian posterior updated for each of the three targets: `P(G | O) ∝ P(O | G) * P(G)`
3. **Decision** — target locked when `P(G | O) > 0.8` and two guard conditions are met (minimum travel distance, confirmation window)
4. **Robot Action** — PyBullet IK computes joint angles for the locked target; arm interpolates from home to target using minimum-jerk joint trajectory, arriving slightly before the hand
5. **Feedback** — `xf_predicted` (locked target centre) compared to `xf_actual` (hand final position); Euclidean error logged
6. **Adaptation** — `DurationAdapter` updates a correction factor based on the ratio `D_actual / D_estimated`; applied to the next trial's robot timing

---

## Bayesian Inference

At every time step, for each goal G and current observation O:

```
P(G | O_1:t) ∝ P(O_t | G) * P(G | O_1:t-1)
```

The likelihood combines two cues:

```
likelihood(G) = off_axis_likelihood(G) * direction_likelihood(G)

off_axis_likelihood  = exp( -off_axis_distance / DISTANCE_SCALE )
direction_likelihood = exp( DIRECTION_WEIGHT * cosine_similarity(velocity, direction_to_G) )
```

**Off-axis distance** is the perpendicular distance from the hand to the straight line between the trial start position and goal G (normalised to [0, 1]). It discriminates from the first frame of movement — a hand moving toward green immediately has near-zero off-axis deviation for green and large deviation for red and blue.

**Direction likelihood** uses cosine similarity between hand velocity and the direction toward G. It is neutral (1.0) when hand speed is below 5 px/s.

Posteriors are normalised after every update so they sum to 1.

### Lock guard conditions

To prevent false locks from early noise:

- `LOCK_MIN_TRAVEL_PX = 200` — cumulative step distance must exceed 200 px before any lock is considered
- `LOCK_CONFIRM_FRAMES = 10` — the same target must hold P > 0.8 for 10 consecutive frames (~0.33 s) before the lock is confirmed; the streak resets if probability dips below the threshold

---

## Minimum-Jerk Prediction

The minimum-jerk trajectory equation generates smooth, human-like movement:

```
x(t) = x0 + (xf - x0) * (10τ³ - 15τ⁴ + 6τ⁵),   τ = t / D
```

After target lock, `xf_predicted` is set to the locked target centre.

**Duration estimation using τ back-calculation** (avoids naive `D = distance / speed`, which overestimates badly at early lock times):

1. Compute `progress = |x_current − x0| / |xf − x0|`
2. Solve `s(τ) = progress` for τ using `np.roots()` on `10τ³ − 15τ⁴ + 6τ⁵ − progress = 0`
3. Evaluate the normalised speed profile: `g(τ) = 30τ² − 60τ³ + 30τ⁴`
4. Back-calculate: `D_total = total_dist * g(τ) / current_speed`
5. Remaining duration: `D_remaining = D_total * (1 − τ)`

The robot arm is given `D_robot = D_remaining * ROBOT_D_SCALE` (default 0.85) so it arrives slightly before the hand — making proactiveness visible.

---

## PyBullet Robot Arm

A 2-DOF planar revolute-joint arm is simulated in PyBullet `DIRECT` mode (no GUI window). Analytical inverse kinematics:

```
cos(θ₂) = (r² − L1² − L2²) / (2·L1·L2)
θ₁ = atan2(−x, y) − atan2(L2·sin(θ₂), L1 + L2·cos(θ₂))
```

where L1 = 1.2 m, L2 = 1.0 m, and r = distance from base to target.

**State machine per trial:**
- `IDLE` — arm at home position, waiting for target lock
- `MOVING` — joint angles interpolate toward IK solution over `D_robot` seconds using minimum-jerk profile
- `HOLDING` — arm held at target, waiting for trial reset

The arm render is a 2D OpenCV diagram drawn from current joint angles, composited into the browser frame as the right panel.

---

## How to Run

**Python interpreter:** always use `$HOME/miniforge3/envs/hri/bin/python`

```bash
pip install -r requirements.txt
```

### Live webcam demo (final real-time mode)

```bash
$HOME/miniforge3/envs/hri/bin/python experiments/run_webcam_demo.py
```

Then open **http://localhost:5001** in a browser.

The demo runs a guided 9-trial plan (red → blue → green × 3, interleaved). The overlay shows which target to reach next. Press **r** (Reset) after each trial to advance. After all 9 trials, a summary is printed and figures are saved automatically.

**Browser controls:**

| Button | Terminal key | Action |
|---|---|---|
| ↺ Reset trial | `r` + Enter | Log trial, advance plan |
| 💾 Screenshot | `s` + Enter | Save composite frame to `results/screenshots/` |
| ■ Quit | `q` + Enter | Clean shutdown |

### Simulation / debug mode (no webcam required)

```bash
$HOME/miniforge3/envs/hri/bin/python experiments/run_simulation.py
```

Runs a single simulated trial toward the green target. Saves trajectory and interception figures. Also starts the Flask server so the simulation scene can be viewed in the browser.

### Multi-trial simulation evaluation

```bash
$HOME/miniforge3/envs/hri/bin/python experiments/run_trials.py
```

Runs 9 simulated trials (3 per target, interleaved) with Gaussian noise. Prints a results table and saves prediction error and summary figures.

### Tests

```bash
$HOME/miniforge3/envs/hri/bin/python -m pytest tests/ -q
```

20 tests covering Bayesian inference, minimum-jerk math, and metrics computation.

---

## Browser Interface

The live demo is served as an MJPEG stream at `localhost:5001`. The composite frame has three panels:

```
┌───────────────────────────────────┬──────────────┬──────────────┐
│  Webcam feed (640 × 480)          │  PyBullet    │  Probability │
│  + target rectangles              │  robot arm   │  bars        │
│  + trial instruction banner       │  render      │  R / B / G   │
│  + hand circle (cyan → green)     │  (400 × 400) │              │
│  + predicted trajectory (dashed)  │              │              │
│  + xf_predicted star              │              │              │
│  + status text overlay            │              │              │
└───────────────────────────────────┴──────────────┴──────────────┘
```

- Hand circle is **cyan** when no lock, turns **green** after lock
- Probability bars update in real time; the locked target's bar is dominant
- Robot arm begins moving after lock and reaches the target before the hand

---

## Generated Outputs

### Figures — `results/figures/`

| File | Generated by | Description |
|---|---|---|
| `static_scene.png` | `run_simulation.py` | Top-down view of the 640×480 canvas with the three coloured target rectangles (red left, blue centre, green right) and the hand start marker. Used in the report to introduce the scene layout. |
| `scene_trajectory.png` | `run_simulation.py` | Two-panel figure. Left: the scene with the observed simulated hand trajectory overlaid in the ground-truth target colour, the lock point marked with an orange diamond, the predicted remaining trajectory in dashed grey, and the `xf_predicted` star. Right: the hand speed profile over time showing the characteristic minimum-jerk bell curve (slow → fast → slow). |
| `posterior_probabilities.png` | `run_simulation.py` | Posterior probability P(G \| O) over time for all three targets, plotted as coloured lines. A horizontal dashed line marks the 0.8 lock threshold. A vertical annotation marks the moment the target locks with the confidence value. Shows how the correct target's probability rises while the others fall. |
| `interception_prediction.png` | `run_simulation.py` | Two-panel figure. Left: scene + trajectory + predicted endpoint star. Right: PyBullet robot arm rendered at its IK-computed joint configuration for the locked target, with the end-effector positioned toward the predicted interception point. Shows proactive robot response. |
| `prediction_error.png` | `run_trials.py` | Bar chart of `‖xf_pred − xf_actual‖` (px) for each of the 9 simulation trials, coloured by ground-truth target (red/blue/green). Twin Y-axis (orange) shows `D_correction_factor` evolving across trials as the duration adapter learns. |
| `summary_metrics.png` | `run_trials.py` | Three-panel simulation summary. Left: stacked bar of correct vs wrong predictions per target class. Centre: histogram of prediction errors with mean line. Right: `D_correction_factor` trajectory across trials. Title shows trial count, accuracy, and mean error. |
| `prediction_error_webcam.png` | `run_webcam_demo.py` (after plan complete) | Same format as `prediction_error.png` but from real webcam trials. Each bar represents one of the 9 guided plan trials. Provides the report's primary evaluation evidence. |
| `summary_metrics_webcam.png` | `run_webcam_demo.py` (after plan complete) | Same three-panel format as `summary_metrics.png` but labelled "Stage 9 — Webcam". Title auto-derives accuracy and mean error from the actual webcam results. |

### Screenshots — `results/screenshots/`

| File | Description |
|---|---|
| `webcam_TIMESTAMP.png` | Full browser composite frame (webcam + PyBullet arm + probability bars) saved when the user presses **s** or the ↺ Screenshot button. Capture at lock moment and at trial end for the report. One screenshot is auto-saved when the 9-trial plan completes. |

### Logs — `results/logs/`

| File | Generated by | Description |
|---|---|
| `trials.csv` | All experiment scripts | Per-trial CSV log. One row per completed trial. Appended to across runs. |

**`trials.csv` columns:**

| Column | Description |
|---|---|
| `trial_id` | Sequential trial number |
| `mode` | `webcam` or `simulation` |
| `started_at` | ISO timestamp when the trial was logged |
| `ground_truth_target` | Target the user was told to reach (guided plan) or the simulation ground truth |
| `predicted_target` | Target the system locked onto |
| `target_correct` | `True` / `False` |
| `lock_time` | Time (s from trial start) when P > 0.8 was confirmed |
| `lock_confidence` | Posterior probability at lock moment |
| `xf_predicted_x`, `xf_predicted_y` | Predicted final hand position (px) |
| `xf_actual_x`, `xf_actual_y` | Actual final hand position at trial end (px) |
| `prediction_error_x`, `prediction_error_y` | Signed component errors (px) |
| `prediction_error_norm` | `‖xf_predicted − xf_actual‖` Euclidean error (px) |
| `D_estimated` | Duration estimate from τ back-calculation (s) |
| `D_actual` | Time from lock to trial end (s) |
| `D_adapted` | `D_estimated * correction_factor` applied to robot timing |
| `num_frames` | Number of observations processed in this trial |
| `notes` | Trial mode and plan position, e.g. `webcam guided 3/9` |

---

## Repository Structure

```
project-root/
├── config.py                        — all tuning parameters
├── requirements.txt
│
├── src/
│   ├── scene/targets.py             — Target dataclass, canvas size, 3 target regions
│   ├── perception/
│   │   ├── webcam_perception.py     — MediaPipe wrist tracker, smoothing, velocity
│   │   └── simulated_perception.py  — noise-added minimum-jerk trajectory generator
│   ├── inference/
│   │   └── bayesian_goal_inference.py — Bayesian update, lock guard, confirmation window
│   ├── prediction/
│   │   └── minimum_jerk.py          — trajectory generation, τ back-calculation, D estimation
│   ├── robot/
│   │   └── pybullet_robot.py        — 2-DOF arm, analytical IK, joint stepping, render
│   ├── server/
│   │   └── stream_server.py         — Flask MJPEG server, composite frame builder
│   ├── visualization/
│   │   └── plots.py                 — offline matplotlib figures only
│   └── evaluation/
│       └── metrics.py               — TrialResult, DurationAdapter, TrialLogger
│
├── experiments/
│   ├── run_webcam_demo.py           — Stage 8/9: live demo + guided 9-trial plan
│   ├── run_simulation.py            — Stage 6: single simulated trial + Flask view
│   └── run_trials.py                — Stage 7: 9-trial simulation evaluation
│
├── static/
│   └── webcam.html                  — browser UI (img + 3 control buttons)
│
├── assets/
│   └── planar_arm.urdf              — minimal 2-DOF URDF for PyBullet
│
├── tests/
│   ├── test_bayesian_goal_inference.py
│   ├── test_minimum_jerk.py
│   └── test_metrics.py
│
└── results/
    ├── figures/                     — all matplotlib figures
    ├── screenshots/                 — browser composite snapshots
    └── logs/trials.csv              — per-trial evaluation log
```

---

## Configuration Parameters

All tuning values are in `config.py`:

| Parameter | Value | Description |
|---|---|---|
| `CONFIDENCE_THRESHOLD` | 0.8 | Lock fires when P(G\|O) exceeds this |
| `DISTANCE_SCALE` | 0.25 | Off-axis distance scale (normalised [0,1]); larger = softer spatial penalty |
| `DIRECTION_WEIGHT` | 0.8 | Direction likelihood weight; reduced from 2.0 for 30 fps webcam stability |
| `LOCK_MIN_TRAVEL_PX` | 200 | Cumulative hand travel (px) required before lock is considered |
| `LOCK_CONFIRM_FRAMES` | 10 | Consecutive frames above threshold required to confirm lock |
| `SMOOTHING_ALPHA` | 0.35 | Exponential smoothing coefficient for webcam wrist position |
| `SIMULATION_DT` | 0.05 s | Time step for simulated trajectories and robot joint stepping |
| `ADAPTATION_GAIN` | 0.1 | EMA gain for DurationAdapter correction factor |
| `ROBOT_D_SCALE` | 0.85 | Robot arm duration = `D_remaining * ROBOT_D_SCALE`; arrives before hand |
| `FLASK_PORT` | 5001 | Browser port; open `http://localhost:5001` |
| `WEBCAM_INDEX` | 0 | OpenCV camera index |
| `PYBULLET_RENDER_W/H` | 400 × 400 | Pixel size of robot arm panel in composite |

---

## Limitations

- **Fixed target regions** — targets are drawn as screen rectangles, not physically detected by colour. The user must move their hand toward the on-screen boxes.
- **2-DOF planar arm** — the PyBullet arm operates in a simplified 2D plane; no 3D manipulation or collision detection.
- **No depth perception** — the webcam provides 2D pixel coordinates only; hand depth is not estimated.
- **Minimum-jerk assumption** — the prediction model assumes humans move in smooth minimum-jerk arcs. Hesitations, direction changes, or pauses mid-motion can cause the lock to fire on the wrong target.
- **Single hand** — MediaPipe is configured for one hand; multi-hand or occluded-hand scenarios are not handled.
- **No physical robot** — the system is a software simulation only; the robot arm does not control real hardware.

---

## What to Show in the Report / Video

- The 6-step HRI loop running live (webcam → inference bars updating → lock event → arm moving → error logged)
- The Bayesian probability bars separating gradually as the hand commits to a direction
- The lock happening visibly before the hand reaches the target, with the arm already moving
- `prediction_error_webcam.png` and `summary_metrics_webcam.png` as quantitative results
- `trials.csv` as evidence of logged multi-trial evaluation
- Screenshots captured at the lock moment and at trial end
