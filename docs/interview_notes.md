# Interview / Viva Notes

Everything you need to answer questions about this codebase confidently. Covers the logic behind every major decision, the math explained in plain terms, and how to talk about the evaluation.

---

## 1. What is the project and why does it matter?

The project is a **proactive robot handover system**. The idea is that instead of waiting for a human to explicitly ask for something, a robot should be able to predict what the human wants from their movement and prepare the object in advance.

In this demo: a person moves their hand toward one of three coloured target blocks (red, blue, green). The system watches the movement through a webcam, figures out which block they're going for, predicts where their hand will stop, and moves a virtual robot arm toward that point — all before the hand actually arrives.

The word "proactive" is key. The robot does not wait for the hand to reach the block. It moves first, based on intent inference. This is what separates it from a simple reactive system.

---

## 2. What is the full pipeline, step by step?

```
Webcam frame
    → MediaPipe extracts wrist position
    → Exponential smoothing reduces noise
    → Velocity estimated from consecutive frames
    → Bayesian update for each of 3 targets
    → Lock fires when P > 0.8 (with guards)
    → Minimum-jerk model estimates remaining time D
    → PyBullet IK computes joint angles for target
    → Robot arm interpolates to target over D × 0.85 seconds
    → At trial end: error = xf_predicted − xf_actual logged
    → D correction factor updated for next trial
```

Each step feeds the next. The perception module produces observations. The inference module consumes them. The prediction module fires once at lock. The robot module runs continuously in its own thread. All outputs flow through a shared state object protected by a threading lock.

---

## 3. How does the webcam perception work?

**File:** `src/perception/webcam_perception.py`

OpenCV captures raw frames at ~30 fps. Each frame is flipped horizontally (mirror mode, so left/right match the display) and resized to 640×480.

MediaPipe Hands runs on every frame. It detects 21 hand landmarks. We use **landmark 0, the wrist**, not a fingertip. The wrist was chosen because it is more stable under partial occlusion — if the fingertips leave frame or are curled, the wrist is still tracked reliably.

The raw wrist position is noisy. We apply **exponential smoothing**:

```
smooth_pos = α * raw_pos + (1 − α) * smooth_pos_prev
```

With `α = 0.35` (SMOOTHING_ALPHA). A lower α gives smoother but more lagged position. 0.35 is a trade-off that tracks fast movement without being too jittery.

Velocity is estimated as:

```
velocity = (smooth_pos_current − smooth_pos_prev) / dt
```

where dt is the real wall-clock time between frames. This is important — we use actual elapsed time, not assumed 1/30 s, so the velocity estimate is correct even if frames arrive irregularly.

**Design decision:** we could have used a Kalman filter instead of exponential smoothing. Kalman would also estimate velocity optimally. We chose exponential smoothing because it is simpler, has one tunable parameter, and is easier to explain in an interview.

---

## 4. How does the Bayesian inference work?

**File:** `src/inference/bayesian_goal_inference.py`

At every time step we update a probability distribution over the three goals {red, blue, green}. The update rule is:

```
P(G | O_1:t) ∝ P(O_t | G) * P(G | O_1:t-1)
```

In plain terms: the new belief about goal G equals the likelihood of the current observation given that goal, multiplied by the previous belief. This is standard Bayes' theorem applied sequentially.

The prior starts uniform: P(red) = P(blue) = P(green) = 1/3.

After computing unnormalized values for all three goals, we normalize so they sum to 1.

### The likelihood function

The likelihood `P(O_t | G)` is the tricky part. We can't directly observe intent, so we design a likelihood based on what we can observe: hand position and velocity.

We use two cues multiplied together:

```
likelihood(G) = off_axis_likelihood(G) × direction_likelihood(G)
```

**Off-axis likelihood:**

The hand's path from start to goal G should be roughly a straight line. We compute the perpendicular distance from the current hand position to the line connecting the trial start point and goal G. If the hand is on the line toward G, off-axis distance is zero and likelihood is 1.0. If the hand is far off that line, likelihood drops exponentially:

```
off_axis_likelihood = exp(−off_axis_distance / DISTANCE_SCALE)
```

Coordinates are normalised to [0, 1] before this computation so DISTANCE_SCALE is dimensionless. With DISTANCE_SCALE = 0.25, a 25% off-axis deviation gives likelihood ≈ 0.37.

**Why off-axis instead of raw distance to goal?**

Raw distance to goal is nearly the same for all three targets when the hand is near the start — you haven't moved toward any of them yet, so they're all roughly equidistant. Off-axis distance discriminates immediately: the moment you take even one step, the angle you're moving at tells you something about which goal you're heading toward.

**Direction likelihood:**

We also look at the velocity vector. If you're moving toward goal G, your velocity should point roughly in the direction of G. We compute cosine similarity between the hand velocity and the direction from hand to goal G:

```
direction_likelihood = exp(DIRECTION_WEIGHT × cosine_similarity(velocity, direction_to_G))
```

Cosine similarity is 1.0 when the velocity points directly at G, 0 when perpendicular, −1 when pointing away. The exponential means pointing directly at G gives the highest likelihood. DIRECTION_WEIGHT = 0.8 controls how sharply this discriminates.

When hand speed is below 5 px/s (stationary or barely moving), direction is meaningless — you can't infer intent from near-zero velocity. In this case direction_likelihood is set to 1.0 (neutral) and only off-axis matters.

**Design decision on DIRECTION_WEIGHT:** the original value was 2.0 (from the simulation tuning). At 30 fps webcam, with 2.0, the probability bars would jump from 33% to 80%+ in under 0.2 seconds — visually binary. We reduced it to 0.8 to give gradual, visible probability evolution. The core insight is that the same per-step likelihood compounds much faster at 30 fps than at the 20 fps the simulation was designed for.

### Goal directions from actual start position

A subtle but important detail: we don't compute goal directions from the fixed HAND_START constant. We compute them from wherever the user's hand actually is at the start of each trial. This matters because in webcam mode the user's hand might start anywhere. If we anchored from the wrong start position, the off-axis distances would be systematically wrong and inference would be inaccurate.

The inference thread waits for the first observation of each trial and calls `inference.reset(start_pos=first_obs.position)`, which recomputes all goal direction vectors from that actual starting position.

---

## 5. Why P > 0.8 as the lock threshold?

0.8 is a principled choice that balances two risks:

- Too low (e.g. 0.6): locks early before the hand has clearly committed to a direction → too many wrong predictions, robot moves preemptively to the wrong target
- Too high (e.g. 0.95): never locks until the hand is almost at the target → no longer proactive, the robot arrives too late to be useful

0.8 means we're 80% confident before committing. In a 3-target system starting from uniform (33% each), reaching 80% means one target has accumulated roughly 4× more evidence than each of the other two. That represents a clear commitment to a direction.

### The two additional lock guards

We added two guards on top of the 0.8 threshold because the threshold alone was not enough to prevent false locks:

**LOCK_MIN_TRAVEL_PX = 200:** The hand must have traveled at least 200 px cumulatively before any lock is considered. The target regions are roughly 240 px from the start position, so 200 px means the hand is most of the way there before a lock can fire. This prevents locking on the very first frames of movement when velocity is noisy and direction information is unreliable.

Note: we track cumulative step-by-step distance, not straight-line distance from start. This handles cases where the user starts from a different position each trial.

**LOCK_CONFIRM_FRAMES = 10:** Once the travel guard is cleared, the same target must hold P > 0.8 for 10 consecutive frames (about 0.33 seconds at 30 fps) before the lock is confirmed. If probability dips below 0.8 at any point, the streak resets. This prevents a momentary velocity spike from causing a premature lock — the system requires sustained high confidence, not a brief peak.

**Design decision:** We arrived at these values empirically. With only the 0.8 threshold and no guards, we saw false locks on simulation trials with noise_std=3.0 because noisy early velocity estimates momentarily pointed toward the wrong target. The guards fixed this completely (9/9 accuracy vs 5/9 before).

---

## 6. How does the minimum-jerk prediction work?

**File:** `src/prediction/minimum_jerk.py`

The minimum-jerk model is a well-established model of human arm movement from motor neuroscience. It states that humans move in a way that minimises the jerk (rate of change of acceleration). The resulting trajectory has a specific shape:

```
x(t) = x0 + (xf − x0) × (10τ³ − 15τ⁴ + 6τ⁵),   τ = t/D
```

where x0 is the start position, xf is the final position, D is total duration, and τ is normalised time [0, 1]. The speed profile is bell-shaped: slow at start, fast in the middle, slow at the end.

### The τ back-calculation for D estimation

The naive approach to estimating remaining time is `D = remaining_distance / current_speed`. This fails badly when the lock happens early in the movement, because the hand is still slow (early in the bell curve). A hand that will travel 200 px in 1.5 seconds is only moving at ~20 px/s at t=0.3s, not the 133 px/s average. So `remaining_distance / current_speed` would give ~8 seconds — wildly wrong.

Instead we exploit the minimum-jerk model to figure out where we are in the movement:

1. Compute `progress = |x_current − x0| / |xf − x0|` — how far through the movement we are
2. Solve `s(τ) = progress` where `s(τ) = 10τ³ − 15τ⁴ + 6τ⁵` — this finds τ, our position in normalised time. We use `np.roots()` on the polynomial.
3. Evaluate the normalised speed at that τ: `g(τ) = 30τ² − 60τ³ + 30τ⁴`
4. Back-calculate: `D_total = total_distance × g(τ) / current_speed`
5. Remaining duration: `D_remaining = D_total × (1 − τ)`

This is correct because the minimum-jerk model tells us exactly what speed we should have at position τ. If the actual speed matches the model, we can work backwards to D.

### Why xf_predicted = locked target centre?

We set xf_predicted to the centre of the locked target region. This is simple and defensible: if the system correctly identified the target, the hand should stop at or near its centre. An alternative would be to extrapolate the current trajectory to find where it naturally terminates, but that is sensitive to noise and adds complexity without improving the key demonstration.

### Robot arm timing

`D_robot = D_remaining × ROBOT_D_SCALE` where ROBOT_D_SCALE = 0.85. The robot is given 85% of the estimated remaining hand travel time. This means the robot arrives at the target position slightly before the hand — making the proactiveness visible. If D_robot equalled D_remaining, the robot and hand would arrive simultaneously and the proactive effect would be invisible.

---

## 7. How does the PyBullet robot arm work?

**File:** `src/robot/pybullet_robot.py`

PyBullet runs in DIRECT mode — no GUI window. This means we control it entirely through the Python API. We load a minimal 2-DOF URDF (two revolute joints, link lengths L1=1.2m, L2=1.0m).

### Why 2-DOF?

A 2-DOF planar arm is the simplest arm that can reach different positions in a plane. It has exactly one joint angle pair per reachable position (with the elbow-up/elbow-down ambiguity resolved by convention). This means the IK has a closed-form analytical solution — no numerical iteration needed. It is also straightforward to explain: "two joints, two angles."

### Analytical IK

Given a target position (x, y) in the arm's local frame:

```
r = sqrt(x² + y²)
cos(θ₂) = (r² − L1² − L2²) / (2·L1·L2)
sin(θ₂) = −sqrt(1 − cos²(θ₂))    [elbow-down convention]
θ₁ = atan2(−x, y) − atan2(L2·sin(θ₂), L1 + L2·cos(θ₂))
```

This gives exact joint angles in one computation, no iterative solver required.

We convert pixel coordinates to world coordinates using a fixed scale factor. Each target maps to a specific world position (red: left, blue: centre, green: right) at a reachable distance from the arm base.

### State machine

- **IDLE** — arm at home position (slightly bent upward), waiting for target lock
- **MOVING** — joint angles interpolate from current to IK target using minimum-jerk: `θ(t) = θ0 + (θ_target − θ0) × (10τ³ − 15τ⁴ + 6τ⁵)`. Smooth, no abrupt starts or stops.
- **HOLDING** — arm held at target until trial reset

The same minimum-jerk polynomial is used for both the hand trajectory model and the robot joint trajectory. This is consistent and means we need only one polynomial implementation.

### Rendering

PyBullet renders are done with `getCameraImage()` in the spec, but on macOS the tiny renderer is unreliable. Instead, we draw the arm geometry directly with OpenCV using forward kinematics to compute joint and end-effector positions, then `cv2.line` and `cv2.circle`. The visual result is equivalent and works reliably on all platforms.

---

## 8. How does the threading model work?

**File:** `experiments/run_webcam_demo.py`

There are four worker threads plus a Flask thread:

| Thread | What it does |
|---|---|
| webcam | Reads frames, runs MediaPipe, writes to shared state |
| inference | Reads latest observation, runs Bayesian update, fires prediction at lock |
| pybullet | Steps robot simulation, renders arm, writes robot frame to shared state |
| composite | Reads all shared state, builds MJPEG frame, pushes to Flask buffer |

All shared state lives in `DemoState`. A single `threading.Lock` protects it. Any thread that reads or writes shared fields must hold the lock.

### The duplicate observation problem

The inference thread runs as fast as the CPU allows — no sleep. The webcam thread pushes a new frame at ~30 fps. Without a guard, the inference thread would read the same observation 30–100 times between new frames arriving, applying the Bayesian likelihood update each time. This would cause near-instant probability collapse.

Fix: the inference thread tracks `last_obs_ts` (the timestamp of the last processed observation) and skips any observation whose timestamp matches. Each observation is processed exactly once.

### The reset coordination problem

When the user presses Reset, all four threads need to know and start a new trial. We use a `reset_count` integer in the shared state. Each thread stores its own `my_reset` value. When `reset_count > my_reset`, the thread knows a reset happened, performs its local cleanup, and updates `my_reset = reset_count`. This is simpler and more reliable than using Events or Barriers, and handles rapid successive resets correctly.

### MJPEG streaming

Flask serves `GET /video_feed` with `Content-Type: multipart/x-mixed-replace; boundary=frame`. Each frame is JPEG-encoded with Pillow and pushed as a multipart boundary. The browser displays it via a plain `<img src="/video_feed">` tag — no JavaScript polling, no WebSocket.

---

## 9. How does the evaluation work?

### Guided trial plan

In the webcam demo, we run a fixed 9-trial plan:

```
red → blue → green → blue → green → red → green → red → blue
```

Three trials per target, interleaved so the robot arm sweeps to different positions each trial. The overlay shows the user which target to reach next, so **ground truth is known automatically** — we don't need to detect the physical target or rely on the user to declare intent.

### Accuracy

For each trial: `target_correct = (predicted_target == ground_truth_target)`. Accuracy = correct trials / total trials. In simulation with noise_std=1.5, we consistently achieve 9/9 (100%). In webcam trials, accuracy depends on how deliberately the user moves.

### Prediction error

```
error = xf_predicted − xf_actual
prediction_error_norm = ||xf_predicted − xf_actual||₂
```

`xf_predicted` is the locked target centre (fixed at lock time). `xf_actual` is the hand position at the moment the user presses Reset, which approximates where the hand stopped. In simulation, errors are 1–4 px because the simulation hand does stop exactly at the target centre. In webcam, errors will be larger (typically 10–50 px) because real hands don't stop at pixel-perfect centres.

### Duration adaptation

After each trial:

```
correction_factor += ADAPTATION_GAIN × (D_actual/D_estimated − correction_factor)
```

This is an exponential moving average of the ratio between actual and estimated duration. If the robot consistently arrives too early (D_estimated too large), the correction factor falls below 1.0 and reduces future estimates. ADAPTATION_GAIN = 0.1 means each trial contributes 10% to the correction — slow but stable adaptation.

`D_adapted = D_estimated × correction_factor` is what gets applied to the robot arm timing.

### What the figures show

- `prediction_error_webcam.png` — bar per trial coloured by target, D_correction on twin axis. Shows whether error is consistent across targets and whether adaptation is working.
- `summary_metrics_webcam.png` — three panels: accuracy per target class, error distribution histogram with mean, and D_correction trajectory across trials. This is the primary quantitative result for the report.

---

## 10. Design decisions specific to this project

These are choices you could justify if asked "why did you do it this way?"

**Wrist vs fingertip tracking:** Wrist (landmark 0) is more stable than fingertip (landmark 8). Fingertips are frequently occluded or off-frame. The wrist gives a reliable anchor point throughout the movement.

**Exponential smoothing vs Kalman filter:** Exponential smoothing is simpler, has one parameter, and works well for this application. Kalman filter would also estimate velocity, handle missing observations, and potentially give better estimates — but adds complexity that isn't necessary here.

**Off-axis distance vs Euclidean distance to goal:** Euclidean distance to goal doesn't discriminate well early in the movement (all targets are far away). Off-axis distance discriminates from the first step because the direction of movement immediately favours one target line over the others.

**Fixed target regions vs colour detection:** Detecting physical coloured blocks with OpenCV would make the system fragile — it requires controlled lighting and specific block colours. Fixed screen regions are reliable and allow the demo to focus on intent inference rather than object detection.

**PyBullet DIRECT mode vs GUI mode:** DIRECT mode means no PyBullet GUI window. The arm is rendered by our own OpenCV drawing into the composite frame. This keeps the whole demo visible in one browser window rather than split across windows.

**2-DOF arm vs 3-DOF or 6-DOF:** 2-DOF has an analytical IK solution that is fully explainable. Higher DOF would require numerical IK and introduce redundancy (multiple solutions). The 2-DOF arm is sufficient to demonstrate proactive reaching and is the right choice for a course demo.

**Threading vs asyncio:** Threading is simpler to reason about for CPU-bound work (MediaPipe inference, PyBullet stepping). Python's GIL does limit true parallelism for CPU work, but in this case the bottleneck is I/O (webcam frame rate, Flask response time) rather than CPU, so threading works fine.

**Flask MJPEG vs WebSocket:** MJPEG via `multipart/x-mixed-replace` requires no client-side JavaScript. The browser displays it as a standard `<img>` tag. A WebSocket approach would allow bidirectional communication but adds frontend complexity with no benefit for a display-only feed.

**reset_count integer vs Event/Barrier for reset coordination:** A counter is simpler. Each thread stores `my_reset` and detects `reset_count > my_reset`. Events require explicit `.set()` and `.clear()`. Barriers require all threads to reach the barrier point simultaneously. The counter approach handles threads running at different speeds naturally.

---

## 11. Limitations and honest answers

**Why might the system lock onto the wrong target?**

If the user's hand path curves (e.g. moves diagonally before heading toward the target), the off-axis and direction cues might briefly favour the wrong target. The confirmation window (10 frames) mitigates this but doesn't eliminate it. A user who moves hesitantly or changes direction mid-movement will confuse the system.

**Is the prediction error meaningful?**

In the webcam demo, `xf_actual` is the hand position when the user presses Reset, which may not be exactly where the hand stopped — it depends on when the user reacts. The error is an approximation. In a real system, you would use a dwell detection algorithm to find the true resting position.

**Does the robot actually arrive before the hand?**

In simulation, yes — the timing is controlled. In webcam mode, D_remaining is estimated from the minimum-jerk model, and the robot is given D_remaining × 0.85 seconds. If the D estimate is accurate, the robot arrives first. If the user moves unusually fast or slow, the estimate will be off. The D_correction_factor adapts across trials to compensate.

**Why is LOCK_MIN_TRAVEL_PX = 200 so large?**

The total path from start to any target is ~240 px. Requiring 200 px means the lock can only fire in the last 17% of the journey. This is conservative but prevents the arm from firing too early when the user's intent isn't clear. In a more sophisticated system, you would dynamically scale this based on observed trajectory consistency.

**What would you improve with more time?**

- Physical target detection (colour segmentation or ArUco markers) so the targets don't have to be fixed screen regions
- Kalman filter for smoother velocity estimates
- A proper dwell detector for accurate xf_actual measurement
- Higher-DOF arm for more realistic reaching
- User-specific adaptation: learn the individual's movement style across sessions

---

## 12. Key numbers to remember

| Metric | Value | Context |
|---|---|---|
| Lock threshold | P > 0.8 | 80% confidence required |
| Min travel before lock | 200 px | ~83% of path to target |
| Confirmation window | 10 frames | ~0.33 s at 30 fps |
| DISTANCE_SCALE | 0.25 | 25% off-axis deviation → likelihood 0.37 |
| DIRECTION_WEIGHT | 0.8 | Reduced from 2.0 for 30 fps stability |
| Smoothing alpha | 0.35 | Exponential position smoothing |
| Robot D scale | 0.85 | Arm arrives 15% ahead of hand |
| Arm link lengths | L1=1.2m, L2=1.0m | 2-DOF planar, max reach 2.2m |
| Simulation accuracy | 9/9 (100%) | noise_std=1.5 px |
| Simulation mean error | ~2 px | Tight because hand stops at exact target centre |
| Webcam expected error | 10–50 px | Real hand stopping position varies |
| Targets | 3 | Red (left), Blue (centre), Green (right) |
| Canvas | 640×480 px | Matches webcam capture size |
