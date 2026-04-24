# Real-Time Human Intent Recognition for Proactive Robot Handover

## Project Goal

This project builds a real-time Human-Robot Interaction (HRI) demo. A webcam tracks the human hand as it moves toward one of three coloured target blocks (red, blue, green). The system infers the intended target using Bayesian goal recognition, predicts the final hand position using a minimum-jerk trajectory model, and shows a proactive virtual robot response — all in real time.

---

## Real-Time Perception Route

This project follows the **real-time perception route**. The final demo uses live webcam-based hand tracking, not simulation. Simulation is implemented only as a development and debugging tool (Stages 3–7). The final evidence — screenshots, figures, logs, and video — must come from webcam trials.

---

## Assignment Requirements

| Requirement | Implementation |
|---|---|
| Webcam-based perception | MediaPipe hand tracking via OpenCV |
| Bayesian intent inference | `src/inference/bayesian_goal_inference.py` |
| Minimum-jerk prediction | `src/prediction/minimum_jerk.py` |
| Target lock at P > 0.8 | Confidence threshold in `config.py` |
| Virtual robot response | `src/robot/virtual_robot.py` |
| Prediction error logging | `src/evaluation/metrics.py` |
| Trial logs | `results/logs/trials.csv` |

---

## HRI Loop

```
Perception → Inference → Decision → Robot Action → Feedback → Adaptation
```

1. **Perception** — webcam frame → MediaPipe → wrist/fingertip position
2. **Inference** — Bayesian update: `P(G | O) ∝ P(O | G) * P(G)` for each target
3. **Decision** — lock target when `P(G | O) > 0.8`
4. **Robot Action** — virtual gripper moves toward predicted interception point
5. **Feedback** — compare `xf_predicted` vs `xf_actual`, compute error
6. **Adaptation** — adjust duration estimate `D` for next trial

---

## Bayesian Inference

At every time step, for each goal G and current observation O:

```
P(G | O) = P(O | G) * P(G) / sum_i( P(O | G_i) * P(G_i) )
```

Likelihood uses two cues:

```
likelihood = distance_likelihood * direction_likelihood
distance_likelihood  = exp(-distance / distance_scale)
direction_likelihood = exp(direction_weight * cosine_similarity)
```

Posteriors are normalised after every update.

---

## Minimum-Jerk Prediction

After target lock, predicted trajectory follows:

```
x(t) = x0 + (xf - x0) * (10*tau^3 - 15*tau^4 + 6*tau^5)
tau = t / D
```

`D` is estimated from remaining distance and current hand speed.

---

## Target Lock Threshold

The target is locked when the maximum posterior exceeds **0.8**:

```
P(G | O) > 0.8  →  target locked
```

This threshold is defined in `config.py` as `CONFIDENCE_THRESHOLD`.

---

## How to Run

```bash
pip install -r requirements.txt

# Webcam demo (final real-time mode)
python experiments/run_webcam_demo.py

# Simulation/debug mode
python experiments/run_simulation.py

# Multi-trial evaluation
python experiments/run_trials.py
```

---

## Outputs

```
results/figures/          — probability plots, trajectory plots, summary metrics
results/screenshots/      — webcam demo screenshots for the report
results/animations/       — optional HRI loop animation
results/logs/trials.csv   — per-trial log with error, lock time, accuracy
```

---

## Current Limitations

- Targets are fixed screen regions; they are not physically detected by colour.
- The virtual robot is a 2D marker, not a physical robot arm.
- Minimum-jerk duration `D` is estimated from speed; no force/dynamics model.

---

## Report / Video / Interview Notes

See `docs/interview_notes.md` for viva question answers and `docs/demo_script.md` for the 10-minute video outline (added in Stage 10).
