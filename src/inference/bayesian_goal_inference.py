import numpy as np
from typing import Optional

import config
from src.scene.targets import Target, HAND_START, CANVAS_W, CANVAS_H

# Velocity below this (px/s) is treated as stationary — direction likelihood is neutral.
_MIN_SPEED_PX = 5.0

_NORM = np.array([CANVAS_W, CANVAS_H], dtype=float)
_START_NORM = HAND_START / _NORM


def _normalize(pos_px: np.ndarray) -> np.ndarray:
    return pos_px / _NORM


class BayesianGoalInference:
    """
    Bayesian goal recognition over three candidate targets.

    At every timestep:
        P(G | O_1:t)  ∝  P(O_t | G)  *  P(G | O_1:t-1)

    Likelihood  =  off_axis_likelihood  *  direction_likelihood

    off_axis_likelihood:
        How far is the hand from the straight-line path between START and goal G?
        exp( -off_axis_normalized / DISTANCE_SCALE )
        Discriminates from the first frame of movement.

    direction_likelihood:
        Is the hand's velocity vector pointing toward goal G?
        exp( DIRECTION_WEIGHT * cosine_similarity(velocity, direction_to_G) )
        Neutral (1.0) when speed is near zero.
    """

    def __init__(self, targets: list[Target]):
        self.targets = targets
        self.n = len(targets)
        self.posterior: dict[str, float] = {t.name: 1.0 / self.n for t in targets}

        self.history: dict[str, list] = {
            "timestamps": [],
            **{t.name: [] for t in targets},
        }

        self.locked_target: Optional[Target] = None
        self.lock_time: Optional[float] = None
        self.lock_confidence: Optional[float] = None

        # Precompute unit vectors from START toward each goal (normalized coords).
        self._goal_dir: dict[str, np.ndarray] = {}
        for t in targets:
            goal_norm = _normalize(t.position)
            vec = goal_norm - _START_NORM
            dist = np.linalg.norm(vec)
            self._goal_dir[t.name] = vec / dist if dist > 1e-9 else vec

    # ------------------------------------------------------------------
    # Likelihood components
    # ------------------------------------------------------------------

    def _off_axis_distance(self, hand_norm: np.ndarray, goal: Target) -> float:
        """Perpendicular distance from hand to the START→goal line (normalized coords)."""
        hand_vec = hand_norm - _START_NORM
        goal_dir = self._goal_dir[goal.name]
        along = np.dot(hand_vec, goal_dir)
        along = max(along, 0.0)          # clamp: ignore hand moving behind start
        off_axis_vec = hand_vec - along * goal_dir
        return float(np.linalg.norm(off_axis_vec))

    def _off_axis_likelihood(self, hand_norm: np.ndarray, goal: Target) -> float:
        off_axis = self._off_axis_distance(hand_norm, goal)
        return float(np.exp(-off_axis / config.DISTANCE_SCALE))

    def _direction_likelihood(self, hand_px: np.ndarray, velocity_px: np.ndarray,
                               goal: Target) -> float:
        speed = float(np.linalg.norm(velocity_px))
        if speed < _MIN_SPEED_PX:
            return 1.0                   # neutral when stationary

        hand_norm = _normalize(hand_px)
        goal_norm = _normalize(goal.position)
        to_goal = goal_norm - hand_norm
        to_goal_dist = np.linalg.norm(to_goal)
        if to_goal_dist < 1e-9:
            return 1.0                   # already at goal

        unit_vel = velocity_px / speed
        # Direction to goal in pixel space (preserves angle correctly)
        to_goal_px = goal.position - hand_px
        to_goal_px_norm = to_goal_px / np.linalg.norm(to_goal_px)
        cosine_sim = float(np.dot(unit_vel, to_goal_px_norm))
        return float(np.exp(config.DIRECTION_WEIGHT * cosine_sim))

    # ------------------------------------------------------------------
    # Main update
    # ------------------------------------------------------------------

    def update(self, observation) -> dict[str, float]:
        """
        Consume one observation, update posteriors, check lock condition.
        Returns the current posterior dict.
        """
        hand_norm = _normalize(observation.position)

        # Compute likelihoods
        likelihoods: dict[str, float] = {}
        for t in self.targets:
            ol = self._off_axis_likelihood(hand_norm, t)
            dl = self._direction_likelihood(observation.position, observation.velocity, t)
            likelihoods[t.name] = ol * dl

        # Bayesian update: likelihood × prior
        unnormalized = {
            name: likelihoods[name] * self.posterior[name]
            for name in self.posterior
        }

        # Normalize
        total = sum(unnormalized.values())
        if total > 1e-12:
            self.posterior = {name: v / total for name, v in unnormalized.items()}
        # else: numerical underflow — keep previous posterior

        # Record history
        self.history["timestamps"].append(observation.timestamp)
        for t in self.targets:
            self.history[t.name].append(self.posterior[t.name])

        # Lock check
        max_name = max(self.posterior, key=self.posterior.get)
        max_prob = self.posterior[max_name]
        if self.locked_target is None and max_prob > config.CONFIDENCE_THRESHOLD:
            target_map = {t.name: t for t in self.targets}
            self.locked_target = target_map[max_name]
            self.lock_time = observation.timestamp
            self.lock_confidence = max_prob

        return dict(self.posterior)

    def reset(self) -> None:
        self.posterior = {t.name: 1.0 / self.n for t in self.targets}
        self.history = {
            "timestamps": [],
            **{t.name: [] for t in self.targets},
        }
        self.locked_target = None
        self.lock_time = None
        self.lock_confidence = None
