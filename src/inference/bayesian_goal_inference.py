import numpy as np
from typing import Optional

import config
from src.scene.targets import Target, HAND_START, CANVAS_W, CANVAS_H

# Velocity below this (px/s) is treated as stationary — direction likelihood is neutral.
_MIN_SPEED_PX = 5.0

_NORM = np.array([CANVAS_W, CANVAS_H], dtype=float)
_DEFAULT_START_NORM = HAND_START / _NORM


def _normalize(pos_px: np.ndarray) -> np.ndarray:
    return pos_px / _NORM


class BayesianGoalInference:
    """
    Bayesian goal recognition over three candidate targets.

    At every timestep:
        P(G | O_1:t)  ∝  P(O_t | G)  *  P(G | O_1:t-1)

    Likelihood  =  off_axis_likelihood  *  direction_likelihood

    off_axis_likelihood:
        Perpendicular distance from hand to the START→goal line.
        exp( -off_axis_normalized / DISTANCE_SCALE )
        Discriminates from the first frame of movement.

    direction_likelihood:
        Cosine similarity between hand velocity and direction to goal G.
        exp( DIRECTION_WEIGHT * cosine_sim(velocity, to_goal) )
        Neutral (1.0) when speed is near zero.

    reset(start_pos) should be called at the start of each new trial so
    the off-axis reference and goal directions are anchored at the actual
    starting position, not the fixed HAND_START constant.
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

        # Within-trial cumulative travel distance (used for the travel guard).
        self._traveled: float = 0.0
        self._last_pos: Optional[np.ndarray] = None

        # Lock confirmation: require LOCK_CONFIRM_FRAMES consecutive frames above
        # CONFIDENCE_THRESHOLD for the same candidate before committing.
        self._confirm_candidate: Optional[str] = None
        self._confirm_count: int = 0

        # Start reference for off-axis computation — updated by reset(start_pos).
        self._start_norm: np.ndarray = _DEFAULT_START_NORM.copy()

        # Unit vectors from start toward each goal (normalized coords).
        self._goal_dir: dict[str, np.ndarray] = {}
        self._recompute_goal_dirs(self._start_norm)

    # ------------------------------------------------------------------
    # Likelihood components
    # ------------------------------------------------------------------

    def _recompute_goal_dirs(self, start_norm: np.ndarray) -> None:
        for t in self.targets:
            goal_norm = _normalize(t.position)
            vec = goal_norm - start_norm
            dist = np.linalg.norm(vec)
            self._goal_dir[t.name] = vec / dist if dist > 1e-9 else vec

    def _off_axis_distance(self, hand_norm: np.ndarray, goal: Target) -> float:
        """Perpendicular distance from hand to the START→goal line (normalised coords)."""
        hand_vec = hand_norm - self._start_norm
        goal_dir = self._goal_dir[goal.name]
        along = max(np.dot(hand_vec, goal_dir), 0.0)
        off_axis_vec = hand_vec - along * goal_dir
        return float(np.linalg.norm(off_axis_vec))

    def _off_axis_likelihood(self, hand_norm: np.ndarray, goal: Target) -> float:
        return float(np.exp(-self._off_axis_distance(hand_norm, goal) / config.DISTANCE_SCALE))

    def _direction_likelihood(self, hand_px: np.ndarray, velocity_px: np.ndarray,
                               goal: Target) -> float:
        speed = float(np.linalg.norm(velocity_px))
        if speed < _MIN_SPEED_PX:
            return 1.0

        to_goal_px = goal.position - hand_px
        to_goal_dist = np.linalg.norm(to_goal_px)
        if to_goal_dist < 1e-9:
            return 1.0

        unit_vel = velocity_px / speed
        to_goal_unit = to_goal_px / to_goal_dist
        cosine_sim = float(np.dot(unit_vel, to_goal_unit))
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

        # Cumulative within-trial travel (lock guard — position-space, not from HAND_START)
        if self._last_pos is not None:
            self._traveled += float(np.linalg.norm(observation.position - self._last_pos))
        self._last_pos = observation.position.copy()

        # Likelihoods
        likelihoods: dict[str, float] = {}
        for t in self.targets:
            ol = self._off_axis_likelihood(hand_norm, t)
            dl = self._direction_likelihood(observation.position, observation.velocity, t)
            likelihoods[t.name] = ol * dl

        # Bayesian update
        unnormalized = {
            name: likelihoods[name] * self.posterior[name]
            for name in self.posterior
        }
        total = sum(unnormalized.values())
        if total > 1e-12:
            self.posterior = {name: v / total for name, v in unnormalized.items()}

        # Record history
        self.history["timestamps"].append(observation.timestamp)
        for t in self.targets:
            self.history[t.name].append(self.posterior[t.name])

        # Lock check — requires:
        #   1. Enough travel so the hand has clearly committed to a direction.
        #   2. LOCK_CONFIRM_FRAMES consecutive frames where the same target
        #      holds P > CONFIDENCE_THRESHOLD, preventing momentary spikes.
        max_name = max(self.posterior, key=self.posterior.get)
        max_prob = self.posterior[max_name]
        if self.locked_target is None and self._traveled >= config.LOCK_MIN_TRAVEL_PX:
            if max_prob > config.CONFIDENCE_THRESHOLD:
                if max_name == self._confirm_candidate:
                    self._confirm_count += 1
                else:
                    self._confirm_candidate = max_name
                    self._confirm_count = 1

                if self._confirm_count >= config.LOCK_CONFIRM_FRAMES:
                    target_map = {t.name: t for t in self.targets}
                    self.locked_target = target_map[max_name]
                    self.lock_time = observation.timestamp
                    self.lock_confidence = max_prob
            else:
                # Probability dipped below threshold — reset the confirmation streak.
                self._confirm_candidate = None
                self._confirm_count = 0

        return dict(self.posterior)

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def reset(self, start_pos: Optional[np.ndarray] = None) -> None:
        """
        Reset for a new trial.

        Parameters
        ----------
        start_pos : optional pixel-space starting position.
            When provided, goal directions and the off-axis reference are
            re-anchored at this position so inference works correctly when
            the user does not start from the fixed HAND_START constant.
            Pass the first observed hand position at the start of each trial.
        """
        self.posterior = {t.name: 1.0 / self.n for t in self.targets}
        self.history = {
            "timestamps": [],
            **{t.name: [] for t in self.targets},
        }
        self.locked_target = None
        self.lock_time = None
        self.lock_confidence = None
        self._traveled = 0.0
        self._last_pos = None
        self._confirm_candidate = None
        self._confirm_count = 0

        if start_pos is not None:
            self._start_norm = _normalize(np.asarray(start_pos, dtype=float))
        else:
            self._start_norm = _DEFAULT_START_NORM.copy()

        self._recompute_goal_dirs(self._start_norm)
