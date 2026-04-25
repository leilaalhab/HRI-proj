import os
import numpy as np
import cv2
import pybullet
import pybullet_data

import config
from src.prediction.minimum_jerk import minimum_jerk_trajectory

# Analytical 2-DOF planar IK constants
_L1 = 1.2   # link 1 length (m)
_L2 = 1.0   # link 2 length (m)

# When joint_1 = 0, link_1 points along +Y.
# End-effector world position:
#   x_ee = -L1*sin(θ1) - L2*sin(θ1 + θ2)
#   y_ee =  L1*cos(θ1) + L2*cos(θ1 + θ2)

# World-space positions for each candidate target.
# Chosen to be comfortably reachable (r < L1+L2 = 2.2 m).
_TARGET_WORLD = {
    "red":   np.array([-1.0, 1.5, 0.0]),
    "blue":  np.array([ 0.0, 2.0, 0.0]),
    "green": np.array([ 1.0, 1.5, 0.0]),
}

_HOME_ANGLES = np.array([0.0, 0.4])  # resting pose: arm slightly bent upward

_URDF_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "assets", "planar_arm.urdf",
)

# PyBullet joint indices for the two revolute joints
_JOINT_1_IDX = 0
_JOINT_2_IDX = 1


class PybulletRobot:
    """
    2-DOF planar robot arm simulated with PyBullet in DIRECT mode.

    After target lock:
      1. activate(target_name, D_robot) computes IK and builds a joint trajectory.
      2. step() advances joint positions along that trajectory using minimum-jerk.
      3. render() returns an RGB numpy array via getCameraImage.

    The arm operates in its own world coordinate space, independent of the
    pixel canvas used for hand tracking.
    """

    def __init__(self):
        self._client = pybullet.connect(pybullet.DIRECT)
        pybullet.setAdditionalSearchPath(
            pybullet_data.getDataPath(), physicsClientId=self._client
        )
        pybullet.setGravity(0, 0, 0, physicsClientId=self._client)

        self._arm = pybullet.loadURDF(
            _URDF_PATH,
            basePosition=[0, 0, 0],
            useFixedBase=True,
            physicsClientId=self._client,
        )

        self._current_angles = _HOME_ANGLES.copy()
        self._apply_angles(self._current_angles)
        pybullet.stepSimulation(physicsClientId=self._client)

        # Movement state
        self._state = "IDLE"          # IDLE | MOVING | HOLDING
        self._joint_traj: list[np.ndarray] = []
        self._traj_idx: int = 0
        self._target_world: np.ndarray | None = None

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def activate(self, target_name: str, D_robot: float) -> None:
        """Compute IK for target and build minimum-jerk joint trajectory."""
        target_world = _TARGET_WORLD.get(target_name)
        if target_world is None:
            return

        self._target_world = target_world
        target_angles = self._ik(target_world)
        self._joint_traj = _build_joint_traj(
            self._current_angles, target_angles, D_robot, config.SIMULATION_DT
        )
        self._traj_idx = 0
        self._state = "MOVING"

    def step(self) -> None:
        """Advance arm one timestep along the joint trajectory."""
        if self._state != "MOVING":
            return
        if self._traj_idx >= len(self._joint_traj):
            self._state = "HOLDING"
            return
        self._current_angles = self._joint_traj[self._traj_idx]
        self._apply_angles(self._current_angles)
        pybullet.stepSimulation(physicsClientId=self._client)
        self._traj_idx += 1

    def step_to_end(self) -> None:
        """Fast-forward to the final joint configuration (for static figures)."""
        while self._state == "MOVING":
            self.step()

    def render(self) -> np.ndarray:
        """
        Return an RGB (H, W, 3) numpy array visualising the arm pose.

        PyBullet is used for IK and simulation; the render is a clean 2D
        diagram drawn with OpenCV from the current joint angles.
        """
        w, h = config.PYBULLET_RENDER_W, config.PYBULLET_RENDER_H
        canvas = np.full((h, w, 3), 20, dtype=np.uint8)  # near-black bg

        t1, t2 = float(self._current_angles[0]), float(self._current_angles[1])

        # Forward kinematics (same convention as IK):
        #   x_ee = -L1*sin(θ1) - L2*sin(θ1+θ2)
        #   y_ee =  L1*cos(θ1) + L2*cos(θ1+θ2)
        base = np.array([0.0, 0.0])
        j1   = np.array([-_L1 * np.sin(t1),
                          _L1 * np.cos(t1)])
        ee   = np.array([-_L1 * np.sin(t1) - _L2 * np.sin(t1 + t2),
                          _L1 * np.cos(t1) + _L2 * np.cos(t1 + t2)])

        # Map world coords → canvas pixels
        # World y range: 0 .. 2.3 (arm fully extended upward)
        # World x range: -1.2 .. 1.2
        margin = 60
        def to_px(pt):
            px = int(margin + (pt[0] + 1.25) / 2.5 * (w - 2 * margin))
            py = int(h - margin - pt[1] / 2.4 * (h - 2 * margin))
            return (px, py)

        b_px  = to_px(base)
        j1_px = to_px(j1)
        ee_px = to_px(ee)

        # Target indicator (ghost circle at locked target world position)
        if self._target_world is not None:
            t_px = to_px(self._target_world[:2])
            cv2.circle(canvas, t_px, 14, (60, 60, 60), -1)
            cv2.circle(canvas, t_px, 14, (100, 100, 100), 2)

        # Link 2 (lighter blue)
        cv2.line(canvas, j1_px, ee_px, (100, 160, 220), 10, cv2.LINE_AA)
        # Link 1 (steel blue)
        cv2.line(canvas, b_px, j1_px, (60, 110, 180), 14, cv2.LINE_AA)

        # Joint circles
        cv2.circle(canvas, b_px,  10, (180, 180, 180), -1)  # base
        cv2.circle(canvas, j1_px,  8, (140, 200, 240), -1)  # elbow
        cv2.circle(canvas, ee_px, 11, (220,  60,  60), -1)  # end-effector (red)

        # Labels
        cv2.putText(canvas, "base",  (b_px[0] - 12, b_px[1] + 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (140, 140, 140), 1, cv2.LINE_AA)
        cv2.putText(canvas, "EE",    (ee_px[0] - 10, ee_px[1] - 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (220, 100, 100), 1, cv2.LINE_AA)
        cv2.putText(canvas, f"theta1={np.degrees(t1):.1f}deg",
                    (8, h - 36), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (160, 160, 160), 1)
        cv2.putText(canvas, f"theta2={np.degrees(t2):.1f}deg",
                    (8, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (160, 160, 160), 1)

        return canvas

    def reset(self) -> None:
        """Return arm to home position and clear state."""
        self._current_angles = _HOME_ANGLES.copy()
        self._apply_angles(self._current_angles)
        pybullet.stepSimulation(physicsClientId=self._client)
        self._state = "IDLE"
        self._joint_traj = []
        self._traj_idx = 0
        self._target_world = None

    def disconnect(self) -> None:
        pybullet.disconnect(physicsClientId=self._client)

    @property
    def state(self) -> str:
        return self._state

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _apply_angles(self, angles: np.ndarray) -> None:
        pybullet.resetJointState(
            self._arm, _JOINT_1_IDX, float(angles[0]),
            physicsClientId=self._client,
        )
        pybullet.resetJointState(
            self._arm, _JOINT_2_IDX, float(angles[1]),
            physicsClientId=self._client,
        )

    def _ik(self, target_world: np.ndarray) -> np.ndarray:
        """
        Analytical 2-DOF planar IK.

        With joint_1=0, link_1 points along +Y.
        End-effector position:
            x = -L1*sin(θ1) - L2*sin(θ1+θ2)
            y =  L1*cos(θ1) + L2*cos(θ1+θ2)

        Solution (elbow-up configuration):
            cos(θ2) = (r² - L1² - L2²) / (2·L1·L2)
            θ1 = atan2(-x, y) - atan2(L2·sin(θ2), L1 + L2·cos(θ2))
        """
        x = float(target_world[0])
        y = float(target_world[1])
        r = np.sqrt(x**2 + y**2)

        # Clamp to reachable workspace with small margin
        r_min = abs(_L1 - _L2) + 1e-4
        r_max = _L1 + _L2 - 1e-4
        r = np.clip(r, r_min, r_max)

        # Scale target to lie on the reachable boundary if clamped
        if np.sqrt(x**2 + y**2) > r_max:
            scale = r_max / np.sqrt(x**2 + y**2)
            x, y = x * scale, y * scale

        cos_t2 = (r**2 - _L1**2 - _L2**2) / (2.0 * _L1 * _L2)
        theta2 = np.arccos(np.clip(cos_t2, -1.0, 1.0))

        k1 = _L1 + _L2 * np.cos(theta2)
        k2 = _L2 * np.sin(theta2)
        theta1 = np.arctan2(-x, y) - np.arctan2(k2, k1)

        return np.array([theta1, theta2])


def _build_joint_traj(
    start: np.ndarray,
    end: np.ndarray,
    D: float,
    dt: float,
) -> list[np.ndarray]:
    """
    Minimum-jerk interpolation applied independently to each joint angle.
    Returns a list of 2-element angle arrays, one per timestep.
    """
    traj_j1 = minimum_jerk_trajectory(
        np.array([start[0]]), np.array([end[0]]), D, dt
    )
    traj_j2 = minimum_jerk_trajectory(
        np.array([start[1]]), np.array([end[1]]), D, dt
    )
    return [np.array([p1[0], p2[0]]) for p1, p2 in zip(traj_j1, traj_j2)]
