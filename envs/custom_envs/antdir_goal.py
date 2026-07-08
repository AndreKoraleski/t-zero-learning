"""
Ant direction goal: reward aligned with a (possibly time-varying) planar target,
with goal-related features appended to observations.

Schedule modes:
  - fixed: explicit `directions` and `switch_after_steps` (length len(directions)-1).
  - random_two: each episode samples two distinct directions from `direction_pool`
    and a switch index uniformly from `switch_step_range` (inclusive).

Switch semantics: after each env step, `steps_done` increments (1..). While
`steps_done <= cumsum(switch_after_steps)[0]` stay in segment 0; then segment 1,
etc. (searchsorted on cumulative switch lengths).

Visualization: ``make_antdir_goal_env`` defaults to bundled ``assets/ant_vis.xml`` (larger
floor) and a farther tracking camera unless ``xml_file`` / ``default_camera_config`` are
set. For ``default_camera_config``, the baseline ``{"distance": 9.0}`` is merged with any
dict you pass: your keys override the baseline.
"""

from __future__ import annotations

from pathlib import Path

import gymnasium as gym
import numpy as np

_ANT_VIS_XML = Path(__file__).resolve().parent / "assets" / "ant_vis.xml"
_DEFAULT_CAMERA_CONFIG = {"distance": 9.0}

CARDINAL_DIRECTIONS = np.array(
    [[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]], dtype=np.float64
)

_ANT_GOAL_KEYS = frozenset(
    {
        "schedule_mode",
        "directions",
        "switch_after_steps",
        "direction_pool",
        "switch_step_range",
        "obs_include_dtheta",
    }
)


def _normalize_dir(d: np.ndarray) -> np.ndarray:
    d = np.asarray(d, dtype=np.float64).reshape(-1)
    if d.shape != (2,):
        raise ValueError(f"direction must have shape (2,), got {d.shape}")
    n = np.linalg.norm(d)
    if n < 1e-8:
        return np.array([1.0, 0.0], dtype=np.float64)
    return d / n


def _heading_sin_cos(vx: float, vy: float, ux: float, uy: float, eps: float = 1e-8):
    nv = float(np.hypot(vx, vy))
    if nv < eps:
        return 0.0, 1.0
    vxn, vyn = vx / nv, vy / nv
    dot = vxn * ux + vyn * uy
    cross = vxn * uy - vyn * ux
    return float(cross), float(dot)


def _stamp_disk(out: np.ndarray, row: int, col: int, color: np.ndarray, radius: int) -> None:
    h, w = out.shape[:2]
    r0, r1 = max(0, row - radius), min(h, row + radius + 1)
    c0, c1 = max(0, col - radius), min(w, col + radius + 1)
    if r0 >= r1 or c0 >= c1:
        return
    rs = np.arange(r0, r1, dtype=np.int64)
    cs = np.arange(c0, c1, dtype=np.int64)
    rr, cc = np.meshgrid(rs, cs, indexing="ij")
    mask = (rr - row) ** 2 + (cc - col) ** 2 <= radius**2
    out[rr[mask], cc[mask]] = color


def _draw_segment_rgb(
    out: np.ndarray,
    r0: int,
    c0: int,
    r1: int,
    c1: int,
    color: np.ndarray,
    thick: int = 2,
) -> None:
    h, w = out.shape[:2]
    n = int(max(abs(r1 - r0), abs(c1 - c0), 1)) * 2 + 1
    for t in np.linspace(0.0, 1.0, n, dtype=np.float64):
        r = int(round(r0 + (r1 - r0) * t))
        c = int(round(c0 + (c1 - c0) * t))
        if 0 <= r < h and 0 <= c < w:
            _stamp_disk(out, r, c, color, max(0, thick // 2))


def _draw_arrow_world_hud(
    out: np.ndarray,
    origin_row: int,
    origin_col: int,
    wx: float,
    wy: float,
    length_px: float,
    color: np.ndarray,
    thick: int = 2,
) -> None:
    """Draw a planar arrow: world +x -> +column, world +y -> -row."""
    nrm = float(np.hypot(wx, wy))
    if nrm < 1e-8:
        return
    dx, dy = wx / nrm, wy / nrm
    tip_c = int(round(origin_col + dx * length_px))
    tip_r = int(round(origin_row - dy * length_px))
    _draw_segment_rgb(out, origin_row, origin_col, tip_r, tip_c, color, thick=thick)
    head = 0.28 * length_px
    back_c = tip_c - dx * head
    back_r = tip_r + dy * head
    px, py = -dy, dx
    wing = 0.45 * head
    _draw_segment_rgb(
        out,
        tip_r,
        tip_c,
        int(round(back_r + py * wing)),
        int(round(back_c + px * wing)),
        color,
        thick=max(1, thick - 1),
    )
    _draw_segment_rgb(
        out,
        tip_r,
        tip_c,
        int(round(back_r - py * wing)),
        int(round(back_c - px * wing)),
        color,
        thick=max(1, thick - 1),
    )


def _overlay_motion_goal_hud(
    frame: np.ndarray,
    vx: float,
    vy: float,
    ux: float,
    uy: float,
) -> np.ndarray:
    """
    Bottom-left HUD: green arrow = horizontal velocity direction; magenta = goal ``u``.
    Axes are world X/Y (not projected onto the camera view).
    """
    out = np.array(frame, copy=True, dtype=np.uint8, order="C")
    h, w = out.shape[:2]
    margin = 8
    box = 104
    r1 = h - margin
    r0 = max(0, r1 - box)
    c0 = margin
    c1 = min(w, c0 + box)
    if r0 >= r1 or c0 >= c1:
        return out
    out[r0:r1, c0:c1] //= 2
    cy = (r0 + r1) // 2
    cx = (c0 + c1) // 2
    arrow_len = min(box // 2 - 6, 40)
    vel_color = np.array([0, 220, 0], dtype=np.uint8)
    goal_color = np.array([255, 0, 200], dtype=np.uint8)
    _draw_arrow_world_hud(out, cy, cx, vx, vy, float(arrow_len), vel_color, thick=3)
    _draw_arrow_world_hud(out, cy, cx, ux, uy, float(arrow_len * 0.85), goal_color, thick=2)
    legend_r = r0 + 6
    _stamp_disk(out, legend_r, c0 + 8, vel_color, 3)
    _stamp_disk(out, legend_r, c0 + 28, goal_color, 3)
    return out


class AntDirGoalWrapper(gym.Wrapper):
    """
    Appends goal / velocity / error features and applies direction-conditioned
    forward reward. Direction schedule is configured per mode (see module doc).
    """

    def __init__(
        self,
        env,
        schedule_mode: str = "fixed",
        directions: list | None = None,
        switch_after_steps: list | None = None,
        direction_pool: list | None = None,
        switch_step_range: tuple[int, int] | list[int] | None = None,
        obs_include_dtheta: bool = False,
    ):
        """
        Parameters
        ----------
        env : gymnasium.Env
            Underlying environment (expected: ``Ant-v5`` with 1-D Box observations).
        schedule_mode : str, default ``"fixed"``
            ``"fixed"``: use ``directions`` and ``switch_after_steps`` for every episode.
            ``"random_two"``: each ``reset`` samples two distinct headings from
            ``direction_pool`` and a switch step from ``switch_step_range``.
        directions : list of [x, y] | None
            Used when ``schedule_mode="fixed"``. Rows are planar goal directions
            (normalized internally). Default ``[[1, 0], [-1, 0]]`` if ``None``.
            Length ``n`` must match ``len(switch_after_steps) == n - 1`` (or
            ``n == 1`` with ``switch_after_steps=[]``).
        switch_after_steps : list of int | None
            Used when ``schedule_mode="fixed"``. Positive segment lengths in env steps:
            after ``sum(switch_after_steps[:k])`` completed steps, the goal advances
            to ``directions[k+1]``. Default ``[500]`` for two directions, or ``[]``
            if ``len(directions) == 1``.
        direction_pool : list of [x, y] | None
            Candidate directions for ``schedule_mode="random_two"``. Shape ``(m, 2)``.
            Default: four cardinals ``(±1,0), (0,±1)``.
        switch_step_range : (int, int) | [int, int] | None
            Inclusive integer range for the first-segment length when
            ``schedule_mode="random_two"``. Default ``(200, 800)``.
        obs_include_dtheta : bool, default False
            If True, append signed heading error ``atan2(sin Δθ, cos Δθ)`` to the
            observation (9 goal dims); otherwise 8 dims (no redundant scalar if
            ``sin/cos`` are already present).
        """
        super().__init__(env)
        self.schedule_mode = schedule_mode
        self.obs_include_dtheta = bool(obs_include_dtheta)
        self._goal_dim = 9 if self.obs_include_dtheta else 8

        if direction_pool is None:
            self._direction_pool = CARDINAL_DIRECTIONS.copy()
        else:
            self._direction_pool = np.array(direction_pool, dtype=np.float64)
            if self._direction_pool.ndim != 2 or self._direction_pool.shape[1] != 2:
                raise ValueError("direction_pool must be shape (n, 2)")

        if switch_step_range is None:
            self._switch_step_range = (200, 800)
        else:
            lo, hi = switch_step_range
            self._switch_step_range = (int(lo), int(hi))

        self._fixed_directions = None
        self._fixed_switch_after = None
        if schedule_mode == "fixed":
            if directions is None:
                directions = [[1.0, 0.0], [-1.0, 0.0]]
            dirs_pre = np.array(directions, dtype=np.float64)
            if dirs_pre.ndim != 2 or dirs_pre.shape[1] != 2:
                raise ValueError("directions must be shape (n, 2)")
            if switch_after_steps is None:
                switch_after_steps = (
                    [] if len(dirs_pre) == 1 else [500]
                )
            self._fixed_directions = dirs_pre
            self._fixed_switch_after = np.array(switch_after_steps, dtype=np.int64)
            exp_sw = max(0, len(self._fixed_directions) - 1)
            if self._fixed_switch_after.size != exp_sw:
                raise ValueError(
                    "switch_after_steps must have length len(directions) - 1"
                )
            if exp_sw > 0 and np.any(self._fixed_switch_after <= 0):
                raise ValueError("switch_after_steps entries must be positive")

        elif schedule_mode == "random_two":
            if len(self._direction_pool) < 2:
                raise ValueError("random_two needs at least two directions in direction_pool")
        else:
            raise ValueError("schedule_mode must be 'fixed' or 'random_two'")

        base_low = env.observation_space.low
        base_high = env.observation_space.high
        base_shape = env.observation_space.shape
        if len(base_shape) != 1:
            raise ValueError("AntDirGoalWrapper expects 1-D Box observations from Ant-v5")
        low = np.concatenate([base_low, np.full(self._goal_dim, -np.inf)])
        high = np.concatenate([base_high, np.full(self._goal_dim, np.inf)])
        self.observation_space = gym.spaces.Box(low=low, high=high, dtype=np.float64)

        self._steps_done = 0
        self._directions = self._fixed_directions
        self._switch_after_steps = (
            self._fixed_switch_after
            if self._fixed_switch_after is not None
            else np.zeros(0, dtype=np.int64)
        )
        self.u = np.array([1.0, 0.0], dtype=np.float64)
        self._episode_switch_step: int | None = None
        self._episode_directions_sample: np.ndarray | None = None
        self._hud_vx = 0.0
        self._hud_vy = 0.0
        self._hud_ux = 1.0
        self._hud_uy = 0.0

    def render(self, *args, **kwargs):
        """Render the base env, drawing the world-frame velocity/goal HUD
        (see ``_overlay_motion_goal_hud``) on RGB frames."""
        frame = self.env.render(*args, **kwargs)
        if (
            frame is None
            or not isinstance(frame, np.ndarray)
            or frame.ndim != 3
            or frame.shape[2] != 3
        ):
            return frame
        return _overlay_motion_goal_hud(
            frame,
            self._hud_vx,
            self._hud_vy,
            self._hud_ux,
            self._hud_uy,
        )

    def _segment_index(self) -> int:
        if self._directions is None or len(self._directions) == 0:
            return 0
        if len(self._switch_after_steps) == 0:
            return 0
        cum = np.cumsum(self._switch_after_steps)
        idx = int(np.searchsorted(cum, self._steps_done, side="left"))
        return min(idx, len(self._directions) - 1)

    def _prepare_schedule(self) -> None:
        if self.schedule_mode == "fixed":
            self._directions = self._fixed_directions
            self._switch_after_steps = self._fixed_switch_after
            self._episode_switch_step = None
            self._episode_directions_sample = None
            return

        rng = self.np_random
        lo, hi = self._switch_step_range
        switch_step = int(rng.integers(lo, hi + 1))
        idx = rng.choice(len(self._direction_pool), size=2, replace=False)
        d0 = self._direction_pool[int(idx[0])]
        d1 = self._direction_pool[int(idx[1])]
        self._directions = np.stack([d0, d1], axis=0)
        self._switch_after_steps = np.array([switch_step], dtype=np.int64)
        self._episode_switch_step = switch_step
        self._episode_directions_sample = self._directions.copy()

    def _goal_extras(self, info: dict) -> np.ndarray:
        vx = float(info["x_velocity"])
        vy = float(info["y_velocity"])
        ux, uy = float(self.u[0]), float(self.u[1])
        sin_e, cos_e = _heading_sin_cos(vx, vy, ux, uy)
        dtheta = float(np.arctan2(sin_e, cos_e))
        parts = [
            ux,
            uy,
            vx,
            vy,
            vx - ux,
            vy - uy,
            sin_e,
            cos_e,
        ]
        if self.obs_include_dtheta:
            parts.append(dtheta)
        return np.array(parts, dtype=np.float64)

    def _augment_obs(self, obs: np.ndarray, info: dict) -> np.ndarray:
        return np.concatenate([obs, self._goal_extras(info)], axis=-1)

    def _forward_reward(self, info: dict) -> float:
        vx = info["x_velocity"]
        vy = info["y_velocity"]
        v = np.array([vx, vy], dtype=np.float64)
        norm_v = np.linalg.norm(v)
        if norm_v < 1e-8:
            return 0.0
        norm_u = np.linalg.norm(self.u)
        if norm_u < 1e-8:
            return 0.0
        v_n = v / norm_v
        u_n = self.u / norm_u
        return float(norm_v * np.dot(v_n, u_n))

    def _compute_reward(self, info: dict):
        reward_forward = self._forward_reward(info)
        reward_ctrl = info["reward_ctrl"]
        reward_contact = info["reward_contact"]
        reward_survive = info["reward_survive"]
        reward = reward_forward + reward_ctrl + reward_contact + reward_survive
        return reward, dict(reward_forward=reward_forward, reward_total=reward)

    def _schedule_info(self, segment_idx: int) -> dict:
        out = dict(
            target_dir=self.u.copy(),
            direction_segment=int(segment_idx),
        )
        if self._episode_switch_step is not None:
            out["sampled_switch_step"] = int(self._episode_switch_step)
        if self._episode_directions_sample is not None:
            out["sampled_directions"] = self._episode_directions_sample.copy()
        return out

    def reset(self, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        self._steps_done = 0
        self._prepare_schedule()
        self.u = _normalize_dir(self._directions[0])
        self._hud_vx = 0.0
        self._hud_vy = 0.0
        self._hud_ux = float(self.u[0])
        self._hud_uy = float(self.u[1])
        reward_info_stub = {"x_velocity": 0.0, "y_velocity": 0.0}
        obs = self._augment_obs(obs, reward_info_stub)
        info = dict(info)
        info.update(self._schedule_info(0))
        return obs, info

    def step(self, action):
        obs, _, terminated, truncated, info = self.env.step(action)
        self._steps_done += 1
        seg = self._segment_index()
        self.u = _normalize_dir(self._directions[seg])
        reward, reward_info = self._compute_reward(info)
        info = dict(info)
        info.update(reward_info)
        info.update(self._schedule_info(seg))
        obs = self._augment_obs(obs, info)
        self._hud_vx = float(info["x_velocity"])
        self._hud_vy = float(info["y_velocity"])
        self._hud_ux = float(self.u[0])
        self._hud_uy = float(self.u[1])
        return obs, reward, terminated, truncated, info


def make_antdir_goal_env(render_mode="rgb_array", **kwargs):
    ant_kwargs = {k: v for k, v in kwargs.items() if k not in _ANT_GOAL_KEYS}
    goal_kwargs = {k: v for k, v in kwargs.items() if k in _ANT_GOAL_KEYS}
    if "xml_file" not in ant_kwargs:
        ant_kwargs["xml_file"] = str(_ANT_VIS_XML)
    if "default_camera_config" not in ant_kwargs:
        ant_kwargs["default_camera_config"] = dict(_DEFAULT_CAMERA_CONFIG)
    else:
        merged = dict(_DEFAULT_CAMERA_CONFIG)
        merged.update(ant_kwargs["default_camera_config"])
        ant_kwargs["default_camera_config"] = merged
    base_env = gym.make(
        "Ant-v5",
        render_mode=render_mode,
        healthy_z_range=(0.2, 0.9),
        **ant_kwargs,
    )
    return AntDirGoalWrapper(base_env, **goal_kwargs)
