"""
Pure-PyTorch logic of the FormationNav task.

Nothing in this file imports Isaac Sim, so it runs (and is unit-tested) on any machine.
Both the Isaac Sim environment (`env.py`) and the point-mass surrogate
(`pointmass_env.py`) only supply drone states and consume observations / rewards.

Episode structure (per parallel environment)
-------------------------------------------
1. FORM   The drones start on the ground in a planar grid. Each drone is assigned a slot of
          the requested formation hovering above the start area and must take off and
          build the formation.
2. NAV    Once every drone has been within `form_tolerance` of its slot for `form_steps`
          steps, the formation's reference point moves toward the goal along a straight
          line through waypoints spaced `waypoint_spacing` apart (receding-horizon waypoints,
          paper Sec. IV-E). The drones track their slots and avoid obstacles by locally
          deforming the formation.
3. HOLD   When the last waypoint (the goal) is active and the swarm centroid is within
          `hold_enter_dist` of it, the drones must hold the formation at the goal until the
          episode ends.

Obstacles
---------
* static : vertical cylinders (pillars) placed in a corridor along the path.
* dynamic: spheres moving back and forth on straight segments that cross the path.
Scenario per episode: none | static | dynamic | mixed (or a random mix for training).

All positions are in the environment frame (the env origin is the start-area centre).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, fields
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

PHASE_FORM, PHASE_NAV, PHASE_HOLD = 0, 1, 2
SCENARIOS = ("none", "static", "dynamic", "mixed")
FORMATIONS = ("plane", "cube", "sphere", "pyramid", "line", "column", "v", "circle")
STAT_KEYS = (
    "return",
    "episode_len",
    "formation_error",
    "slot_error",
    "progress",
    "success",
    "hold_ratio",
    "collisions_drone",
    "collisions_obstacle",
    "min_obstacle_clearance",
    "min_separation",
    "smoothness",
    "time_to_form",
    "time_to_goal",
    "crashed",
    "crash_ground",
    "crash_flip",
    "crash_bounds",
    "crash_drone_collision",
    "crash_obstacle_collision",
    "scenario",
    "formation_id",
)


# ----------------------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------------------


@dataclass
class FormationNavConfig:
    # swarm and formation
    num_drones: int = 8
    formation: str = "dynamic"  # a name from FORMATIONS, "dynamic" (sample from pool) or "custom"
    formation_pool: Tuple[str, ...] = ("cube", "sphere", "pyramid", "plane")
    custom_formation: Optional[Tuple[Tuple[float, float, float], ...]] = None  # metres
    formation_spacing: float = 1.2  # minimum distance between formation slots (m)
    formation_altitude: float = 1.5  # altitude of the lowest formation slot (m)
    ground_spacing: float = 1.2  # spacing of the take-off grid (m)
    spawn_height: float = 0.1  # z of the drones at spawn (resting on the ground)
    goal_distance: Tuple[float, float] = (8.0, 14.0)  # horizontal start-goal distance (m)
    random_heading: bool = True  # random direction of travel per episode

    # waypoints (paper Sec. IV-E)
    waypoint_mode: str = "discrete"  # discrete (paper) | carrot (continuously moving reference)
    waypoint_spacing: float = 1.0
    waypoint_reach_dist: float = 0.6
    carrot_speed: float = 1.0
    carrot_lead: float = 1.5
    target_obs_clip: float = 3.0  # clip the slot vector in the observation (keeps it in-distribution)

    # phases
    form_tolerance: float = 0.35
    form_steps: int = 15
    form_timeout: int = 300  # steps; FORM -> NAV is forced after this
    hold_enter_dist: float = 0.5
    success_dist: float = 0.3
    hold_speed: float = 0.3

    # local sensing (decentralised observation)
    neighbour_radius: float = 3.0
    max_neighbours: int = 7
    obstacle_sensing_range: float = 4.0
    max_obstacles_obs: int = 4

    # obstacles
    scenario: str = "train_mix"  # none | static | dynamic | mixed | train_mix
    scenario_probs: Tuple[float, float, float, float] = (0.2, 0.3, 0.2, 0.3)  # for train_mix
    num_static: Tuple[int, int] = (3, 6)  # (min, max) active pillars; max = number of pillar slots
    num_dynamic: Tuple[int, int] = (1, 3)
    static_radius: Tuple[float, float] = (0.3, 0.6)
    static_height: float = 6.0
    dynamic_radius: Tuple[float, float] = (0.25, 0.4)
    dynamic_speed: Tuple[float, float] = (0.3, 1.0)
    dynamic_half_path: Tuple[float, float] = (2.0, 4.0)
    corridor_half_width: float = 2.5
    obstacle_clearance: float = 2.5  # no obstacles within this distance of start / goal
    randomize_obstacle_radius: bool = False  # Isaac Sim uses fixed prim radii per slot

    # safety / termination
    drone_radius: float = 0.25
    collision_dist: float = 0.35  # drone-drone centre distance counted as a collision
    safe_dist: float = 0.7  # separation below which the avoidance penalty starts
    obstacle_safe_dist: float = 0.6  # obstacle clearance below which the penalty starts
    crash_height: float = 0.2  # below this after `takeoff_grace` steps = crash (must exceed spawn_height)
    takeoff_grace: int = 150  # steps during which being near the ground is not a crash
    max_tilt: float = 1.2  # rad
    out_of_bounds: float = 8.0  # max lateral distance from the path (m)
    max_altitude: float = 5.0
    terminate_on_collision: bool = True

    # actions
    action_mode: str = "dir_speed"  # dir_speed (paper: [dir_xyz, speed]) | velocity
    max_speed: float = 1.5

    # rewards (paper Sec. IV-A plus obstacle / hold terms)
    w_nav: float = 5.0  # progress toward own slot: d(t-1) - d(t)
    w_slot: float = 0.5  # dense proximity to own slot: exp(-d / slot_sigma)
    slot_sigma: float = 0.5
    w_reached: float = 0.2  # d < success_dist
    w_form: float = 1.0  # Procrustes formation reward, staged: only after FORM (Sec. IV-F)
    w_avoid: float = 1.0  # inter-drone separation
    w_obstacle: float = 2.0  # obstacle clearance
    w_tilt: float = 5.0
    tilt_soft: float = 0.35
    w_smooth: float = 0.05  # -||a_t - a_{t-1}||^2
    w_hold: float = 1.0  # at the goal, within success_dist and slower than hold_speed
    w_crash: float = 10.0

    # time step of one environment step (set by the environment)
    dt: float = 0.032

    @classmethod
    def from_dict(cls, d: Dict) -> "FormationNavConfig":
        names = {f.name for f in fields(cls)}
        kwargs = {}
        for k, v in dict(d).items():
            if k not in names:
                continue
            if isinstance(v, list):
                v = tuple(tuple(x) if isinstance(x, list) else x for x in v)
            kwargs[k] = v
        return cls(**kwargs)

    @property
    def action_dim(self) -> int:
        return 4 if self.action_mode == "dir_speed" else 3


# ----------------------------------------------------------------------------------------
# Formation templates
# ----------------------------------------------------------------------------------------
# Convention: the template's +x axis points in the direction of travel, +z is up.


def _plane(n: int) -> np.ndarray:
    side = int(math.ceil(math.sqrt(n)))
    pts = [[i, j, 0.0] for i in range(side) for j in range(side)][:n]
    return np.asarray(pts, dtype=np.float64)


def _cube(n: int) -> np.ndarray:
    side = max(2, int(math.ceil(n ** (1.0 / 3.0))))
    lattice = np.asarray(
        [[x, y, z] for x in range(side) for y in range(side) for z in range(side)],
        dtype=np.float64,
    )
    centre = np.full(3, (side - 1) / 2.0)
    # keep the points farthest from the centre (corners first) so that small swarms
    # still look like a cube
    order = np.argsort(-np.sum((lattice - centre) ** 2, axis=1), kind="mergesort")
    return lattice[order[:n]]


def _sphere(n: int) -> np.ndarray:
    if n == 1:
        return np.zeros((1, 3))
    golden = math.pi * (3.0 - math.sqrt(5.0))
    pts = []
    for i in range(n):
        y = 1.0 - 2.0 * i / (n - 1)
        r = math.sqrt(max(0.0, 1.0 - y * y))
        pts.append([r * math.cos(i * golden), r * math.sin(i * golden), y])
    return np.asarray(pts)


def _pyramid(n: int) -> np.ndarray:
    pts = []
    layer = 0
    while len(pts) < n:
        side = layer + 1
        off = (side - 1) / 2.0
        ring = [[x - off, y - off, -float(layer)] for x in range(side) for y in range(side)]
        # outermost points of a partially filled layer first
        ring.sort(key=lambda p: -(p[0] ** 2 + p[1] ** 2))
        pts.extend(ring[: n - len(pts)])
        layer += 1
    return np.asarray(pts, dtype=np.float64)


def _line(n: int) -> np.ndarray:  # line abreast: perpendicular to the direction of travel
    return np.stack([np.zeros(n), np.arange(n, dtype=np.float64), np.zeros(n)], axis=1)


def _column(n: int) -> np.ndarray:  # single file along the direction of travel
    return np.stack([np.arange(n, dtype=np.float64), np.zeros(n), np.zeros(n)], axis=1)


def _v(n: int) -> np.ndarray:
    pts = [[0.0, 0.0, 0.0]]
    k = 1
    while len(pts) < n:
        pts.append([-float(k), float(k), 0.0])
        if len(pts) < n:
            pts.append([-float(k), -float(k), 0.0])
        k += 1
    return np.asarray(pts)


def _circle(n: int) -> np.ndarray:
    ang = 2.0 * math.pi * np.arange(n) / n
    return np.stack([np.cos(ang), np.sin(ang), np.zeros(n)], axis=1)


_BUILDERS = {
    "plane": _plane,
    "cube": _cube,
    "sphere": _sphere,
    "pyramid": _pyramid,
    "line": _line,
    "column": _column,
    "v": _v,
    "circle": _circle,
}


def formation_template(name: str, n: int, spacing: float) -> torch.Tensor:
    """(n, 3) slot offsets centred at the centroid with minimum pairwise distance `spacing`."""
    if name not in _BUILDERS:
        raise ValueError(f"Unknown formation '{name}'. Choose from {sorted(_BUILDERS)}.")
    pts = _BUILDERS[name](n)
    pts = pts - pts.mean(axis=0, keepdims=True)
    if n > 1:
        d = np.linalg.norm(pts[:, None] - pts[None], axis=-1)
        d[np.diag_indices(n)] = np.inf
        pts = pts * (spacing / d.min())
    return torch.as_tensor(pts, dtype=torch.float32)


def custom_template(points: Sequence[Sequence[float]], n: int) -> torch.Tensor:
    pts = torch.as_tensor(points, dtype=torch.float32)
    if pts.shape != (n, 3):
        raise ValueError(f"custom_formation must have shape ({n}, 3), got {tuple(pts.shape)}")
    return pts - pts.mean(0, keepdim=True)


# ----------------------------------------------------------------------------------------
# Geometry helpers
# ----------------------------------------------------------------------------------------


def rotate_z(v: torch.Tensor, yaw: torch.Tensor) -> torch.Tensor:
    """Rotate vectors v (B, ..., 3) about +z by yaw (B,)."""
    c = torch.cos(yaw).reshape(-1, *([1] * (v.dim() - 2)))
    s = torch.sin(yaw).reshape(-1, *([1] * (v.dim() - 2)))
    x, y, z = v.unbind(-1)
    return torch.stack([c * x - s * y, s * x + c * y, z], dim=-1)


def greedy_assignment(cost: torch.Tensor) -> torch.Tensor:
    """
    Batched greedy linear assignment.

    cost: (B, n, n) cost of giving slot j to drone i.
    Returns perm (B, n) with perm[b, i] = slot of drone i (a permutation).
    """
    B, n, _ = cost.shape
    cost = cost.clone()
    perm = torch.empty(B, n, dtype=torch.long, device=cost.device)
    rows = torch.arange(B, device=cost.device)
    for _ in range(n):
        flat = cost.reshape(B, -1).argmin(dim=1)
        i, j = flat // n, flat % n
        perm[rows, i] = j
        cost[rows, i, :] = float("inf")
        cost[rows, :, j] = float("inf")
    return perm


def procrustes_error(pos: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Mean squared distance between `pos` and `target` after the optimal rigid alignment
    (translation + proper rotation) of the target onto the positions (Kabsch).

    pos, target: (B, n, 3). Returns (B,).
    """
    n = pos.shape[-2]
    P = pos - pos.mean(-2, keepdim=True)
    Q = target - target.mean(-2, keepdim=True)
    H = torch.einsum("bni,bnj->bij", Q, P)
    U, S, Vh = torch.linalg.svd(H)
    d = torch.sign(torch.det(U) * torch.det(Vh))
    d = torch.where(d == 0, torch.ones_like(d), d)
    trace = S[:, 0] + S[:, 1] + d * S[:, 2]
    err = (P.square().sum((-1, -2)) + Q.square().sum((-1, -2)) - 2.0 * trace) / n
    return err.clamp_min(0.0)


def max_pairwise_sq(x: torch.Tensor) -> torch.Tensor:
    """x: (B, n, 3) -> (B,) max squared pairwise distance."""
    return torch.cdist(x, x).square().flatten(1).max(dim=1).values


# ----------------------------------------------------------------------------------------
# Core
# ----------------------------------------------------------------------------------------


class FormationNavCore:
    """
    Batched state machine, obstacle field, reward and observation of FormationNav.

    Typical per-step call order used by the environments:
        core.advance()                         # time += dt, dynamic obstacles move
        ... physics ...
        reward, terminated = core.update(pos, vel, up, actions)
        obs, state = core.observations(pos, vel, heading, up, ang_vel)
    """

    def __init__(
        self,
        cfg: FormationNavConfig,
        num_envs: int,
        device="cpu",
        static_radii: Optional[Sequence[float]] = None,
        dynamic_radii: Optional[Sequence[float]] = None,
        seed: Optional[int] = None,
    ):
        self.cfg = cfg
        self.num_envs = E = num_envs
        self.n = n = cfg.num_drones
        self.device = torch.device(device)
        self.generator = torch.Generator(device=self.device)
        self.generator.manual_seed(0 if seed is None else seed)

        # formations
        if cfg.formation == "custom":
            self.formation_names = ["custom"]
            templates = [custom_template(cfg.custom_formation, n)]
        elif cfg.formation == "dynamic":
            self.formation_names = list(cfg.formation_pool)
            templates = [formation_template(f, n, cfg.formation_spacing) for f in self.formation_names]
        else:
            self.formation_names = [cfg.formation]
            templates = [formation_template(cfg.formation, n, cfg.formation_spacing)]
        self.templates = torch.stack(templates).to(self.device)  # (S, n, 3)
        self.fixed_formation: Optional[int] = None

        # obstacle slots: first the static pillars, then the dynamic spheres
        self.M_s = int(cfg.num_static[1])
        self.M_d = int(cfg.num_dynamic[1])
        self.M = self.M_s + self.M_d
        if static_radii is None:
            static_radii = np.linspace(*cfg.static_radius, self.M_s) if self.M_s else []
        if dynamic_radii is None:
            dynamic_radii = np.linspace(*cfg.dynamic_radius, self.M_d) if self.M_d else []
        self.slot_radii = torch.as_tensor(
            list(static_radii) + list(dynamic_radii), dtype=torch.float32, device=self.device
        )
        self.is_static = torch.zeros(self.M, dtype=torch.bool, device=self.device)
        self.is_static[: self.M_s] = True
        self.fixed_scenario: Optional[int] = None
        self.set_scenario(cfg.scenario)

        f32 = dict(dtype=torch.float32, device=self.device)
        # episode layout
        self.heading = torch.zeros(E, **f32)
        self.direction = torch.zeros(E, 3, **f32)
        self.path_len = torch.ones(E, **f32)
        self.assembly = torch.zeros(E, 3, **f32)
        self.goal = torch.zeros(E, 3, **f32)
        self.formation_id = torch.zeros(E, dtype=torch.long, device=self.device)
        self.slot_offset = torch.zeros(E, n, 3, **f32)  # assigned, rotated slot offsets
        self.form_scale = torch.ones(E, **f32)  # max pairwise squared distance of the formation
        self.scenario_id = torch.zeros(E, dtype=torch.long, device=self.device)

        # phase / waypoint state
        self.phase = torch.zeros(E, dtype=torch.long, device=self.device)
        self.s_ref = torch.zeros(E, **f32)  # arc length of the reference point along the path
        self.form_counter = torch.zeros(E, dtype=torch.long, device=self.device)
        self.steps = torch.zeros(E, dtype=torch.long, device=self.device)
        self.phase_steps = torch.zeros(E, dtype=torch.long, device=self.device)
        self.prev_dist = torch.zeros(E, n, **f32)
        self.prev_action = torch.zeros(E, n, cfg.action_dim, **f32)

        # obstacles
        self.obs_active = torch.zeros(E, self.M, dtype=torch.bool, device=self.device)
        self.obs_radius = self.slot_radii.expand(E, self.M).clone()
        self.obs_anchor = torch.zeros(E, self.M, 3, **f32)  # pillar base / sphere segment centre
        self.obs_axis = torch.zeros(E, self.M, 3, **f32)  # unit direction of motion
        self.obs_half = torch.ones(E, self.M, **f32)
        self.obs_speed = torch.zeros(E, self.M, **f32)
        self.obs_phase = torch.zeros(E, self.M, **f32)
        self.obs_pos = torch.zeros(E, self.M, 3, **f32)
        self.obs_vel = torch.zeros(E, self.M, 3, **f32)

        # stats (running sums and extrema)
        self.stats = {k: torch.zeros(E, 1, **f32) for k in STAT_KEYS}
        self._acc = {k: torch.zeros(E, **f32) for k in ("form_sum", "form_cnt", "slot_sum", "smooth_sum", "hold_good", "hold_cnt")}
        self.last_min_clearance = torch.full((E, n), float("inf"), **f32)
        self.last_min_separation = torch.full((E, n), float("inf"), **f32)

    # ------------------------------------------------------------------------------------
    # dimensions
    # ------------------------------------------------------------------------------------

    @property
    def neighbour_feat_dim(self) -> int:
        return 11  # rel pos 3, rel vel 3, distance 1, desired rel offset 3, valid 1

    @property
    def obstacle_feat_dim(self) -> int:
        return 8  # vector to closest surface point 3, surface distance 1, rel vel 3, valid 1

    @property
    def obs_dim(self) -> int:
        own = 13  # vel 3, heading 3, up 3, angular vel 3, altitude 1
        return (
            own
            + 3  # phase one-hot
            + 3  # vector to own slot (clipped)
            + self.cfg.max_neighbours * self.neighbour_feat_dim
            + self.cfg.max_obstacles_obs * self.obstacle_feat_dim
        )

    @property
    def state_dim(self) -> int:
        return 9 * self.n + 8 * self.M + 6

    # ------------------------------------------------------------------------------------
    # configuration at run time (evaluation)
    # ------------------------------------------------------------------------------------

    def set_scenario(self, scenario: str):
        """none | static | dynamic | mixed | train_mix."""
        if scenario == "train_mix":
            self.fixed_scenario = None
        elif scenario in SCENARIOS:
            self.fixed_scenario = SCENARIOS.index(scenario)
        else:
            raise ValueError(f"Unknown scenario '{scenario}'")

    def set_formation(self, name: Optional[str]):
        """Force one formation from the pool (None: sample from the pool)."""
        self.fixed_formation = None if name is None else self.formation_names.index(name)

    # ------------------------------------------------------------------------------------
    # sampling helpers
    # ------------------------------------------------------------------------------------

    def _rand(self, *shape) -> torch.Tensor:
        return torch.rand(*shape, generator=self.generator, device=self.device)

    def _uniform(self, lo, hi, *shape) -> torch.Tensor:
        return lo + (hi - lo) * self._rand(*shape)

    def _randint(self, lo: int, hi: int, *shape) -> torch.Tensor:
        """Integers in [lo, hi] (inclusive)."""
        return torch.randint(lo, hi + 1, shape, generator=self.generator, device=self.device)

    # ------------------------------------------------------------------------------------
    # reset
    # ------------------------------------------------------------------------------------

    def reset(self, env_ids: torch.Tensor) -> torch.Tensor:
        """Sample new episodes for `env_ids`. Returns drone start positions (B, n, 3)."""
        cfg, n = self.cfg, self.n
        env_ids = env_ids.to(self.device)
        B = env_ids.numel()
        if B == 0:
            return torch.zeros(0, n, 3, device=self.device)

        heading = self._uniform(0.0, 2 * math.pi, B) if cfg.random_heading else torch.zeros(B, device=self.device)
        direction = torch.stack([torch.cos(heading), torch.sin(heading), torch.zeros_like(heading)], -1)
        path_len = self._uniform(*cfg.goal_distance, B)

        # formation and slots
        S = self.templates.shape[0]
        if self.fixed_formation is not None:
            fid = torch.full((B,), self.fixed_formation, dtype=torch.long, device=self.device)
        else:
            fid = self._randint(0, S - 1, B)
        template = rotate_z(self.templates[fid], heading)  # (B, n, 3)
        assembly = torch.zeros(B, 3, device=self.device)
        assembly[:, 2] = cfg.formation_altitude - template[..., 2].min(dim=1).values
        goal = assembly + direction * path_len.unsqueeze(-1)

        # planar take-off grid on the ground, rotated like the formation
        cols = int(math.ceil(math.sqrt(n)))
        idx = torch.arange(n, device=self.device)
        grid = torch.stack(
            [
                (idx // cols).float() - ((n - 1) // cols) / 2.0,
                (idx % cols).float() - (cols - 1) / 2.0,
                torch.zeros(n, device=self.device),
            ],
            -1,
        ) * cfg.ground_spacing
        start = rotate_z(grid.expand(B, n, 3), heading)
        start = start + torch.cat([self._uniform(-0.05, 0.05, B, n, 2), torch.zeros(B, n, 1, device=self.device)], -1)
        start[..., 2] = cfg.spawn_height

        # assign drones to slots (shortest take-off paths, avoids crossing)
        slots_abs = assembly.unsqueeze(1) + template
        cost = torch.cdist(start[..., :2], slots_abs[..., :2]).square()
        perm = greedy_assignment(cost)
        offset = torch.gather(template, 1, perm.unsqueeze(-1).expand(B, n, 3))

        self.heading[env_ids] = heading
        self.direction[env_ids] = direction
        self.path_len[env_ids] = path_len
        self.assembly[env_ids] = assembly
        self.goal[env_ids] = goal
        self.formation_id[env_ids] = fid
        self.slot_offset[env_ids] = offset
        self.form_scale[env_ids] = max_pairwise_sq(offset).clamp_min(1e-6)

        self.phase[env_ids] = PHASE_FORM
        self.s_ref[env_ids] = 0.0
        self.form_counter[env_ids] = 0
        self.steps[env_ids] = 0
        self.phase_steps[env_ids] = 0
        self.prev_action[env_ids] = 0.0

        self._reset_obstacles(env_ids, direction, path_len, assembly, template)
        self._update_obstacle_kinematics(env_ids)

        slot = self.reference(env_ids).unsqueeze(1) + offset
        self.prev_dist[env_ids] = (slot - start).norm(dim=-1)

        for v in self.stats.values():
            v[env_ids] = 0.0
        for v in self._acc.values():
            v[env_ids] = 0.0
        self.stats["min_obstacle_clearance"][env_ids] = cfg.obstacle_sensing_range
        self.stats["min_separation"][env_ids] = 10.0
        self.stats["time_to_form"][env_ids] = -1.0
        self.stats["time_to_goal"][env_ids] = -1.0
        self.stats["scenario"][env_ids] = self.scenario_id[env_ids].float().unsqueeze(-1)
        self.stats["formation_id"][env_ids] = fid.float().unsqueeze(-1)
        self.last_min_clearance[env_ids] = float("inf")
        self.last_min_separation[env_ids] = float("inf")
        return start

    def _reset_obstacles(self, env_ids, direction, path_len, assembly, template):
        cfg = self.cfg
        B = env_ids.numel()
        dev = self.device
        if self.fixed_scenario is not None:
            scenario = torch.full((B,), self.fixed_scenario, dtype=torch.long, device=dev)
        else:
            probs = torch.as_tensor(cfg.scenario_probs, dtype=torch.float32, device=dev)
            scenario = torch.multinomial(probs.expand(B, -1), 1, generator=self.generator).squeeze(-1)
        self.scenario_id[env_ids] = scenario

        has_static = (scenario == 1) | (scenario == 3)
        has_dynamic = (scenario == 2) | (scenario == 3)
        lateral = torch.stack([-direction[:, 1], direction[:, 0], torch.zeros_like(path_len)], -1)
        s_lo = torch.full_like(path_len, cfg.obstacle_clearance)
        s_hi = (path_len - cfg.obstacle_clearance).clamp_min(cfg.obstacle_clearance + 1e-3)

        active = torch.zeros(B, self.M, dtype=torch.bool, device=dev)
        anchor = torch.zeros(B, self.M, 3, device=dev)
        axis = torch.zeros(B, self.M, 3, device=dev)
        half = torch.ones(B, self.M, device=dev)
        speed = torch.zeros(B, self.M, device=dev)
        phase = torch.zeros(B, self.M, device=dev)

        if self.M_s > 0:
            k = self._randint(cfg.num_static[0], cfg.num_static[1], B)
            slot_idx = torch.arange(self.M_s, device=dev)
            active[:, : self.M_s] = has_static.unsqueeze(-1) & (slot_idx < k.unsqueeze(-1))
            # stratified along the path so pillars do not pile up
            u = (slot_idx.float() + self._rand(B, self.M_s)) / self.M_s
            s = s_lo.unsqueeze(-1) + u * (s_hi - s_lo).unsqueeze(-1)
            off = self._uniform(-cfg.corridor_half_width, cfg.corridor_half_width, B, self.M_s)
            pos = assembly.unsqueeze(1) + direction.unsqueeze(1) * s.unsqueeze(-1) + lateral.unsqueeze(1) * off.unsqueeze(-1)
            pos[..., 2] = 0.0  # pillar base on the ground
            anchor[:, : self.M_s] = pos

        if self.M_d > 0:
            k = self._randint(cfg.num_dynamic[0], cfg.num_dynamic[1], B)
            slot_idx = torch.arange(self.M_d, device=dev)
            active[:, self.M_s :] = has_dynamic.unsqueeze(-1) & (slot_idx < k.unsqueeze(-1))
            s = s_lo.unsqueeze(-1) + self._rand(B, self.M_d) * (s_hi - s_lo).unsqueeze(-1)
            off = self._uniform(-1.0, 1.0, B, self.M_d)
            z_lo = cfg.formation_altitude
            z_hi = cfg.formation_altitude + (template[..., 2].max(1).values - template[..., 2].min(1).values)
            z = z_lo + self._rand(B, self.M_d) * (z_hi - z_lo).unsqueeze(-1)
            centre = assembly.unsqueeze(1) + direction.unsqueeze(1) * s.unsqueeze(-1) + lateral.unsqueeze(1) * off.unsqueeze(-1)
            centre[..., 2] = z
            # cross the path at 60-120 degrees
            sign = torch.where(self._rand(B, self.M_d) < 0.5, -1.0, 1.0)
            ang = torch.atan2(direction[:, 1], direction[:, 0]).unsqueeze(-1) + sign * self._uniform(
                math.pi / 3, 2 * math.pi / 3, B, self.M_d
            )
            ax = torch.stack([torch.cos(ang), torch.sin(ang), torch.zeros_like(ang)], -1)
            h = self._uniform(*cfg.dynamic_half_path, B, self.M_d)
            anchor[:, self.M_s :] = centre
            axis[:, self.M_s :] = ax
            half[:, self.M_s :] = h
            speed[:, self.M_s :] = self._uniform(*cfg.dynamic_speed, B, self.M_d)
            phase[:, self.M_s :] = self._rand(B, self.M_d) * 4.0 * h

        radius = self.slot_radii.expand(B, self.M).clone()
        if cfg.randomize_obstacle_radius:
            lo = torch.cat([torch.full((self.M_s,), cfg.static_radius[0]), torch.full((self.M_d,), cfg.dynamic_radius[0])]).to(dev)
            hi = torch.cat([torch.full((self.M_s,), cfg.static_radius[1]), torch.full((self.M_d,), cfg.dynamic_radius[1])]).to(dev)
            radius = lo + (hi - lo) * self._rand(B, self.M)

        self.obs_active[env_ids] = active
        self.obs_anchor[env_ids] = anchor
        self.obs_axis[env_ids] = axis
        self.obs_half[env_ids] = half
        self.obs_speed[env_ids] = speed
        self.obs_phase[env_ids] = phase
        self.obs_radius[env_ids] = radius

    def _update_obstacle_kinematics(self, env_ids: Optional[torch.Tensor] = None):
        """Triangle-wave motion of the dynamic spheres along their segment (closed form)."""
        sl = slice(None) if env_ids is None else env_ids
        t = (self.steps[sl].float() * self.cfg.dt).unsqueeze(-1)
        half = self.obs_half[sl]
        period = 4.0 * half
        x = torch.remainder(self.obs_phase[sl] + self.obs_speed[sl] * t, period)
        u = (x - 2.0 * half).abs() - half  # in [-half, half]
        du = self.obs_speed[sl] * torch.sign(x - 2.0 * half)
        pos = self.obs_anchor[sl] + self.obs_axis[sl] * u.unsqueeze(-1)
        vel = self.obs_axis[sl] * du.unsqueeze(-1)
        static = self.is_static.unsqueeze(0).unsqueeze(-1)
        pos = torch.where(static, self.obs_anchor[sl], pos)
        vel = torch.where(static, torch.zeros_like(vel), vel)
        self.obs_pos[sl] = pos
        self.obs_vel[sl] = vel

    def obstacle_render_poses(self, far_below: float = -50.0) -> torch.Tensor:
        """Centre positions for the obstacle prims (inactive ones are moved out of sight)."""
        pos = self.obs_pos.clone()
        pillar_centre = torch.zeros_like(pos[..., 2])
        pillar_centre[:, : self.M_s] = self.cfg.static_height / 2.0
        pos[..., 2] = torch.where(self.is_static.unsqueeze(0), pillar_centre, pos[..., 2])
        pos[~self.obs_active] = torch.tensor([0.0, 0.0, far_below], device=self.device)
        return pos

    # ------------------------------------------------------------------------------------
    # per-step logic
    # ------------------------------------------------------------------------------------

    def advance(self):
        """Advance time by one environment step (moves the dynamic obstacles)."""
        self.steps += 1
        self.phase_steps += 1
        self._update_obstacle_kinematics()

    def reference(self, env_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        sl = slice(None) if env_ids is None else env_ids
        return self.assembly[sl] + self.direction[sl] * self.s_ref[sl].unsqueeze(-1)

    def slots(self) -> torch.Tensor:
        return self.reference().unsqueeze(1) + self.slot_offset

    def action_to_velocity(self, actions: torch.Tensor) -> torch.Tensor:
        """Map policy actions (E, n, A) to velocity commands (E, n, 3)."""
        a = actions.clamp(-1.0, 1.0)
        if self.cfg.action_mode == "dir_speed":
            direction = a[..., :3] / a[..., :3].norm(dim=-1, keepdim=True).clamp_min(1e-6)
            return direction * a[..., 3:4].abs() * self.cfg.max_speed
        return a[..., :3] * self.cfg.max_speed

    def obstacle_surface(self, pos: torch.Tensor):
        """
        Vector from each drone to the closest point of every obstacle and the signed
        surface distance. pos (E, n, 3) -> vec (E, n, M, 3), dist (E, n, M) (inf if inactive).
        """
        diff = self.obs_pos.unsqueeze(1) - pos.unsqueeze(2)  # (E, n, M, 3)
        r = self.obs_radius.unsqueeze(1)  # (E, 1, M)
        # pillars: horizontal distance to the axis (they are taller than the flight envelope)
        dxy = diff.clone()
        dxy[..., 2] = 0.0
        nxy = dxy.norm(dim=-1)
        cyl_vec = dxy * (1.0 - r / nxy.clamp_min(1e-6)).unsqueeze(-1)
        above = (pos[..., 2].unsqueeze(-1) - self.cfg.static_height).clamp_min(0.0)
        cyl_dist = torch.sqrt((nxy - r).clamp_min(0.0).square() + above.square()) - (r - nxy).clamp_min(0.0)
        # spheres
        nd = diff.norm(dim=-1)
        sph_vec = diff * (1.0 - r / nd.clamp_min(1e-6)).unsqueeze(-1)
        sph_dist = nd - r

        static = self.is_static.view(1, 1, -1)
        vec = torch.where(static.unsqueeze(-1), cyl_vec, sph_vec)
        dist = torch.where(static, cyl_dist, sph_dist)
        active = self.obs_active.unsqueeze(1)
        dist = torch.where(active, dist, torch.full_like(dist, float("inf")))
        vec = torch.where(active.unsqueeze(-1), vec, torch.zeros_like(vec))
        return vec, dist

    @torch.no_grad()
    def update(self, pos: torch.Tensor, vel: torch.Tensor, up: torch.Tensor, actions: torch.Tensor):
        """
        Compute rewards and terminations for the step that just happened, then advance the
        FORM / NAV / HOLD state machine and the waypoints.

        pos, vel, up: (E, n, 3) in the env frame. actions: (E, n, A) raw policy actions.
        Returns reward (E, n, 1) and terminated (E, 1).
        """
        cfg, E, n = self.cfg, self.num_envs, self.n
        actions = actions.clamp(-1.0, 1.0)
        slot = self.slots()
        dist = (slot - pos).norm(dim=-1)  # (E, n)
        nan = torch.isnan(pos).any(-1) | torch.isnan(vel).any(-1)
        pos = torch.nan_to_num(pos)
        vel = torch.nan_to_num(vel)

        # --- reward terms (paper Sec. IV-A) ------------------------------------------------
        r_nav = self.prev_dist - dist
        r_slot = torch.exp(-dist / cfg.slot_sigma)
        r_reached = (dist < cfg.success_dist).float()

        form_err = procrustes_error(pos, slot) / self.form_scale  # (E,)
        staged = (self.phase >= PHASE_NAV).float()
        r_form = -(form_err * staged).unsqueeze(-1).expand(E, n)

        pair = torch.cdist(pos, pos)
        pair = pair + torch.eye(n, device=self.device) * 1e6
        min_sep = pair.min(dim=-1).values  # (E, n)
        r_avoid = -(1.0 - min_sep / cfg.safe_dist).clamp(0.0, 1.0).square()
        drone_collision = min_sep < cfg.collision_dist

        _, surf = self.obstacle_surface(pos)
        min_clear = surf.min(dim=-1).values if self.M > 0 else torch.full((E, n), float("inf"), device=self.device)
        r_obstacle = -(1.0 - min_clear / cfg.obstacle_safe_dist).clamp(0.0, 1.0).square()
        obstacle_collision = min_clear < cfg.drone_radius

        tilt = torch.acos(up[..., 2].clamp(-1.0, 1.0))
        r_tilt = -(tilt - cfg.tilt_soft).clamp_min(0.0).square()

        delta_a = (actions - self.prev_action).square().sum(-1)
        r_smooth = -delta_a

        speed = vel.norm(dim=-1)
        holding = (self.phase == PHASE_HOLD).unsqueeze(-1) & (dist < cfg.success_dist) & (speed < cfg.hold_speed)
        r_hold = holding.float()

        reward = (
            cfg.w_nav * r_nav
            + cfg.w_slot * r_slot
            + cfg.w_reached * r_reached
            + cfg.w_form * r_form
            + cfg.w_avoid * r_avoid
            + cfg.w_obstacle * r_obstacle
            + cfg.w_tilt * r_tilt
            + cfg.w_smooth * r_smooth
            + cfg.w_hold * r_hold
        )

        # --- termination ---------------------------------------------------------------------
        low = (pos[..., 2] < cfg.crash_height) & (self.steps > cfg.takeoff_grace).unsqueeze(-1)
        flipped = tilt > cfg.max_tilt
        rel = pos - self.assembly.unsqueeze(1)
        along = (rel * self.direction.unsqueeze(1)).sum(-1)
        lateral = (rel - self.direction.unsqueeze(1) * along.unsqueeze(-1))[..., :2].norm(dim=-1)
        oob = (lateral > cfg.out_of_bounds) | (pos[..., 2] > cfg.max_altitude) | (along < -cfg.out_of_bounds) | (
            along > self.path_len.unsqueeze(-1) + cfg.out_of_bounds
        )
        crash = low | flipped | oob | nan
        if cfg.terminate_on_collision:
            crash = crash | drone_collision | obstacle_collision
        terminated = crash.any(dim=-1, keepdim=True)
        reward = reward - cfg.w_crash * terminated.float()
        causes = {
            "crash_ground": low,
            "crash_flip": flipped | nan,
            "crash_bounds": oob,
            "crash_drone_collision": drone_collision,
            "crash_obstacle_collision": obstacle_collision,
        }
        for k, v in causes.items():
            self.stats[k] = torch.maximum(self.stats[k], (v.any(-1, keepdim=True) & terminated).float())

        # --- stats ---------------------------------------------------------------------------
        st, acc = self.stats, self._acc
        st["return"] += reward.mean(-1, keepdim=True)
        st["episode_len"] = self.steps.float().unsqueeze(-1)
        acc["form_cnt"] += staged
        acc["form_sum"] += form_err * staged
        st["formation_error"] = (acc["form_sum"] / acc["form_cnt"].clamp_min(1.0)).unsqueeze(-1)
        acc["slot_sum"] += dist.mean(-1)
        st["slot_error"] = (acc["slot_sum"] / self.steps.float().clamp_min(1.0)).unsqueeze(-1)
        centroid = pos.mean(1)
        prog = ((centroid - self.assembly) * self.direction).sum(-1) / self.path_len
        st["progress"] = (prog * (self.phase > PHASE_FORM)).clamp(0.0, 1.0).unsqueeze(-1)
        in_hold = (self.phase == PHASE_HOLD).float()
        acc["hold_cnt"] += in_hold
        acc["hold_good"] += in_hold * holding.all(-1).float()
        st["hold_ratio"] = (acc["hold_good"] / acc["hold_cnt"].clamp_min(1.0)).unsqueeze(-1)
        st["collisions_drone"] += drone_collision.any(-1, keepdim=True).float()
        st["collisions_obstacle"] += obstacle_collision.any(-1, keepdim=True).float()
        st["min_obstacle_clearance"] = torch.minimum(st["min_obstacle_clearance"], min_clear.min(-1, keepdim=True).values.clamp_max(cfg.obstacle_sensing_range))
        st["min_separation"] = torch.minimum(st["min_separation"], min_sep.min(-1, keepdim=True).values)
        acc["smooth_sum"] += delta_a.mean(-1)
        st["smoothness"] = (acc["smooth_sum"] / self.steps.float().clamp_min(1.0)).unsqueeze(-1)
        st["crashed"] = terminated.float()
        self.last_min_clearance = min_clear
        self.last_min_separation = min_sep

        # --- phase machine and waypoints ----------------------------------------------------
        self._advance_phase(pos, dist)

        self.prev_action = actions.clone()
        self.prev_dist = (self.slots() - pos).norm(dim=-1)
        return reward.unsqueeze(-1), terminated

    def _advance_phase(self, pos: torch.Tensor, dist: torch.Tensor):
        cfg = self.cfg
        centroid = pos.mean(1)
        t = self.steps.float() * cfg.dt

        # FORM -> NAV
        forming = self.phase == PHASE_FORM
        in_form = (dist < cfg.form_tolerance).all(-1)
        self.form_counter = torch.where(forming & in_form, self.form_counter + 1, torch.zeros_like(self.form_counter))
        to_nav = forming & ((self.form_counter >= cfg.form_steps) | (self.phase_steps >= cfg.form_timeout))
        self.stats["time_to_form"] = torch.where(
            (to_nav & in_form).unsqueeze(-1), t.unsqueeze(-1), self.stats["time_to_form"]
        )
        self.phase = torch.where(to_nav, torch.full_like(self.phase, PHASE_NAV), self.phase)
        self.phase_steps = torch.where(to_nav, torch.zeros_like(self.phase_steps), self.phase_steps)
        if cfg.waypoint_mode == "discrete":
            first = torch.minimum(torch.full_like(self.path_len, cfg.waypoint_spacing), self.path_len)
            self.s_ref = torch.where(to_nav, first, self.s_ref)

        # NAV: waypoint progression
        nav = (self.phase == PHASE_NAV) & ~to_nav
        gap = (centroid - self.reference()).norm(dim=-1)
        if cfg.waypoint_mode == "discrete":
            reached = nav & (gap < cfg.waypoint_reach_dist) & (self.s_ref < self.path_len)
            self.s_ref = torch.where(reached, torch.minimum(self.s_ref + cfg.waypoint_spacing, self.path_len), self.s_ref)
        else:
            move = nav & (gap < cfg.carrot_lead)
            self.s_ref = torch.where(move, torch.minimum(self.s_ref + cfg.carrot_speed * cfg.dt, self.path_len), self.s_ref)

        # NAV -> HOLD
        at_goal = (centroid - self.goal).norm(dim=-1) < cfg.hold_enter_dist
        to_hold = (self.phase == PHASE_NAV) & (self.s_ref >= self.path_len - 1e-6) & at_goal
        self.stats["time_to_goal"] = torch.where(to_hold.unsqueeze(-1), t.unsqueeze(-1), self.stats["time_to_goal"])
        self.stats["success"] = torch.maximum(self.stats["success"], to_hold.float().unsqueeze(-1))
        self.phase = torch.where(to_hold, torch.full_like(self.phase, PHASE_HOLD), self.phase)
        self.phase_steps = torch.where(to_hold, torch.zeros_like(self.phase_steps), self.phase_steps)

    # ------------------------------------------------------------------------------------
    # observations
    # ------------------------------------------------------------------------------------

    @torch.no_grad()
    def observations(
        self,
        pos: torch.Tensor,
        vel: torch.Tensor,
        heading: torch.Tensor,
        up: torch.Tensor,
        ang_vel: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Decentralised observation per drone (E, n, obs_dim) and centralised critic state
        (E, state_dim). Only relative / local quantities enter the actor observation (no
        absolute x, y), so long routes stay in the training distribution.
        """
        cfg, E, n = self.cfg, self.num_envs, self.n
        pos = torch.nan_to_num(pos)
        vel = torch.nan_to_num(vel)
        ref = self.reference()
        slot = ref.unsqueeze(1) + self.slot_offset
        to_slot = slot - pos
        norm = to_slot.norm(dim=-1, keepdim=True)
        to_slot = to_slot * (cfg.target_obs_clip / norm.clamp_min(cfg.target_obs_clip))
        phase = torch.nn.functional.one_hot(self.phase, 3).float().unsqueeze(1).expand(E, n, 3)
        own = torch.cat([vel, heading, up, ang_vel, pos[..., 2:3]], -1)

        obs = torch.cat(
            [own, phase, to_slot, self._neighbour_features(pos, vel), self._obstacle_features(pos, vel)],
            dim=-1,
        )

        drone_state = torch.cat([pos - ref.unsqueeze(1), vel, slot - pos], -1).reshape(E, -1)
        active = self.obs_active.float().unsqueeze(-1)
        obstacle_state = torch.cat(
            [
                (self.obs_pos - ref.unsqueeze(1)) * active,
                self.obs_vel * active,
                self.obs_radius.unsqueeze(-1) * active,
                active,
            ],
            -1,
        ).reshape(E, -1)
        state = torch.cat(
            [drone_state, obstacle_state, torch.nn.functional.one_hot(self.phase, 3).float(), self.goal - ref], -1
        )
        return obs, state

    def _neighbour_features(self, pos: torch.Tensor, vel: torch.Tensor) -> torch.Tensor:
        """Adaptive (radius-based) neighbour selection, closest `max_neighbours`, zero padded."""
        cfg, E, n = self.cfg, self.num_envs, self.n
        K = cfg.max_neighbours
        out = torch.zeros(E, n, K, self.neighbour_feat_dim, device=self.device)
        k = min(K, n - 1)
        if k <= 0:
            return out.reshape(E, n, -1)
        rel = pos.unsqueeze(1) - pos.unsqueeze(2)  # [e, i, j] = p_j - p_i
        d = rel.norm(dim=-1)
        d = d + torch.eye(n, device=self.device) * 1e6
        d = torch.where(d <= cfg.neighbour_radius, d, torch.full_like(d, float("inf")))
        vals, idx = torch.topk(d, k, dim=-1, largest=False)
        valid = torch.isfinite(vals).unsqueeze(-1).float()
        g3 = idx.unsqueeze(-1).expand(E, n, k, 3)
        rel_pos = torch.gather(rel, 2, g3)
        rel_vel = torch.gather(vel.unsqueeze(1).expand(E, n, n, 3), 2, g3) - vel.unsqueeze(2)
        off = self.slot_offset
        rel_off = torch.gather(off.unsqueeze(1).expand(E, n, n, 3), 2, g3) - off.unsqueeze(2)
        feats = torch.cat([rel_pos, rel_vel, torch.nan_to_num(vals, posinf=0.0).unsqueeze(-1), rel_off, torch.ones_like(valid)], -1)
        out[:, :, :k] = feats * valid
        return out.reshape(E, n, -1)

    def _obstacle_features(self, pos: torch.Tensor, vel: torch.Tensor) -> torch.Tensor:
        cfg, E, n = self.cfg, self.num_envs, self.n
        K = cfg.max_obstacles_obs
        out = torch.zeros(E, n, K, self.obstacle_feat_dim, device=self.device)
        k = min(K, self.M)
        if k <= 0:
            return out.reshape(E, n, -1)
        vec, dist = self.obstacle_surface(pos)
        dist = torch.where(dist <= cfg.obstacle_sensing_range, dist, torch.full_like(dist, float("inf")))
        vals, idx = torch.topk(dist, k, dim=-1, largest=False)
        valid = torch.isfinite(vals).unsqueeze(-1).float()
        sel_vec = torch.gather(vec, 2, idx.unsqueeze(-1).expand(E, n, k, 3))
        obs_vel = torch.gather(self.obs_vel.unsqueeze(1).expand(E, n, self.M, 3), 2, idx.unsqueeze(-1).expand(E, n, k, 3))
        rel_vel = obs_vel - vel.unsqueeze(2)
        feats = torch.cat([sel_vec, torch.nan_to_num(vals, posinf=0.0).unsqueeze(-1), rel_vel, torch.ones_like(valid)], -1)
        out[:, :, :k] = feats * valid
        return out.reshape(E, n, -1)
