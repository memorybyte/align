"""
FormationNav: OmniDrones / Isaac Sim environment.

A swarm of quadrotors takes off from the ground, builds a given formation, flies it to a
goal through static and/or dynamic obstacles (or none), and holds the formation there.

The task logic lives in `core.py` (pure PyTorch); this class only
  * spawns the drones, pillars and moving spheres,
  * converts the policy's velocity commands to rotor commands with the Lee controller,
  * reads the drone state and hands it to the core.

Importing this module registers the environment in `IsaacEnv.REGISTRY` under the name
"FormationNav". It must be imported after the Isaac Sim app is running.
"""

from typing import List, Optional

import torch
from omegaconf import OmegaConf
from tensordict.tensordict import TensorDict, TensorDictBase
from torchrl.data import BoundedTensorSpec, CompositeSpec, UnboundedContinuousTensorSpec

from omni.isaac.core.utils.viewports import set_camera_view
from pxr import Gf, UsdGeom

import omni_drones.utils.kit as kit_utils
import omni_drones.utils.scene as scene_utils
from omni_drones.envs.isaac_env import AgentSpec, IsaacEnv
from omni_drones.envs.utils import create_obstacle
from omni_drones.robots.drone import MultirotorBase
from omni_drones.views import RigidPrimView

from .core import STAT_KEYS, FormationNavConfig, FormationNavCore

PILLAR_COLOR = (0.55, 0.55, 0.6)
MOVER_COLOR = (0.95, 0.45, 0.1)


def _color(prim, rgb):
    UsdGeom.Gprim(prim).CreateDisplayColorAttr().Set([Gf.Vec3f(*rgb)])


class FormationNav(IsaacEnv):
    r"""
    Take off -> build formation -> waypoint navigation with obstacle avoidance -> hold.

    ## Observation (per drone, decentralised)
    velocity, heading, up, angular velocity, altitude, phase one-hot, vector to its
    formation slot (clipped), the closest `max_neighbours` drones within
    `neighbour_radius` (relative position / velocity / distance / desired relative offset /
    valid flag) and the closest `max_obstacles_obs` obstacles within
    `obstacle_sensing_range` (vector to the closest surface point / distance / relative
    velocity / valid flag).

    ## State (centralised critic)
    every drone's position relative to the formation reference, velocity and slot error,
    every obstacle slot, the phase and the vector to the goal.

    ## Action
    `dir_speed` (paper): [dir_x, dir_y, dir_z, speed] in [-1, 1] ->
    v = max_speed * |speed| * dir / |dir|, tracked by the Lee position controller.
    `velocity`: [v_x, v_y, v_z] * max_speed.

    ## Reward
    progress to own slot, slot proximity, reaching bonus, Procrustes formation error
    (staged: only after the formation is built), inter-drone separation, obstacle clearance,
    tilt, action smoothness, holding at the goal, crash penalty. See `core.py`.

    ## Episode end
    terminated on crash (ground after take-off grace, flip, out of bounds, NaN) and, if
    `terminate_on_collision`, on drone-drone / drone-obstacle collision; truncated at
    `max_episode_length`.
    """

    def __init__(self, cfg, headless):
        task_cfg = OmegaConf.to_container(cfg.task, resolve=True)
        self.fn_cfg = FormationNavConfig.from_dict(task_cfg)
        self.fn_cfg.dt = cfg.sim.dt * cfg.sim.substeps
        self.obstacle_collision = bool(task_cfg.get("obstacle_physics_collision", False))
        self.follow_camera = bool(task_cfg.get("follow_camera", True))
        super().__init__(cfg, headless)

        self.drone.initialize()
        self.static_view: Optional[RigidPrimView] = None
        self.dynamic_view: Optional[RigidPrimView] = None
        if self.core.M_s > 0:
            self.static_view = RigidPrimView(
                "/World/envs/env_*/pillar_*", reset_xform_properties=False, shape=[self.num_envs, -1]
            )
            self.static_view.initialize()
        if self.core.M_d > 0:
            self.dynamic_view = RigidPrimView(
                "/World/envs/env_*/mover_*", reset_xform_properties=False, shape=[self.num_envs, -1]
            )
            self.dynamic_view.initialize()

        n = self.fn_cfg.num_drones
        self.identity_rot = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device).expand(self.num_envs, n, 4)
        self.target_yaw = torch.zeros(self.num_envs, n, 1, device=self.device)
        self._reward = torch.zeros(self.num_envs, n, 1, device=self.device)
        self._terminated = torch.zeros(self.num_envs, 1, dtype=torch.bool, device=self.device)

    # ------------------------------------------------------------------------------------
    # scene
    # ------------------------------------------------------------------------------------

    def _design_scene(self) -> Optional[List[str]]:
        cfg = self.fn_cfg
        drone_model_cfg = self.cfg.task.drone_model
        self.drone, self.controller = MultirotorBase.make(
            drone_model_cfg.name, drone_model_cfg.controller, device=self.device
        )
        if self.controller is None:
            raise ValueError("FormationNav needs a velocity-capable controller, e.g. LeePositionController.")

        scene_utils.design_scene()

        # placeholder spawn; every episode starts from `_reset_idx`
        self.drone.spawn(
            translations=[(i * cfg.ground_spacing, 0.0, cfg.spawn_height) for i in range(cfg.num_drones)]
        )

        self.core = FormationNavCore(cfg, self.num_envs, self.device)
        radii = self.core.slot_radii.tolist()
        for j in range(self.core.M_s):
            prim = create_obstacle(
                f"/World/envs/env_0/pillar_{j}",
                prim_type="Cylinder",
                translation=(0.0, 0.0, -50.0),
                attributes={"radius": radii[j], "height": cfg.static_height, "axis": "Z"},
            )
            _color(prim, PILLAR_COLOR)
            if not self.obstacle_collision:
                kit_utils.set_collision_properties(prim.GetPath().pathString, collision_enabled=False)
        for j in range(self.core.M_d):
            prim = create_obstacle(
                f"/World/envs/env_0/mover_{j}",
                prim_type="Sphere",
                translation=(0.0, 0.0, -50.0),
                attributes={"radius": radii[self.core.M_s + j]},
            )
            _color(prim, MOVER_COLOR)
            if not self.obstacle_collision:
                kit_utils.set_collision_properties(prim.GetPath().pathString, collision_enabled=False)
        return ["/World/defaultGroundPlane"]

    def _set_specs(self):
        n = self.fn_cfg.num_drones
        M = self.core.M
        observation_spec = CompositeSpec({
            "agents": CompositeSpec({
                "observation": UnboundedContinuousTensorSpec((n, self.core.obs_dim)),
                "observation_central": UnboundedContinuousTensorSpec((self.core.state_dim,)),
            }),
            # not used by the policy; handy for plots and evaluation
            "info": CompositeSpec({
                "drone_pos": UnboundedContinuousTensorSpec((n, 3)),
                "slot_pos": UnboundedContinuousTensorSpec((n, 3)),
                "goal_pos": UnboundedContinuousTensorSpec((3,)),
                "obstacle_pos": UnboundedContinuousTensorSpec((M, 3)),
                "obstacle_radius": UnboundedContinuousTensorSpec((M,)),
                "obstacle_active": UnboundedContinuousTensorSpec((M,)),
                "phase": UnboundedContinuousTensorSpec((1,)),
                "route": UnboundedContinuousTensorSpec((self.core.Kp, 3)),
            }),
        }).expand(self.num_envs).to(self.device)
        self.observation_spec = observation_spec
        self.action_spec = CompositeSpec({
            "agents": CompositeSpec({
                "action": BoundedTensorSpec(-1.0, 1.0, (n, self.fn_cfg.action_dim)),
            })
        }).expand(self.num_envs).to(self.device)
        self.reward_spec = CompositeSpec({
            "agents": CompositeSpec({"reward": UnboundedContinuousTensorSpec((n, 1))})
        }).expand(self.num_envs).to(self.device)
        self.agent_spec["drone"] = AgentSpec(
            "drone",
            n,
            observation_key=("agents", "observation"),
            action_key=("agents", "action"),
            reward_key=("agents", "reward"),
            state_key=("agents", "observation_central"),
        )
        stats_spec = CompositeSpec(
            {k: UnboundedContinuousTensorSpec(1) for k in STAT_KEYS}
        ).expand(self.num_envs).to(self.device)
        self.observation_spec["stats"] = stats_spec
        self.stats = stats_spec.zero()

    # ------------------------------------------------------------------------------------
    # evaluation helpers
    # ------------------------------------------------------------------------------------

    def set_scenario(self, scenario: str):
        """none | static | dynamic | mixed | train_mix (applies from the next reset)."""
        self.core.set_scenario(scenario)

    def set_formation(self, name: Optional[str]):
        self.core.set_formation(name)

    # ------------------------------------------------------------------------------------
    # simulation loop
    # ------------------------------------------------------------------------------------

    def _reset_idx(self, env_ids: torch.Tensor):
        self.drone._reset_idx(env_ids, self.training)
        start = self.core.reset(env_ids)
        self.drone.set_world_poses(
            start + self.envs_positions[env_ids].unsqueeze(1), self.identity_rot[env_ids], env_ids
        )
        self.drone.set_velocities(torch.zeros(len(env_ids), self.fn_cfg.num_drones, 6, device=self.device), env_ids)
        poses = self.core.obstacle_render_poses()
        if self.static_view is not None:
            self.static_view.set_world_poses(
                poses[env_ids, : self.core.M_s] + self.envs_positions[env_ids].unsqueeze(1), env_indices=env_ids
            )
        if self.dynamic_view is not None:
            self.dynamic_view.set_world_poses(
                poses[env_ids, self.core.M_s :] + self.envs_positions[env_ids].unsqueeze(1), env_indices=env_ids
            )
        self._terminated[env_ids] = False

    def _step(self, tensordict: TensorDictBase) -> TensorDictBase:
        # once per environment step: advance time and move the dynamic obstacles
        self.core.advance()
        if self.dynamic_view is not None:
            poses = self.core.obstacle_render_poses()[:, self.core.M_s :]
            self.dynamic_view.set_world_poses(poses + self.envs_positions.unsqueeze(1))
        return super()._step(tensordict)

    def _pre_sim_step(self, tensordict: TensorDictBase):
        # called every physics substep: close the velocity loop at the simulation rate
        actions = tensordict[("agents", "action")]
        target_vel = self.core.action_to_velocity(actions)
        self.drone.get_state()
        root_state = torch.cat([self.drone.pos, self.drone.rot, self.drone.vel_w[..., :6]], dim=-1)
        rotor_cmds = self.controller.compute(root_state, target_vel=target_vel, target_yaw=self.target_yaw)
        self.effort = self.drone.apply_action(rotor_cmds)

    def _post_sim_step(self, tensordict: TensorDictBase):
        self.drone.get_state()
        self._reward, self._terminated = self.core.update(
            pos=self.drone.pos,
            vel=self.drone.vel_w[..., :3],
            up=self.drone.up,
            actions=tensordict[("agents", "action")],
        )

    def _compute_state_and_obs(self):
        self.drone.get_state()
        obs, state = self.core.observations(
            pos=self.drone.pos,
            vel=self.drone.vel_w[..., :3],
            heading=self.drone.heading,
            up=self.drone.up,
            ang_vel=self.drone.vel_w[..., 3:6],
        )
        for k in STAT_KEYS:
            self.stats[k] = self.core.stats[k]

        if self._should_render(0):
            self._debug_vis()

        info = {
            "drone_pos": self.drone.pos.clone(),
            "slot_pos": self.core.slots(),
            "goal_pos": self.core.goal.clone(),
            "obstacle_pos": self.core.obs_pos.clone(),
            "obstacle_radius": self.core.obs_radius.clone(),
            "obstacle_active": self.core.obs_active.float(),
            "phase": self.core.phase.float().unsqueeze(-1),
            "route": self.core.path_pts.clone(),
        }
        return TensorDict(
            {
                "agents": {"observation": obs, "observation_central": state},
                "info": info,
                "stats": self.stats.clone(),
            },
            self.batch_size,
        )

    def _compute_reward_and_done(self):
        truncated = (self.progress_buf >= self.max_episode_length).unsqueeze(-1)
        terminated = self._terminated
        return TensorDict(
            {
                "agents": {"reward": self._reward},
                "done": terminated | truncated,
                "terminated": terminated,
                "truncated": truncated,
            },
            self.batch_size,
        )

    # ------------------------------------------------------------------------------------
    # visualisation (central env only)
    # ------------------------------------------------------------------------------------

    def _debug_vis(self):
        i = int(self.central_env_idx)
        origin = self.envs_positions[i].cpu()
        core = self.core
        self.debug_draw.clear()
        # route of the formation reference (straight line or planned around the pillars)
        g = core.goal[i].cpu()
        path = core.path_pts[i].cpu() + origin
        self.debug_draw.plot(path, size=2.0, color=(0.2, 0.8, 0.2, 1.0))
        # current formation slots
        slots = core.slots()[i].cpu() + origin
        draw = self.debug_draw._draw
        draw.draw_points(slots.tolist(), [(0.1, 0.4, 1.0, 1.0)] * slots.shape[0], [12.0] * slots.shape[0])
        draw.draw_points([(g + origin).tolist()], [(1.0, 0.1, 0.1, 1.0)], [20.0])
        if self.follow_camera:
            centroid = self.drone.pos[i].mean(0).cpu() + origin
            set_camera_view(
                eye=(centroid + torch.as_tensor(self.cfg.viewer.eye)).tolist(),
                target=(centroid + torch.as_tensor(self.cfg.viewer.lookat)).tolist(),
            )
