"""
FormationNavLite: the FormationNav task without Isaac Sim (pure PyTorch).

It has the same observation / action / reward / stats specs as the Isaac Sim environment and
uses the same `FormationNavCore`, so it can be used to
  * debug the task and reward design on a laptop CPU,
  * smoke-test and pre-train the MAPPO-LSTM policy.

Two dynamics backends:
  * "pointmass" (default): first-order velocity tracking with acceleration limit (a crude
    stand-in for the Lee controller + quadrotor); gravity is ignored except for the ground
    contact and the tilt is approximated from the horizontal acceleration.
  * "quadrotor": the rigid-body model of `quadrotor.py` (OmniDrones' rotor, force and downwash
    models, the simulated asset's mass and arm length) flown by the same Lee controller and
    downwash feed-forward as the Isaac Sim env, at the same physics rate (substeps).
"""

import math
import os
from typing import Optional

import torch
import yaml
from tensordict import TensorDict, TensorDictBase
from torchrl.data import (
    BoundedTensorSpec,
    CompositeSpec,
    DiscreteTensorSpec,
    UnboundedContinuousTensorSpec,
)
from torchrl.envs import EnvBase

from .controller import LeeVelocityController, quat_rotate
from .core import STAT_KEYS, FormationNavConfig, FormationNavCore
from .quadrotor import OMNIDRONES_ASSETS, QuadrotorModel

GRAVITY = 9.81
ASSET_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")


def load_uav_params(drone: str) -> dict:
    """Parameter yaml of an OmniDrones drone shipped in formation_nav/assets (crazyflie, hummingbird)."""
    with open(os.path.join(ASSET_DIR, f"{drone.lower()}.yaml")) as f:
        return yaml.safe_load(f)


class FormationNavLite(EnvBase):
    def __init__(
        self,
        cfg: FormationNavConfig,
        num_envs: int = 64,
        max_episode_length: int = 800,
        device="cpu",
        tau: float = 0.15,
        max_acc: float = 6.0,
        seed: Optional[int] = None,
        dynamics: str = "pointmass",
        drone: str = "crazyflie",
        substeps: int = 2,
        downwash_scale: float = 1.0,
        downwash_feedforward: bool = True,
    ):
        """
        tau, max_acc: point-mass dynamics. drone, substeps, downwash_scale (OmniDrones' model,
        0 = off), downwash_feedforward: quadrotor dynamics (physics dt = cfg.dt / substeps, as in
        Isaac Sim: 0.032 / 2).
        """
        super().__init__(device=device, batch_size=[num_envs])
        if dynamics not in ("pointmass", "quadrotor"):
            raise ValueError(f"dynamics must be 'pointmass' or 'quadrotor', got {dynamics!r}")
        self.cfg = cfg
        self.num_envs = num_envs
        self.max_episode_length = max_episode_length
        self.tau = tau
        self.max_acc = max_acc
        self.dynamics = dynamics
        self.core = FormationNavCore(
            cfg, num_envs, device, seed=seed,
        )
        n = cfg.num_drones
        self.pos = torch.zeros(num_envs, n, 3, device=self.device)
        self.vel = torch.zeros(num_envs, n, 3, device=self.device)
        self.acc = torch.zeros(num_envs, n, 3, device=self.device)
        self.progress = torch.zeros(num_envs, dtype=torch.long, device=self.device)
        if dynamics == "quadrotor":
            params = load_uav_params(drone)
            asset = OMNIDRONES_ASSETS.get(params["name"], {})
            self.substeps = int(substeps)
            self.downwash_feedforward = downwash_feedforward
            self.quad = QuadrotorModel(
                params, (num_envs, n), cfg.dt / self.substeps, device=self.device,
                downwash=downwash_scale > 0, downwash_scale=downwash_scale, **asset,
            )
            self.controller = LeeVelocityController(params, mass=self.quad.mass).to(self.device)
            # OmniDrones resets the rotors to the hover throttle
            self.hover_throttle = math.sqrt(self.quad.mass * GRAVITY / float(self.quad.max_thrust.sum()))
            self.target_yaw = torch.zeros(num_envs, n, 1, device=self.device)
        self._make_specs()

    def _make_specs(self):
        n, M = self.cfg.num_drones, self.core.M
        self.observation_spec = CompositeSpec(
            {
                "agents": CompositeSpec(
                    {
                        "observation": UnboundedContinuousTensorSpec((n, self.core.obs_dim)),
                        "observation_central": UnboundedContinuousTensorSpec((self.core.state_dim,)),
                    }
                ),
                "info": CompositeSpec(
                    {
                        "drone_pos": UnboundedContinuousTensorSpec((n, 3)),
                        "slot_pos": UnboundedContinuousTensorSpec((n, 3)),
                        "goal_pos": UnboundedContinuousTensorSpec((3,)),
                        "obstacle_pos": UnboundedContinuousTensorSpec((M, 3)),
                        "obstacle_radius": UnboundedContinuousTensorSpec((M,)),
                        "obstacle_active": UnboundedContinuousTensorSpec((M,)),
                        "phase": UnboundedContinuousTensorSpec((1,)),
                        "route": UnboundedContinuousTensorSpec((self.core.Kp, 3)),
                    }
                ),
                "stats": CompositeSpec({k: UnboundedContinuousTensorSpec(1) for k in STAT_KEYS}),
            }
        ).expand(self.num_envs).to(self.device)
        self.action_spec = CompositeSpec(
            {"agents": CompositeSpec({"action": BoundedTensorSpec(-1.0, 1.0, (n, self.cfg.action_dim))})}
        ).expand(self.num_envs).to(self.device)
        self.reward_spec = CompositeSpec(
            {"agents": CompositeSpec({"reward": UnboundedContinuousTensorSpec((n, 1))})}
        ).expand(self.num_envs).to(self.device)
        self.done_spec = CompositeSpec(
            {
                "done": DiscreteTensorSpec(2, (1,), dtype=torch.bool),
                "terminated": DiscreteTensorSpec(2, (1,), dtype=torch.bool),
                "truncated": DiscreteTensorSpec(2, (1,), dtype=torch.bool),
            }
        ).expand(self.num_envs).to(self.device)

    # evaluation helpers, same as the Isaac Sim env
    def set_scenario(self, scenario: str):
        self.core.set_scenario(scenario)

    def set_formation(self, name):
        self.core.set_formation(name)

    def _attitude(self):
        if self.dynamics == "quadrotor":
            return self.quad.axes()
        n = self.cfg.num_drones
        up = torch.cat([-self.acc[..., :2], torch.full_like(self.acc[..., :1], GRAVITY)], -1)
        up = up / up.norm(dim=-1, keepdim=True)
        heading = torch.tensor([1.0, 0.0, 0.0], device=self.device).expand(self.num_envs, n, 3)
        return heading, up

    def _observe(self) -> TensorDict:
        heading, up = self._attitude()
        if self.dynamics == "quadrotor":
            ang_vel = quat_rotate(self.quad.rot, self.quad.omega)
        else:
            ang_vel = torch.zeros_like(self.vel)
        obs, state = self.core.observations(self.pos, self.vel, heading, up, ang_vel)
        return TensorDict(
            {
                "agents": {"observation": obs, "observation_central": state},
                "info": {
                    "drone_pos": self.pos.clone(),
                    "slot_pos": self.core.slots(),
                    "goal_pos": self.core.goal.clone(),
                    "obstacle_pos": self.core.obs_pos.clone(),
                    "obstacle_radius": self.core.obs_radius.clone(),
                    "obstacle_active": self.core.obs_active.float(),
                    "phase": self.core.phase.float().unsqueeze(-1),
                    "route": self.core.path_pts.clone(),
                },
                "stats": {k: v.clone() for k, v in self.core.stats.items()},
            },
            self.batch_size,
            device=self.device,
        )

    def _reset(self, tensordict: Optional[TensorDictBase] = None, **kwargs) -> TensorDictBase:
        if tensordict is not None and "_reset" in tensordict.keys():
            mask = tensordict.get("_reset").reshape(self.num_envs)
        else:
            mask = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        env_ids = mask.nonzero().squeeze(-1)
        start = self.core.reset(env_ids)
        start[..., 2] = 0.0  # resting on the ground
        self.pos[env_ids] = start
        self.vel[env_ids] = 0.0
        self.acc[env_ids] = 0.0
        self.progress[env_ids] = 0
        if self.dynamics == "quadrotor":
            self.quad.reset(env_ids, start)
            self.quad.throttle[env_ids] = self.hover_throttle
            self.pos, self.vel = self.quad.pos, self.quad.vel
        out = self._observe()
        false = torch.zeros(self.num_envs, 1, dtype=torch.bool, device=self.device)
        out.set("done", false.clone())
        out.set("terminated", false.clone())
        out.set("truncated", false.clone())
        return out

    def _step(self, tensordict: TensorDictBase) -> TensorDictBase:
        actions = tensordict.get(("agents", "action"))
        dt = self.cfg.dt
        self.core.advance()
        v_cmd = self.core.action_to_velocity(actions)
        if self.dynamics == "quadrotor":
            self._quadrotor_step(v_cmd)
        else:
            acc = ((v_cmd - self.vel) / self.tau)
            acc = acc * (self.max_acc / acc.norm(dim=-1, keepdim=True).clamp_min(self.max_acc))
            self.acc = acc
            self.vel = self.vel + acc * dt
            self.pos = self.pos + self.vel * dt
            on_ground = self.pos[..., 2] <= 0.0
            self.pos[..., 2] = self.pos[..., 2].clamp_min(0.0)
            self.vel[..., 2] = torch.where(on_ground, self.vel[..., 2].clamp_min(0.0), self.vel[..., 2])

        _, up = self._attitude()
        reward, terminated = self.core.update(self.pos, self.vel, up, actions)
        self.progress += 1
        truncated = (self.progress >= self.max_episode_length).unsqueeze(-1)
        out = self._observe()
        out.set(("agents", "reward"), reward)
        out.set("terminated", terminated)
        out.set("truncated", truncated)
        out.set("done", terminated | truncated)
        return out

    def _quadrotor_step(self, v_cmd: torch.Tensor):
        """Velocity command -> Lee controller -> rotor model at the physics rate (like env.py)."""
        quad = self.quad
        for _ in range(self.substeps):
            target_acc = -quad.external_force / self.controller.mass if self.downwash_feedforward else None
            cmds = self.controller.compute(
                quad.root_state(), target_vel=v_cmd, target_acc=target_acc, target_yaw=self.target_yaw
            )
            quad.step(cmds)
        self.pos, self.vel = quad.pos, quad.vel

    def _set_seed(self, seed: Optional[int]):
        if seed is not None:
            self.core.generator.manual_seed(seed)
            torch.manual_seed(seed)
