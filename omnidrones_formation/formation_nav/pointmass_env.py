"""
FormationNavLite: the FormationNav task with point-mass drones (pure PyTorch, no Isaac Sim).

It has the same observation / action / reward / stats specs as the Isaac Sim environment and
uses the same `FormationNavCore`, so it can be used to
  * debug the task and reward design on a laptop CPU,
  * smoke-test and pre-train the MAPPO-LSTM policy.

Dynamics: first-order velocity tracking with acceleration limit (a crude stand-in for the
Lee controller + quadrotor), gravity is ignored except for the ground contact, and the tilt
is approximated from the horizontal acceleration.
"""

from typing import Optional

import torch
from tensordict import TensorDict, TensorDictBase
from torchrl.data import (
    BoundedTensorSpec,
    CompositeSpec,
    DiscreteTensorSpec,
    UnboundedContinuousTensorSpec,
)
from torchrl.envs import EnvBase

from .core import STAT_KEYS, FormationNavConfig, FormationNavCore

GRAVITY = 9.81


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
    ):
        super().__init__(device=device, batch_size=[num_envs])
        self.cfg = cfg
        self.num_envs = num_envs
        self.max_episode_length = max_episode_length
        self.tau = tau
        self.max_acc = max_acc
        self.core = FormationNavCore(
            cfg, num_envs, device, seed=seed,
        )
        n = cfg.num_drones
        self.pos = torch.zeros(num_envs, n, 3, device=self.device)
        self.vel = torch.zeros(num_envs, n, 3, device=self.device)
        self.acc = torch.zeros(num_envs, n, 3, device=self.device)
        self.progress = torch.zeros(num_envs, dtype=torch.long, device=self.device)
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
        n = self.cfg.num_drones
        up = torch.cat([-self.acc[..., :2], torch.full_like(self.acc[..., :1], GRAVITY)], -1)
        up = up / up.norm(dim=-1, keepdim=True)
        heading = torch.tensor([1.0, 0.0, 0.0], device=self.device).expand(self.num_envs, n, 3)
        return heading, up

    def _observe(self) -> TensorDict:
        heading, up = self._attitude()
        obs, state = self.core.observations(self.pos, self.vel, heading, up, torch.zeros_like(self.vel))
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

    def _set_seed(self, seed: Optional[int]):
        if seed is not None:
            self.core.generator.manual_seed(seed)
            torch.manual_seed(seed)
