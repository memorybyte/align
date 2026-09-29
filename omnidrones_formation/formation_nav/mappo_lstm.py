"""
MAPPO with the paper's FC-LSTM-FC actor and centralised FC-LSTM-FC critic, for TorchRL /
OmniDrones environments.

Actor  (shared by all drones, decentralised):
    LayerNorm(obs) -> Linear 256 -> ReLU -> LayerNorm -> LSTM 256 -> LayerNorm -> Linear A
    Gaussian head with a learned, state-independent log-std.
Critic (centralised, training only):
    LayerNorm(state) -> Linear 256 -> ReLU -> LayerNorm -> LSTM 256 -> LayerNorm -> Linear n
    one value per drone (the rewards are per drone).

Recurrent state is carried through the TorchRL collector in the tensordict:
    "actor_h", "actor_c"   (num_envs, n, H)
    "critic_h", "critic_c" (num_envs, H)
The policy writes the updated state under ("next", key); the collector's step_mdp moves it
to the root for the next step. `is_init` (from the InitTracker transform) resets the state
at the first step of every episode.

PPO updates unroll both LSTMs over chunks of `seq_len` steps starting from the stored
state and resetting at `is_init`, so the log-probs in the importance ratio are computed with
exactly the memory used during the rollout (ratio == 1 before the first gradient step).

Time-limit truncation is handled correctly: every step bootstraps from the critic's value
of its own next observation (computed with the stored "next" hidden state), and only
`terminated` cuts the bootstrap, while `done` cuts the GAE recursion.
"""

from dataclasses import dataclass
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from tensordict import TensorDictBase
from tensordict.nn import TensorDictModuleBase
from torchrl.data import CompositeSpec, TensorSpec
from torchrl.envs.utils import ExplorationType, exploration_type

OBS_KEY = ("agents", "observation")
STATE_KEY = ("agents", "observation_central")
ACTION_KEY = ("agents", "action")
REWARD_KEY = ("agents", "reward")
HIDDEN_KEYS = ("actor_h", "actor_c", "critic_h", "critic_c")


@dataclass
class MAPPOLSTMConfig:
    name: str = "mappo_lstm"
    train_every: int = 64  # rollout length per environment
    ppo_epochs: int = 4
    num_minibatches: int = 8
    seq_len: int = 16  # BPTT chunk length
    hidden_size: int = 256
    lr: float = 5e-4
    critic_lr: float = 5e-4
    clip_param: float = 0.2
    entropy_coef: float = 0.001
    value_loss_coef: float = 1.0
    huber_delta: float = 10.0
    max_grad_norm: float = 10.0
    gamma: float = 0.99
    gae_lambda: float = 0.95
    log_std_init: float = -0.5
    value_norm_beta: float = 0.995
    # linear learning-rate decay to lr_final_frac * lr over this many train_op calls
    # (0: constant lr; -1: the training script fills in the total number of iterations)
    lr_decay_iters: int = -1
    lr_final_frac: float = 0.1
    # input normalisation: "running" = per-feature running mean/std (updated after each PPO
    # update, so rollout and update of a batch use the same statistics); "layernorm" =
    # LayerNorm across the raw features as in the original my-mappo code
    input_norm: str = "running"

    @classmethod
    def from_any(cls, cfg) -> "MAPPOLSTMConfig":
        if isinstance(cfg, cls):
            return cfg
        try:
            from omegaconf import OmegaConf

            cfg = OmegaConf.to_container(cfg, resolve=True)
        except Exception:
            cfg = dict(cfg)
        names = set(cls.__dataclass_fields__)
        return cls(**{k: v for k, v in cfg.items() if k in names})


class RunningNorm(nn.Module):
    """Per-feature running mean / std normalisation of the network input."""

    def __init__(self, dim: int, clip: float = 10.0):
        super().__init__()
        self.clip = clip
        self.register_buffer("mean", torch.zeros(dim))
        self.register_buffer("var", torch.ones(dim))
        self.register_buffer("count", torch.tensor(1e-4))

    @torch.no_grad()
    def update(self, x: torch.Tensor):
        x = x.reshape(-1, x.shape[-1])
        b_mean, b_var, b_count = x.mean(0), x.var(0, unbiased=False), x.shape[0]
        delta = b_mean - self.mean
        total = self.count + b_count
        new_var = (self.var * self.count + b_var * b_count + delta.square() * self.count * b_count / total) / total
        # in place: TorchRL's collector acts with a copy of the policy that shares these buffers
        self.mean.add_(delta * b_count / total)
        self.var.copy_(new_var)
        self.count.copy_(total)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return ((x - self.mean) / (self.var.sqrt() + 1e-6)).clamp(-self.clip, self.clip)


class FCLSTMFC(nn.Module):
    """FC -> LSTM -> FC backbone of the paper (Fig. 1/2)."""

    def __init__(self, in_dim: int, out_dim: int, hidden: int, out_gain: float, input_norm: str = "running"):
        super().__init__()
        self.hidden = hidden
        if input_norm == "running":
            self.in_norm = RunningNorm(in_dim)
        elif input_norm == "layernorm":
            self.in_norm = nn.LayerNorm(in_dim)
        else:
            raise ValueError(f"input_norm must be 'running' or 'layernorm', got {input_norm!r}")
        self.fc = nn.Linear(in_dim, hidden)
        self.fc_norm = nn.LayerNorm(hidden)
        self.cell = nn.LSTMCell(hidden, hidden)
        self.out_norm = nn.LayerNorm(hidden)
        self.head = nn.Linear(hidden, out_dim)
        nn.init.orthogonal_(self.fc.weight, gain=2 ** 0.5)
        nn.init.zeros_(self.fc.bias)
        nn.init.orthogonal_(self.cell.weight_ih)
        nn.init.orthogonal_(self.cell.weight_hh)
        nn.init.zeros_(self.cell.bias_ih)
        nn.init.zeros_(self.cell.bias_hh)
        nn.init.orthogonal_(self.head.weight, gain=out_gain)
        nn.init.zeros_(self.head.bias)

    def step(self, x: torch.Tensor, h: torch.Tensor, c: torch.Tensor, reset: torch.Tensor):
        """x (*B, in), h/c (*B, H), reset (*B, 1) bool -> out (*B, out), h, c."""
        keep = (~reset).to(h.dtype)
        h, c = h * keep, c * keep
        batch = x.shape[:-1]
        feat = self.fc_norm(F.relu(self.fc(self.in_norm(x))))
        h, c = self.cell(feat.reshape(-1, self.hidden), (h.reshape(-1, self.hidden), c.reshape(-1, self.hidden)))
        h, c = h.reshape(*batch, self.hidden), c.reshape(*batch, self.hidden)
        return self.head(self.out_norm(h)), h, c

    def unroll(self, x: torch.Tensor, h: torch.Tensor, c: torch.Tensor, reset: torch.Tensor):
        """x (B, T, *, in), h/c (B, *, H), reset (B, T, *, 1) -> out (B, T, *, out)."""
        outs = []
        for t in range(x.shape[1]):
            o, h, c = self.step(x[:, t], h, c, reset[:, t])
            outs.append(o)
        return torch.stack(outs, dim=1)


class ValueNorm(nn.Module):
    """Running mean / std of the returns (debiased EMA, as in MAPPO)."""

    def __init__(self, beta: float = 0.995, eps: float = 1e-5):
        super().__init__()
        self.beta, self.eps = beta, eps
        self.register_buffer("mean", torch.zeros(()))
        self.register_buffer("mean_sq", torch.zeros(()))
        self.register_buffer("debias", torch.zeros(()))

    @torch.no_grad()
    def update(self, x: torch.Tensor):
        self.mean.mul_(self.beta).add_(x.mean() * (1 - self.beta))
        self.mean_sq.mul_(self.beta).add_(x.square().mean() * (1 - self.beta))
        self.debias.mul_(self.beta).add_(1 - self.beta)

    def _stats(self):
        d = self.debias.clamp_min(self.eps)
        mean = self.mean / d
        var = (self.mean_sq / d - mean.square()).clamp_min(1e-2)
        return mean, var.sqrt()

    def normalize(self, x):
        mean, std = self._stats()
        return (x - mean) / std

    def denormalize(self, x):
        mean, std = self._stats()
        return x * std + mean


class MAPPOLSTM(TensorDictModuleBase):
    """
    Drop-in algorithm for OmniDrones' training loop:
        policy = MAPPOLSTM(cfg.algo, env.observation_spec, env.action_spec, env.reward_spec, device)
        collector = SyncDataCollector(env, policy, ...)
        for data in collector: policy.train_op(data)
    """

    def __init__(
        self,
        cfg,
        observation_spec: CompositeSpec,
        action_spec: TensorSpec,
        reward_spec: TensorSpec,
        device="cpu",
    ):
        super().__init__()
        self.cfg = MAPPOLSTMConfig.from_any(cfg)
        self.device = torch.device(device)
        spec = action_spec[ACTION_KEY] if isinstance(action_spec, CompositeSpec) else action_spec
        self.num_agents, self.action_dim = spec.shape[-2:]
        obs_dim = observation_spec[OBS_KEY].shape[-1]
        state_dim = observation_spec[STATE_KEY].shape[-1]
        H = self.cfg.hidden_size

        self.actor = FCLSTMFC(obs_dim, self.action_dim, H, out_gain=0.01, input_norm=self.cfg.input_norm)
        self.log_std = nn.Parameter(torch.full((self.action_dim,), float(self.cfg.log_std_init)))
        self.critic = FCLSTMFC(state_dim, self.num_agents, H, out_gain=1.0, input_norm=self.cfg.input_norm)
        self.value_norm = ValueNorm(self.cfg.value_norm_beta)
        self.to(self.device)

        self.in_keys = [OBS_KEY, STATE_KEY, "is_init", *HIDDEN_KEYS]
        self.out_keys = [ACTION_KEY, "sample_log_prob", "state_value", *[("next", k) for k in HIDDEN_KEYS]]

        self.actor_opt = torch.optim.Adam(list(self.actor.parameters()) + [self.log_std], lr=self.cfg.lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=self.cfg.critic_lr)
        self.num_updates = 0

    # ------------------------------------------------------------------------------------
    # acting
    # ------------------------------------------------------------------------------------

    def _init_hidden(self, td: TensorDictBase):
        batch = td.get(OBS_KEY).shape[:-2]
        H = self.cfg.hidden_size
        dev = td.get(OBS_KEY).device
        for key in HIDDEN_KEYS:
            if key not in td.keys():
                shape = (*batch, self.num_agents, H) if key.startswith("actor") else (*batch, H)
                td.set(key, torch.zeros(shape, device=dev))

    def _dist(self, loc: torch.Tensor):
        scale = self.log_std.clamp(-5.0, 2.0).exp().expand_as(loc)
        return torch.distributions.Normal(loc, scale)

    def forward(self, tensordict: TensorDictBase) -> TensorDictBase:
        self._init_hidden(tensordict)
        obs = tensordict.get(OBS_KEY)  # (*B, n, D)
        state = tensordict.get(STATE_KEY)  # (*B, S)
        is_init = tensordict.get("is_init", None)
        if is_init is None:
            is_init = torch.zeros(*state.shape[:-1], 1, dtype=torch.bool, device=state.device)
        reset_actor = is_init.unsqueeze(-2).expand(*obs.shape[:-1], 1)

        loc, ah, ac = self.actor.step(obs, tensordict.get("actor_h"), tensordict.get("actor_c"), reset_actor)
        dist = self._dist(loc)
        if exploration_type() in (ExplorationType.MODE, ExplorationType.MEAN, ExplorationType.MEDIAN):
            action = loc
        else:
            action = dist.sample()
        value, ch, cc = self.critic.step(state, tensordict.get("critic_h"), tensordict.get("critic_c"), is_init)

        tensordict.set(ACTION_KEY, action)
        tensordict.set("sample_log_prob", dist.log_prob(action).sum(-1))
        tensordict.set("state_value", value.unsqueeze(-1))
        tensordict.set(("next", "actor_h"), ah)
        tensordict.set(("next", "actor_c"), ac)
        tensordict.set(("next", "critic_h"), ch)
        tensordict.set(("next", "critic_c"), cc)
        return tensordict

    # ------------------------------------------------------------------------------------
    # training
    # ------------------------------------------------------------------------------------

    @torch.no_grad()
    def _advantages(self, td: TensorDictBase):
        """GAE over a (num_envs, T) batch. Returns advantages and value targets (E, T, n, 1)."""
        cfg = self.cfg
        rewards = td.get(("next", *REWARD_KEY))  # (E, T, n, 1)
        terminated = td.get(("next", "terminated")).float().unsqueeze(-2)  # (E, T, 1, 1)
        done = td.get(("next", "done")).float().unsqueeze(-2)
        values = self.value_norm.denormalize(td.get("state_value"))
        next_state = td.get(("next", *STATE_KEY))
        no_reset = torch.zeros(*next_state.shape[:-1], 1, dtype=torch.bool, device=next_state.device)
        next_values, _, _ = self.critic.step(
            next_state, td.get(("next", "critic_h")), td.get(("next", "critic_c")), no_reset
        )
        next_values = self.value_norm.denormalize(next_values.unsqueeze(-1))

        T = rewards.shape[1]
        adv = torch.zeros_like(rewards)
        gae = torch.zeros_like(rewards[:, 0])
        for t in reversed(range(T)):
            delta = rewards[:, t] + cfg.gamma * (1.0 - terminated[:, t]) * next_values[:, t] - values[:, t]
            gae = delta + cfg.gamma * cfg.gae_lambda * (1.0 - done[:, t]) * gae
            adv[:, t] = gae
        return adv, adv + values

    def train_op(self, tensordict: TensorDictBase) -> Dict[str, float]:
        cfg = self.cfg
        td = tensordict.select(
            OBS_KEY, STATE_KEY, ACTION_KEY, "is_init", "sample_log_prob", "state_value", *HIDDEN_KEYS,
            ("next", *REWARD_KEY), ("next", "terminated"), ("next", "done"), ("next", *STATE_KEY),
            ("next", "critic_h"), ("next", "critic_c"),
        )
        adv, ret = self._advantages(td)
        self.value_norm.update(ret)
        td.set("ret", self.value_norm.normalize(ret))
        td.set("adv", (adv - adv.mean()) / adv.std().clamp_min(1e-6))

        E, T = td.shape[:2]
        L = min(cfg.seq_len, T)
        T_used = (T // L) * L
        chunks = td[:, :T_used].reshape(E, T_used // L, L).reshape(-1, L)  # (C, L)

        infos = []
        for _ in range(cfg.ppo_epochs):
            perm = torch.randperm(chunks.shape[0], device=chunks.device)
            for idx in perm.chunk(cfg.num_minibatches):
                infos.append(self._update(chunks[idx]))
        out = {k: torch.stack([i[k] for i in infos]).mean().item() for k in infos[0]}
        out["lr"] = self._step_lr()
        # refresh the input statistics only now: this batch was collected and optimised with
        # the same statistics, the next rollout uses the updated ones
        if isinstance(self.actor.in_norm, RunningNorm):
            self.actor.in_norm.update(td.get(OBS_KEY))
            self.critic.in_norm.update(td.get(STATE_KEY))
        return out

    def _step_lr(self) -> float:
        self.num_updates += 1
        frac = 1.0
        if self.cfg.lr_decay_iters > 0:
            progress = min(self.num_updates / self.cfg.lr_decay_iters, 1.0)
            frac = 1.0 - (1.0 - self.cfg.lr_final_frac) * progress
        for opt, base in ((self.actor_opt, self.cfg.lr), (self.critic_opt, self.cfg.critic_lr)):
            for group in opt.param_groups:
                group["lr"] = base * frac
        return self.cfg.lr * frac

    def _update(self, chunk: TensorDictBase) -> Dict[str, torch.Tensor]:
        cfg = self.cfg
        is_init = chunk.get("is_init")  # (B, L, 1)
        obs = chunk.get(OBS_KEY)  # (B, L, n, D)
        loc = self.actor.unroll(
            obs,
            chunk.get("actor_h")[:, 0],
            chunk.get("actor_c")[:, 0],
            is_init.unsqueeze(-2).expand(*obs.shape[:-1], 1),
        )
        dist = self._dist(loc)
        log_prob = dist.log_prob(chunk.get(ACTION_KEY)).sum(-1)  # (B, L, n)
        entropy = dist.entropy().sum(-1).mean()

        adv = chunk.get("adv").squeeze(-1)
        ratio = torch.exp(log_prob - chunk.get("sample_log_prob"))
        surr1 = ratio * adv
        surr2 = ratio.clamp(1.0 - cfg.clip_param, 1.0 + cfg.clip_param) * adv
        policy_loss = -torch.min(surr1, surr2).mean()

        values = self.critic.unroll(
            chunk.get(STATE_KEY), chunk.get("critic_h")[:, 0], chunk.get("critic_c")[:, 0], is_init
        ).unsqueeze(-1)
        old_values = chunk.get("state_value")
        ret = chunk.get("ret")
        clipped = old_values + (values - old_values).clamp(-cfg.clip_param, cfg.clip_param)
        value_loss = torch.max(
            F.huber_loss(values, ret, delta=cfg.huber_delta, reduction="none"),
            F.huber_loss(clipped, ret, delta=cfg.huber_delta, reduction="none"),
        ).mean()

        loss = policy_loss - cfg.entropy_coef * entropy + cfg.value_loss_coef * value_loss
        self.actor_opt.zero_grad()
        self.critic_opt.zero_grad()
        loss.backward()
        actor_grad = nn.utils.clip_grad_norm_(list(self.actor.parameters()) + [self.log_std], cfg.max_grad_norm)
        critic_grad = nn.utils.clip_grad_norm_(self.critic.parameters(), cfg.max_grad_norm)
        self.actor_opt.step()
        self.critic_opt.step()

        with torch.no_grad():
            explained_var = 1 - F.mse_loss(values, ret) / ret.var().clamp_min(1e-8)
        return {
            "policy_loss": policy_loss.detach(),
            "value_loss": value_loss.detach(),
            "entropy": entropy.detach(),
            "ratio": ratio.detach().mean(),
            "actor_grad_norm": actor_grad.detach(),
            "critic_grad_norm": critic_grad.detach(),
            "explained_var": explained_var.detach(),
            "action_std": self.log_std.exp().mean().detach(),
        }

    # ------------------------------------------------------------------------------------
    # checkpoints
    # ------------------------------------------------------------------------------------

    def checkpoint(self) -> dict:
        return {
            "model": self.state_dict(),
            "actor_opt": self.actor_opt.state_dict(),
            "critic_opt": self.critic_opt.state_dict(),
            "num_updates": self.num_updates,
        }

    def load_checkpoint(self, ckpt: dict, actor_only: bool = False):
        """
        actor_only: load just the (swarm-size independent) actor, e.g. to run a policy trained
        with 8 drones on 16 or 32 drones. The critic's input size depends on the swarm size.
        """
        state = ckpt.get("model", ckpt)
        if actor_only:
            actor_state = {k: v for k, v in state.items() if k.startswith("actor.") or k == "log_std"}
            missing = self.load_state_dict(actor_state, strict=False).missing_keys
            assert not [k for k in missing if k.startswith("actor.") or k == "log_std"], missing
            return
        self.load_state_dict(state)
        if "actor_opt" in ckpt:
            self.actor_opt.load_state_dict(ckpt["actor_opt"])
            self.critic_opt.load_state_dict(ckpt["critic_opt"])
        self.num_updates = int(ckpt.get("num_updates", 0))
