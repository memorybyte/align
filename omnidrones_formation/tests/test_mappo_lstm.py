"""Tests for the FC-LSTM-FC MAPPO on the point-mass FormationNav environment."""

import pytest
import torch
from torchrl.collectors import SyncDataCollector
from torchrl.envs.transforms import InitTracker, TransformedEnv
from torchrl.envs.utils import ExplorationType, set_exploration_type

from formation_nav import FormationNavConfig, MAPPOLSTM
from formation_nav.pointmass_env import FormationNavLite

ALGO = dict(train_every=24, seq_len=8, num_minibatches=4, ppo_epochs=2, hidden_size=32)


def make(num_envs=8, max_len=20, **task):
    torch.manual_seed(0)
    cfg = FormationNavConfig(num_drones=4, **task)
    env = TransformedEnv(FormationNavLite(cfg, num_envs=num_envs, max_episode_length=max_len, seed=0), InitTracker())
    policy = MAPPOLSTM(ALGO, env.observation_spec, env.action_spec, env.reward_spec)
    collector = SyncDataCollector(
        env, policy, frames_per_batch=num_envs * ALGO["train_every"], total_frames=-1,
        device="cpu", return_same_td=True,
    )
    return env, policy, collector


def test_rollout_contains_episode_resets_inside_chunks():
    env, policy, collector = make(max_len=10)
    data = next(iter(collector)).to_tensordict()
    assert data.shape == (8, 24)
    assert data["is_init"][:, 1:].any()  # resets happen mid-rollout


@torch.no_grad()
def test_ppo_ratio_is_one_before_the_first_update():
    """Re-evaluating stored actions with the chunked LSTM unroll reproduces rollout log-probs."""
    env, policy, collector = make(max_len=10)
    data = next(iter(collector)).to_tensordict()
    E, T = data.shape
    L = ALGO["seq_len"]
    chunks = data[:, : (T // L) * L].reshape(E, T // L, L).reshape(-1, L)
    obs = chunks["agents", "observation"]
    loc = policy.actor.unroll(
        obs, chunks["actor_h"][:, 0], chunks["actor_c"][:, 0],
        chunks["is_init"].unsqueeze(-2).expand(*obs.shape[:-1], 1),
    )
    logp = policy._dist(loc).log_prob(chunks["agents", "action"]).sum(-1)
    torch.testing.assert_close(logp, chunks["sample_log_prob"], atol=1e-4, rtol=1e-4)

    values = policy.critic.unroll(
        chunks["agents", "observation_central"], chunks["critic_h"][:, 0], chunks["critic_c"][:, 0], chunks["is_init"]
    ).unsqueeze(-1)
    torch.testing.assert_close(values, chunks["state_value"], atol=1e-4, rtol=1e-4)


def test_train_op_updates_the_collectors_policy():
    env, policy, collector = make()
    it = iter(collector)
    data = next(it).to_tensordict()
    before = policy.actor.fc.weight.detach().clone()
    info = policy.train_op(data)
    assert all(torch.isfinite(torch.tensor(v)) for v in info.values())
    assert not torch.equal(before, policy.actor.fc.weight)
    # the collector acts with the updated weights (it shares parameter storage)
    collector_policy = collector.policy
    torch.testing.assert_close(collector_policy.actor.fc.weight, policy.actor.fc.weight)
    next(it)


def test_deterministic_mode_uses_the_mean_action():
    env, policy, _ = make()
    td = env.reset()
    with set_exploration_type(ExplorationType.MODE):
        a1 = policy(td.clone())["agents", "action"]
        a2 = policy(td.clone())["agents", "action"]
    torch.testing.assert_close(a1, a2)


def test_checkpoint_roundtrip():
    env, policy, collector = make()
    policy.train_op(next(iter(collector)).to_tensordict())
    ckpt = policy.checkpoint()
    other = MAPPOLSTM(ALGO, env.observation_spec, env.action_spec, env.reward_spec)
    other.load_checkpoint(ckpt)
    for (k, a), (_, b) in zip(policy.state_dict().items(), other.state_dict().items()):
        torch.testing.assert_close(a, b, msg=k)


def test_actor_only_checkpoint_runs_on_a_larger_swarm():
    """The actor's input does not depend on the swarm size (fixed neighbour slots)."""
    env, policy, collector = make()
    ckpt = policy.checkpoint()
    big_cfg = FormationNavConfig(num_drones=8)
    big = TransformedEnv(FormationNavLite(big_cfg, num_envs=2, max_episode_length=20, seed=1), InitTracker())
    other = MAPPOLSTM(ALGO, big.observation_spec, big.action_spec, big.reward_spec)
    with pytest.raises(RuntimeError):
        other.load_checkpoint(ckpt)  # the critic depends on the swarm size
    other.load_checkpoint(ckpt, actor_only=True)
    torch.testing.assert_close(other.actor.fc.weight, policy.actor.fc.weight)
    td = other(big.reset())
    assert td["agents", "action"].shape == (2, 8, 4)
