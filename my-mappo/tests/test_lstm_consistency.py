"""
Regression tests for the MA-LSTM rollout / training consistency.

Two properties must hold for recurrent PPO to be correct:

1. Rollout: the hidden state stored in the buffer for agent k at step t+1 must be
   the LSTM state produced by agent k's own forward pass at step t.
2. Training: before any gradient step, re-evaluating the stored actions with the
   recurrent generator must reproduce the rollout log-probs exactly, i.e. the PPO
   importance ratio is 1 for every sample.

Run from the my-mappo directory:
    PYTHONPATH=. python -m pytest tests/test_lstm_consistency.py -q
"""

import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from onpolicy.config import get_config  # noqa: E402
from onpolicy.envs.env_wrappers import ShareDummyVecEnv  # noqa: E402
from onpolicy.envs.pybullet_drone_env import PyBulletDroneWrapper  # noqa: E402
from onpolicy.runner.shared.pybullet_drone_runner import PyBulletDroneRunner  # noqa: E402


def _make_runner(tmpdir, n_threads=2, num_drones=4, episode_length=20, chunk=5):
    parser = get_config()
    args = parser.parse_known_args([])[0]
    args.env_name = "pybullet-drones"
    args.algorithm_name = "rmappo"
    args.model = "ma_lstm"
    args.use_recurrent_policy = True
    args.use_naive_recurrent_policy = False
    args.recurrent_N = 2
    args.hidden_size = 32
    args.num_drones = num_drones
    args.n_rollout_threads = n_threads
    args.episode_length = episode_length
    args.data_chunk_length = chunk
    args.num_mini_batch = 1
    args.use_wandb = False
    args.formation_type = "dynamic"
    args.neighbour_radius = 1.0
    args.max_dynamic_neighbours = 3

    def env_fn(rank):
        def _init():
            env = PyBulletDroneWrapper(
                num_drones=num_drones,
                formation_type="dynamic",
                neighbour_radius=1.0,
                max_dynamic_neighbours=3,
            )
            env.seed(rank)
            return env
        return _init

    envs = ShareDummyVecEnv([env_fn(i) for i in range(n_threads)])
    config = {
        "all_args": args,
        "envs": envs,
        "eval_envs": None,
        "num_agents": num_drones,
        "device": torch.device("cpu"),
        "run_dir": Path(tmpdir),
    }
    return PyBulletDroneRunner(config), envs


def _rollout(runner, steps):
    runner.warmup()
    for step in range(steps):
        values, actions, logp, rnn, rnn_c, actions_env = runner.collect(step)
        obs, share_obs, rewards, dones, infos, _ = runner.envs.step(actions_env)
        runner.insert((obs, share_obs, rewards, dones, infos, values, actions, logp, rnn, rnn_c))


@pytest.fixture(scope="module")
def runner():
    torch.manual_seed(0)
    np.random.seed(0)
    with tempfile.TemporaryDirectory() as tmp:
        r, envs = _make_runner(tmp)
        _rollout(r, r.episode_length)
        yield r
        envs.close()


@torch.no_grad()
def test_rollout_hidden_states_belong_to_the_right_agent(runner):
    buf = runner.buffer
    policy = runner.trainer.policy
    t = 3
    obs = np.concatenate(buf.obs[t])
    h_in = np.concatenate(buf.rnn_states[t])
    masks = np.concatenate(buf.masks[t])
    _, _, h_out = policy.actor(
        torch.as_tensor(obs), torch.as_tensor(h_in), torch.as_tensor(masks)
    )
    expected = h_out.numpy().reshape(buf.rnn_states[t + 1].shape)
    np.testing.assert_allclose(buf.rnn_states[t + 1], expected, atol=1e-5)


@torch.no_grad()
def test_ppo_ratio_is_one_before_update(runner):
    runner.compute()
    runner.trainer.prep_rollout()
    buf = runner.buffer
    adv = buf.returns[:-1] - buf.value_preds[:-1]
    gen = buf.recurrent_generator(adv, 1, runner.all_args.data_chunk_length)
    for sample in gen:
        (share_obs, obs, rnn, rnn_c, actions, _, _, masks, active, old_logp, _, avail) = sample
        _, logp, _ = runner.trainer.policy.evaluate_actions(
            share_obs, obs, rnn, rnn_c, actions, masks, avail, active
        )
        ratio = torch.exp(logp - torch.as_tensor(old_logp))
        np.testing.assert_allclose(ratio.numpy(), 1.0, atol=1e-4)
