"""FormationNavLite with the quadrotor backend (Crazyflie as OmniDrones simulates it)."""

import pytest
import torch
from torchrl.envs.utils import check_env_specs, step_mdp

from formation_nav import FormationNavConfig
from formation_nav.pointmass_env import FormationNavLite
from formation_nav.scripted import SlotSeeker


def fly_scripted(env, steps):
    policy = SlotSeeker(env.cfg.max_speed)
    td = env.reset()
    crashed = torch.zeros(env.num_envs, dtype=torch.bool)
    for _ in range(steps):
        out = env.step(policy(td))
        crashed |= out["next", "terminated"].squeeze(-1)
        td = step_mdp(out)
    return td, crashed


@pytest.mark.parametrize("formation", ["cube", "sphere", "pyramid"])
def test_scripted_crazyflies_take_off_and_form_under_downwash(formation):
    cfg = FormationNavConfig(scenario="none", formation=formation)
    env = FormationNavLite(cfg, num_envs=4, max_episode_length=400, dynamics="quadrotor", seed=0)
    td, crashed = fly_scripted(env, 220)  # 7 s
    assert not crashed.any()
    assert (env.core.stats["time_to_form"] > 0).all()
    up_z = td["agents", "observation"][..., 8]  # attitude comes from the rigid-body model
    assert (up_z > 0.9).all()


def test_without_feedforward_the_lower_layer_cannot_form():
    cfg = FormationNavConfig(scenario="none", formation="cube")
    env = FormationNavLite(cfg, num_envs=4, max_episode_length=400, dynamics="quadrotor", seed=0,
                           downwash_feedforward=False)
    _, crashed = fly_scripted(env, 220)
    assert crashed.all() and (env.core.stats["time_to_form"] < 0).all()


def test_quadrotor_backend_specs_and_random_rollout():
    cfg = FormationNavConfig(num_drones=4, scenario="mixed")
    quad = FormationNavLite(cfg, num_envs=3, max_episode_length=50, dynamics="quadrotor", seed=0)
    point = FormationNavLite(cfg, num_envs=3, max_episode_length=50, seed=0)
    check_env_specs(quad)
    assert quad.observation_spec == point.observation_spec and quad.action_spec == point.action_spec
    td = quad.rollout(60, break_when_any_done=False)  # random actions, with automatic resets
    assert torch.isfinite(td["next", "agents", "observation"]).all()


def test_unknown_dynamics_is_rejected():
    with pytest.raises(ValueError):
        FormationNavLite(FormationNavConfig(), num_envs=1, dynamics="helicopter")
