"""Unit tests for the pure-PyTorch FormationNav logic (no Isaac Sim needed)."""

import math

import numpy as np
import pytest
import torch

from formation_nav.core import (
    FORMATIONS,
    PHASE_FORM,
    PHASE_HOLD,
    PHASE_NAV,
    FormationNavConfig,
    FormationNavCore,
    formation_template,
    greedy_assignment,
    procrustes_error,
    rotate_z,
)


def make_core(num_envs=8, **kw):
    cfg = FormationNavConfig(**kw)
    core = FormationNavCore(cfg, num_envs, "cpu", seed=0)
    start = core.reset(torch.arange(num_envs))
    return core, start


def attitude(E, n):
    heading = torch.tensor([1.0, 0.0, 0.0]).expand(E, n, 3)
    up = torch.tensor([0.0, 0.0, 1.0]).expand(E, n, 3)
    return heading, up


# ----------------------------------------------------------------------------------------
# formations and geometry
# ----------------------------------------------------------------------------------------


@pytest.mark.parametrize("name", FORMATIONS)
@pytest.mark.parametrize("n", [4, 8, 16, 32])
def test_templates_are_centred_with_requested_spacing(name, n):
    t = formation_template(name, n, spacing=1.2)
    assert t.shape == (n, 3)
    assert torch.allclose(t.mean(0), torch.zeros(3), atol=1e-5)
    d = torch.cdist(t, t) + torch.eye(n) * 1e6
    assert d.min().item() == pytest.approx(1.2, rel=1e-4)


def test_3d_templates_are_3d():
    for name in ("cube", "sphere", "pyramid"):
        t = formation_template(name, 8, 1.0)
        assert t[:, 2].max() - t[:, 2].min() > 0.5


def test_greedy_assignment_is_a_permutation_and_beats_identity():
    torch.manual_seed(0)
    a, b = torch.randn(16, 8, 2), torch.randn(16, 8, 2)
    cost = torch.cdist(a, b).square()
    perm = greedy_assignment(cost)
    assert torch.equal(perm.sort(dim=1).values, torch.arange(8).expand(16, 8))
    chosen = torch.gather(cost, 2, perm.unsqueeze(-1)).sum((1, 2))
    identity = cost.diagonal(dim1=1, dim2=2).sum(-1)
    assert (chosen <= identity + 1e-6).all()


def _kabsch_numpy(P, Q):
    P = P - P.mean(0)
    Q = Q - Q.mean(0)
    U, S, Vt = np.linalg.svd(Q.T @ P)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1] *= -1
        R = Vt.T @ U.T
    return np.mean(np.sum((P - Q @ R.T) ** 2, axis=1))


def test_procrustes_matches_reference_and_is_rigid_invariant():
    g = torch.Generator().manual_seed(1)
    target = torch.randn(32, 8, 3, generator=g)
    yaw = torch.rand(32, generator=g) * 2 * math.pi
    moved = rotate_z(target, yaw) + torch.randn(32, 1, 3, generator=g)
    assert procrustes_error(moved, target).max() < 1e-5  # float32 round-off

    noisy = moved + 0.2 * torch.randn(32, 8, 3, generator=g)
    ours = procrustes_error(noisy, target)
    ref = torch.tensor([_kabsch_numpy(p.double().numpy(), q.double().numpy()) for p, q in zip(noisy, target)])
    assert torch.allclose(ours.double(), ref, atol=1e-4)


# ----------------------------------------------------------------------------------------
# episode sampling
# ----------------------------------------------------------------------------------------


def test_reset_starts_on_ground_with_formation_above_and_goal_at_distance():
    core, start = make_core(num_envs=64)
    cfg = core.cfg
    assert torch.allclose(start[..., 2], torch.full_like(start[..., 2], cfg.spawn_height))
    slots = core.slots()
    assert torch.allclose(slots[..., 2].min(1).values, torch.full((64,), cfg.formation_altitude), atol=1e-5)
    d = (core.goal - core.assembly)[:, :2].norm(dim=-1)
    assert ((d >= cfg.goal_distance[0] - 1e-5) & (d <= cfg.goal_distance[1] + 1e-5)).all()
    # the assigned slots are a permutation of the (rotated) template
    for e in range(4):
        t = rotate_z(core.templates[core.formation_id[e]].unsqueeze(0), core.heading[e : e + 1])[0]
        a = core.slot_offset[e]
        assert torch.cdist(a, t).min(dim=1).values.max() < 1e-5


@pytest.mark.parametrize("scenario,has_static,has_dynamic", [
    ("none", False, False), ("static", True, False), ("dynamic", False, True), ("mixed", True, True)])
def test_scenarios_activate_the_right_obstacles(scenario, has_static, has_dynamic):
    core, _ = make_core(num_envs=32, scenario=scenario)
    active = core.obs_active
    assert active[:, : core.M_s].any().item() == has_static
    assert active[:, core.M_s :].any().item() == has_dynamic
    if has_static:
        k = active[:, : core.M_s].sum(1)
        assert ((k >= core.cfg.num_static[0]) & (k <= core.cfg.num_static[1])).all()


def test_obstacles_keep_start_and_goal_areas_free():
    core, _ = make_core(num_envs=64, scenario="mixed")
    cfg = core.cfg
    for _ in range(50):
        core.advance()
    anchor_xy = core.obs_anchor[..., :2]
    for ref in (core.assembly, core.goal):
        d = (anchor_xy - ref[:, None, :2]).norm(dim=-1)
        # along-path clearance => at least `obstacle_clearance` from start / goal centre
        assert (d[core.obs_active] >= cfg.obstacle_clearance - 1e-4).all()


def test_dynamic_obstacles_move_consistently_with_their_velocity():
    core, _ = make_core(num_envs=16, scenario="dynamic")
    dyn = core.obs_active.clone()
    dyn[:, : core.M_s] = False
    prev = core.obs_pos.clone()
    moved = False
    for _ in range(40):
        vel = core.obs_vel.clone()
        core.advance()
        step = core.obs_pos - prev
        # constant speed along the segment (except at a turning point)
        ok = (step - vel * core.cfg.dt).norm(dim=-1) < 1e-4
        turning = (core.obs_vel - vel).norm(dim=-1) > 1e-6
        assert (ok | turning | ~dyn).all()
        moved |= bool((step.norm(dim=-1)[dyn] > 0).any())
        # stays on its segment
        u = ((core.obs_pos - core.obs_anchor) * core.obs_axis).sum(-1)
        assert (u.abs() <= core.obs_half + 1e-4).all()
        prev = core.obs_pos.clone()
    assert moved


def test_obstacle_surface_distance_for_pillar_and_sphere():
    core, _ = make_core(num_envs=1, scenario="mixed", num_static=(1, 1), num_dynamic=(1, 1))
    core.obs_active[:] = True
    core.obs_pos[0, 0] = torch.tensor([0.0, 0.0, 0.0])  # pillar at the origin
    core.obs_pos[0, 1] = torch.tensor([5.0, 0.0, 2.0])  # sphere
    core.obs_radius[0] = torch.tensor([0.5, 0.4])
    pos = torch.zeros(1, core.n, 3)
    pos[0, 0] = torch.tensor([2.0, 0.0, 1.7])  # 1.5 m from the pillar surface
    pos[0, 1] = torch.tensor([5.0, 1.0, 2.0])  # 0.6 m from the sphere surface
    vec, dist = core.obstacle_surface(pos)
    assert dist[0, 0, 0].item() == pytest.approx(1.5, abs=1e-5)
    assert torch.allclose(vec[0, 0, 0], torch.tensor([-1.5, 0.0, 0.0]), atol=1e-5)
    assert dist[0, 1, 1].item() == pytest.approx(0.6, abs=1e-5)
    assert torch.allclose(vec[0, 1, 1], torch.tensor([0.0, -0.6, 0.0]), atol=1e-5)


# ----------------------------------------------------------------------------------------
# phases, rewards, termination
# ----------------------------------------------------------------------------------------


@pytest.mark.parametrize("mode", ["discrete", "carrot"])
def test_perfect_tracking_goes_form_nav_hold_without_termination(mode):
    E = 4
    core, start = make_core(num_envs=E, scenario="none", waypoint_mode=mode)
    n = core.n
    heading, up = attitude(E, n)
    pos = start.clone()
    seen = set()
    for step in range(2000):
        core.advance()
        pos = core.slots().clone()  # oracle: every drone sits on its slot
        reward, terminated = core.update(pos, torch.zeros_like(pos), up, torch.zeros(E, n, core.cfg.action_dim))
        assert not terminated.any(), f"terminated at step {step}"
        seen.update(core.phase.tolist())
        if (core.phase == PHASE_HOLD).all():
            break
    assert seen == {PHASE_FORM, PHASE_NAV, PHASE_HOLD}
    assert (core.stats["success"] == 1).all()
    assert torch.allclose(core.reference(), core.goal, atol=1e-5)
    # holding perfectly at the goal earns the hold bonus and zero formation error
    core.advance()
    reward, _ = core.update(core.slots(), torch.zeros_like(pos), up, torch.zeros(E, n, core.cfg.action_dim))
    assert (core.stats["formation_error"] < 1e-6).all()
    assert (reward > core.cfg.w_hold).all()


def test_discrete_waypoints_are_spaced_by_waypoint_spacing():
    core, _ = make_core(num_envs=2, scenario="none", waypoint_mode="discrete")
    n, E = core.n, 2
    _, up = attitude(E, n)
    s_values = set()
    for _ in range(400):
        core.advance()
        core.update(core.slots().clone(), torch.zeros(E, n, 3), up, torch.zeros(E, n, core.cfg.action_dim))
        s_values.update(round(v, 4) for v in core.s_ref[:1].tolist())
    s = sorted(s_values)
    steps = np.diff(s)
    assert np.all((np.abs(steps - core.cfg.waypoint_spacing) < 1e-4) | (steps <= core.cfg.waypoint_spacing + 1e-4))


def test_drone_collision_and_ground_crash_terminate():
    core, start = make_core(num_envs=2, scenario="none")
    n = core.n
    _, up = attitude(2, n)
    pos = core.slots().clone()
    pos[0, 1] = pos[0, 0] + torch.tensor([0.1, 0.0, 0.0])  # env 0: two drones collide
    core.advance()
    reward, terminated = core.update(pos, torch.zeros_like(pos), up, torch.zeros(2, n, core.cfg.action_dim))
    assert terminated.tolist() == [[True], [False]]
    assert core.stats["crash_drone_collision"][:, 0].tolist() == [1.0, 0.0]
    assert core.stats["crash_ground"][:, 0].tolist() == [0.0, 0.0]
    # env 1 teleported the same way but did not collide: the only extra term is the crash penalty
    assert (reward[0].mean() - reward[1].mean()).item() < -core.cfg.w_crash + 2.0

    core, start = make_core(num_envs=1, scenario="none")
    for _ in range(core.cfg.takeoff_grace + 1):
        core.advance()
    _, terminated = core.update(start, torch.zeros_like(start), up[:1], torch.zeros(1, n, core.cfg.action_dim))
    assert terminated.item()  # still on the ground after the take-off grace period


def test_staged_formation_reward_only_after_form_phase():
    core, start = make_core(num_envs=1, scenario="none", w_nav=0, w_slot=0, w_reached=0, w_avoid=0, w_obstacle=0,
                            w_tilt=0, w_smooth=0, w_hold=0, takeoff_grace=10_000)
    n = core.n
    _, up = attitude(1, n)
    core.advance()
    reward, _ = core.update(start, torch.zeros_like(start), up, torch.zeros(1, n, 4))
    assert reward.abs().max() == 0  # FORM phase: wrong shape is not penalised
    core.phase[:] = PHASE_NAV
    core.advance()
    reward, _ = core.update(start, torch.zeros_like(start), up, torch.zeros(1, n, 4))
    assert reward.max() < 0  # NAV phase: formation error is penalised


# ----------------------------------------------------------------------------------------
# observations
# ----------------------------------------------------------------------------------------


def test_observation_shapes_and_neighbour_padding():
    core, start = make_core(num_envs=3, num_drones=4, max_neighbours=7)
    heading, up = attitude(3, 4)
    obs, state = core.observations(start, torch.zeros_like(start), heading, up, torch.zeros_like(start))
    assert obs.shape == (3, 4, core.obs_dim)
    assert state.shape == (3, core.state_dim)
    nb = obs[..., 19 : 19 + 7 * 11].reshape(3, 4, 7, 11)
    assert (nb[..., :3, -1] == 1).all()  # 3 real neighbours (radius 3 m covers the grid)
    assert (nb[..., 3:, :] == 0).all()  # padded slots


def test_neighbours_outside_radius_are_masked():
    core, _ = make_core(num_envs=1, num_drones=3, neighbour_radius=1.0)
    pos = torch.tensor([[[0.0, 0.0, 1.0], [0.5, 0.0, 1.0], [5.0, 0.0, 1.0]]])
    heading, up = attitude(1, 3)
    obs, _ = core.observations(pos, torch.zeros_like(pos), heading, up, torch.zeros_like(pos))
    nb = obs[..., 19 : 19 + 7 * 11].reshape(1, 3, 7, 11)
    assert nb[0, 0, :, -1].tolist()[:2] == [1.0, 0.0]
    assert torch.allclose(nb[0, 0, 0, :3], torch.tensor([0.5, 0.0, 0.0]))
    assert nb[0, 2, :, -1].sum() == 0  # isolated drone sees nobody


def test_actor_observation_is_translation_invariant():
    core, start = make_core(num_envs=2, scenario="mixed")
    for _ in range(5):
        core.advance()
    heading, up = attitude(2, core.n)
    pos = core.slots() + 0.3 * torch.randn(2, core.n, 3)
    vel = torch.randn(2, core.n, 3)
    obs_a, _ = core.observations(pos, vel, heading, up, torch.zeros_like(vel))
    shift = torch.tensor([37.0, -12.0, 0.0])
    core.assembly += shift
    core.goal += shift
    core.obs_pos += shift
    obs_b, _ = core.observations(pos + shift, vel, heading, up, torch.zeros_like(vel))
    assert torch.allclose(obs_a, obs_b, atol=1e-4)


def test_velocity_action_mapping():
    core, _ = make_core(num_envs=1, num_drones=2)
    a = torch.tensor([[[3.0, 0.0, 0.0, 0.5], [0.0, 0.0, -1.0, -2.0]]])
    v = core.action_to_velocity(a)
    assert torch.allclose(v[0, 0], torch.tensor([0.5 * core.cfg.max_speed, 0.0, 0.0]))
    assert torch.allclose(v[0, 1], torch.tensor([0.0, 0.0, -core.cfg.max_speed]))


def test_formation_relaxation_near_obstacles():
    """Close to an obstacle, moving away from the slot is not penalised by the tracking terms."""
    kw = dict(num_envs=1, scenario="mixed", num_static=(1, 1), num_dynamic=(0, 0), w_slot=0, w_reached=0,
              w_form=0, w_avoid=0, w_obstacle=0, w_tilt=0, w_smooth=0, w_hold=0, takeoff_grace=10_000)
    rewards = {}
    for relax in (0.0, 1.0):
        core, start = make_core(relax_dist=relax, **kw)
        n = core.n
        _, up = attitude(1, n)
        core.obs_active[:] = True
        core.obs_anchor[0, 0] = start[0, 0] + torch.tensor([0.5, 0.0, 0.0])  # pillar next to drone 0
        core.obs_radius[0, 0] = 0.1
        pos = start.clone()
        pos[0, 0, 2] -= 0.05  # drone 0 moves away from its slot (which is above)
        core.advance()
        reward, _ = core.update(pos, torch.zeros_like(pos), up, torch.zeros(1, n, 4))
        rewards[relax] = reward[0, 0, 0].item()
    assert rewards[0.0] < 0  # progress penalty without relaxation
    assert rewards[1.0] > rewards[0.0]  # clearance 0.4 m of 1.0 m: penalty scaled to 40 %
    assert rewards[1.0] == pytest.approx(0.4 * rewards[0.0], rel=1e-3)


def test_obstacle_curriculum_scales_obstacle_count_and_adapts():
    core, _ = make_core(num_envs=64, scenario="static", obstacle_curriculum=True, curriculum_window=8,
                        curriculum_step=0.5)
    assert core.difficulty == 0.0
    assert core.obs_active[:, : core.M_s].sum(1).max() == 1  # one pillar at difficulty 0
    # pretend every episode succeeded: difficulty rises on the next resets
    core.steps[:] = 10
    core.stats["success"][:] = 1.0
    core.reset(torch.arange(64))
    assert core.difficulty == 0.5
    core.steps[:] = 10
    core.stats["success"][:] = 1.0
    core.reset(torch.arange(64))
    assert core.difficulty == 1.0
    k = core.obs_active[:, : core.M_s].sum(1)
    assert k.min() >= core.cfg.num_static[0] and k.max() <= core.cfg.num_static[1]
    # evaluation freezes full difficulty regardless of the curriculum
    core.difficulty = 0.0
    core.freeze_difficulty(1.0)
    core.reset(torch.arange(64))
    assert core.obs_active[:, : core.M_s].sum(1).min() >= core.cfg.num_static[0]
    assert core.difficulty == 0.0
