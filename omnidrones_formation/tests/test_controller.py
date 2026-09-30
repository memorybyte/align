"""
Tests of the velocity controller (formation_nav/controller.py) and the quadrotor model
(formation_nav/quadrotor.py).

The equivalence tests load OmniDrones' own LeePositionController and downwash model from its
source (OMNIDRONES_DIR, or an installed `omni_drones` package) without starting Isaac Sim; they
are skipped when the source is not found. The flight tests fly the Crazyflie as OmniDrones
simulates it (asset mass 0.0274 kg, rotors 0.0396 m from the centre) at the FormationNav
physics step (16 ms).
"""

import ast
import importlib.util
import math
import os
import sys
import textwrap
import types

import pytest
import torch
import yaml

from formation_nav.controller import LEE_GAINS, LeeVelocityController
from formation_nav.core import formation_template
from formation_nav.quadrotor import OMNIDRONES_ASSETS, QuadrotorModel, downwash_force

ASSETS = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, "formation_nav", "assets")
DT = 0.016


def params(name):
    with open(os.path.join(ASSETS, f"{name}.yaml")) as f:
        return yaml.safe_load(f)


# ----------------------------------------------------------------------------------------
# OmniDrones source
# ----------------------------------------------------------------------------------------


def omnidrones_root():
    root = os.environ.get("OMNIDRONES_DIR")
    if root and os.path.isdir(os.path.join(root, "omni_drones")):
        return root
    try:
        spec = importlib.util.find_spec("omni_drones")
    except (ImportError, ValueError):
        spec = None
    if spec is not None and spec.origin:
        return os.path.dirname(os.path.dirname(spec.origin))
    return None


@pytest.fixture(scope="module")
def omnidrones():
    """OmniDrones' controller and utils modules, loaded from source without omni_drones/__init__.py."""
    root = omnidrones_root()
    if root is None:
        pytest.skip("OmniDrones source not found (set OMNIDRONES_DIR to the repository)")
    names = [
        "omni_drones", "omni_drones.utils", "omni_drones.controllers", "omni_drones.utils.torch",
        "omni_drones.controllers.controller", "omni_drones.controllers.lee_position_controller",
    ]
    saved = {k: sys.modules.get(k) for k in names}
    for pkg in names[:3]:
        m = types.ModuleType(pkg)
        m.__path__ = [os.path.join(root, *pkg.split("."))]
        sys.modules[pkg] = m

    def load(name):
        path = os.path.join(root, *name.split(".")) + ".py"
        spec = importlib.util.spec_from_file_location(name, path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
        return mod

    try:
        utils_torch = load("omni_drones.utils.torch")
        load("omni_drones.controllers.controller")
        lee = load("omni_drones.controllers.lee_position_controller")
        yield types.SimpleNamespace(root=root, lee=lee, utils_torch=utils_torch)
    finally:
        for k, v in saved.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v


def random_states(batch, seed=0):
    g = torch.Generator().manual_seed(seed)
    pos = torch.randn(*batch, 3, generator=g)
    rot = torch.randn(*batch, 4, generator=g)
    rot = rot / rot.norm(dim=-1, keepdim=True)
    vel = torch.randn(*batch, 3, generator=g)
    ang_vel = torch.randn(*batch, 3, generator=g)
    return torch.cat([pos, rot, vel, ang_vel], -1), g


@pytest.mark.parametrize("name", ["hummingbird", "firefly", "neo11"])
def test_rotor_commands_match_omnidrones_lee_position_controller(omnidrones, name):
    with open(os.path.join(omnidrones.root, "omni_drones", "robots", "assets", "usd", f"{name}.yaml")) as f:
        uav = yaml.safe_load(f)
    theirs = omnidrones.lee.LeePositionController(9.81, uav)
    ours = LeeVelocityController(uav)
    state, g = random_states((64, 4))
    targets = dict(
        target_pos=torch.randn(64, 4, 3, generator=g),
        target_vel=torch.randn(64, 4, 3, generator=g),
        target_acc=torch.randn(64, 4, 3, generator=g),
        target_yaw=torch.rand(64, 4, 1, generator=g) * 6.0 - 3.0,
    )
    for kw in (targets, dict(target_vel=targets["target_vel"]), {}):
        a, b = ours.compute(state, **kw), theirs.compute(state, **kw)
        assert torch.allclose(a, b, atol=1e-4, rtol=1e-4), (kw.keys(), (a - b).abs().max())


def test_downwash_force_matches_omnidrones(omnidrones):
    path = os.path.join(omnidrones.root, "omni_drones", "robots", "drone", "multirotor.py")
    src = open(path).read()
    ns = {"torch": torch, "off_diag": omnidrones.utils_torch.off_diag, "normalize": omnidrones.utils_torch.normalize}
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.FunctionDef) and node.name in ("downwash", "separation"):
            code = " " * node.col_offset + ast.get_source_segment(src, node)
            exec(textwrap.dedent(code), ns)
    g = torch.Generator().manual_seed(1)
    pos = torch.randn(16, 6, 3, generator=g) * torch.tensor([1.0, 1.0, 2.0])
    thrust = torch.tensor([0.0, 0.0, 0.3]) + 0.1 * torch.randn(16, 6, 3, generator=g)  # tilted thrusts
    theirs = torch.func.vmap(ns["downwash"])(pos, pos, thrust, kz=0.3).sum(-2)
    assert torch.allclose(downwash_force(pos, thrust), theirs, atol=1e-6)


# ----------------------------------------------------------------------------------------
# gains
# ----------------------------------------------------------------------------------------


def test_crazyflie_gains_give_the_hummingbird_closed_loop():
    cf = LeeVelocityController(params("crazyflie"))
    hb = LeeVelocityController(params("hummingbird"))
    assert torch.allclose(cf.attitude_gain, hb.attitude_gain, rtol=0.01)  # alpha / e_R, 1/s^2
    assert torch.allclose(cf.ang_rate_gain, hb.ang_rate_gain, rtol=0.01)  # alpha / omega, 1/s
    assert torch.equal(cf.vel_gain, hb.vel_gain) and torch.equal(cf.pos_gain, hb.pos_gain)
    wn = cf.attitude_gain[0].sqrt()
    assert abs(wn - 10.0) < 0.1 and abs(cf.ang_rate_gain[0] / (2 * wn) - 0.71) < 0.02


def test_unknown_drone_needs_explicit_gains():
    uav = dict(params("crazyflie"), name="mystery")
    with pytest.raises(KeyError):
        LeeVelocityController(uav)
    LeeVelocityController(uav, gains=LEE_GAINS["crazyflie"])


# ----------------------------------------------------------------------------------------
# flight tests on the quadrotor model
# ----------------------------------------------------------------------------------------


def crazyflie(batch=(1, 1), downwash=False, **controller_kw):
    uav = params("crazyflie")
    asset = OMNIDRONES_ASSETS["crazyflie"]
    quad = QuadrotorModel(uav, batch, DT, downwash=downwash, **asset)
    controller_kw.setdefault("mass", asset["mass"])
    ctrl = LeeVelocityController(uav, dt=DT, **controller_kw)
    return quad, ctrl


def fly(quad, ctrl, target_vel, seconds, feedforward=False, outer=None):
    """Run the loop; returns (time, position, tilt in degrees) histories."""
    yaw = torch.zeros(*quad.batch, 1)
    pos, tilt = [], []
    for k in range(int(round(seconds / DT))):
        v = target_vel(k * DT) if callable(target_vel) else target_vel
        if outer is not None:
            v = outer(quad.pos)
        acc = -quad.external_force / ctrl.mass if feedforward else None
        quad.step(ctrl.compute(quad.root_state(), target_vel=v, target_acc=acc, target_yaw=yaw))
        pos.append(quad.pos.clone())
        tilt.append(torch.rad2deg(torch.acos(quad.axes()[1][..., 2].clamp(-1, 1))))
    return torch.stack(pos), torch.stack(tilt)


def hover_at(quad, pos):
    quad.reset(slice(None), pos)
    quad.throttle[:] = math.sqrt(quad.mass * 9.81 / float(quad.max_thrust.sum()))


def test_crazyflie_hover_and_velocity_step():
    quad, ctrl = crazyflie()
    hover_at(quad, torch.tensor([[[0.0, 0.0, 2.0]]]))
    pos, _ = fly(quad, ctrl, torch.zeros(1, 1, 3), 2.0)
    assert (pos[-1] - torch.tensor([0.0, 0.0, 2.0])).norm() < 0.01
    # 1.5 m/s sideways (the policy's maximum speed)
    target = torch.tensor([[[1.5, 0.0, 0.0]]])
    vel, tilt = [], []
    yaw = torch.zeros(1, 1, 1)
    for _ in range(int(2.0 / DT)):
        quad.step(ctrl.compute(quad.root_state(), target_vel=target, target_yaw=yaw))
        vel.append(quad.vel[0, 0, 0].item())
        tilt.append(math.degrees(math.acos(min(1.0, quad.axes()[1][0, 0, 2].item()))))
    settle = next(i for i in range(len(vel)) if all(abs(v - 1.5) < 0.15 for v in vel[i:])) * DT
    assert settle < 1.0 and max(vel) < 1.5 * 1.05 and max(tilt) < 25.0
    assert abs(quad.pos[0, 0, 2].item() - 2.0) < 0.05  # altitude held while accelerating


def test_crazyflie_takes_off_from_the_ground():
    quad, ctrl = crazyflie()
    quad.reset(slice(None), torch.zeros(1, 1, 3))
    pos, tilt = fly(quad, ctrl, torch.tensor([[[0.0, 0.0, 1.0]]]), 2.0)
    assert pos[-1, 0, 0, 2] > 1.3 and tilt.max() < 1.0  # rotors spin up from rest


def test_crazyflie_recovers_from_tilt_and_spin():
    g = torch.Generator().manual_seed(0)
    quad, ctrl = crazyflie(batch=(256,))
    quad.reset(slice(None), torch.tensor([0.0, 0.0, 3.0]))
    axis = torch.randn(256, 3, generator=g)
    axis[:, 2] = 0.0
    axis = axis / axis.norm(dim=-1, keepdim=True)
    half = math.radians(30.0) / 2
    quad.rot = torch.cat([torch.full((256, 1), math.cos(half)), axis * math.sin(half)], -1)
    spin = torch.randn(256, 3, generator=g)
    quad.omega = spin / spin.norm(dim=-1, keepdim=True) * 3.0  # 3 rad/s about a random axis
    quad.throttle[:] = math.sqrt(quad.mass * 9.81 / float(quad.max_thrust.sum()))
    pos, tilt = fly(quad, ctrl, torch.zeros(256, 3), 2.0)
    assert torch.isfinite(pos).all()
    assert tilt[-1].max() < 10.0 and tilt.max() < 45.0  # upright, at most braking the drift
    assert (pos[..., 2].min() > 2.6) and (pos[-1, :, 2] - 3.0).abs().max() < 0.3


def test_feedforward_holds_stacked_drones_under_omnidrones_downwash():
    """Two Crazyflies 1 m apart, one above the other: the lower one gets 59 % of a weight."""
    column = torch.tensor([[[0.0, 0.0, 1.5], [0.0, 0.0, 2.5]]])
    outer = lambda p: ((column - p) * 1.0).clamp(-1.5, 1.5)  # noqa: E731  (a policy holding the slots)
    results = {}
    for ff in (False, True):
        quad, ctrl = crazyflie(batch=(1, 2), downwash=True)
        hover_at(quad, column)
        pos, _ = fly(quad, ctrl, None, 6.0, feedforward=ff, outer=outer)
        results[ff] = (pos[..., 0, 2] - 1.5).abs().max().item()
    assert results[False] > 0.5  # the lower drone sinks without the feed-forward
    assert results[True] < 0.1


@pytest.mark.parametrize("name", ["cube", "sphere", "pyramid"])
def test_eight_crazyflies_hold_3d_formations_with_feedforward(name):
    slots = formation_template(name, 8, 1.0)
    slots[:, 2] += 1.5 - slots[:, 2].min()
    slots = slots.unsqueeze(0)
    quad, ctrl = crazyflie(batch=(1, 8), downwash=True)
    hover_at(quad, slots)
    outer = lambda p: ((slots - p) * 1.0).clamp(-1.5, 1.5)  # noqa: E731
    pos, _ = fly(quad, ctrl, None, 5.0, feedforward=True, outer=outer)
    assert (pos[-1] - slots).norm(dim=-1).max() < 0.05
    assert quad.throttle.max() < 1.0  # thrust not saturated


def test_integral_action_removes_a_mass_error():
    """The controller believes the yaml mass (0.028 kg); the simulated drone weighs 0.0274 kg."""
    drift = {}
    for ki in (0.0, 1.0):
        quad, ctrl = crazyflie(mass=0.028, integral_gain=[0.0, 0.0, ki], integral_limit=[0.0, 0.0, 2.0])
        hover_at(quad, torch.tensor([[[0.0, 0.0, 2.0]]]))
        fly(quad, ctrl, torch.zeros(1, 1, 3), 8.0)
        drift[ki] = abs(quad.vel[0, 0, 2].item())
    assert drift[0.0] > 0.05 and drift[1.0] < 0.01
    # reset() clears the state, integrate=False freezes it
    assert ctrl.integral.abs().sum() > 0
    ctrl.reset(0)
    assert ctrl.integral.abs().sum() == 0
    frozen = ctrl.integral.clone()
    ctrl.compute(quad.root_state(), target_vel=torch.ones(1, 1, 3), integrate=torch.zeros(1, 1, dtype=torch.bool))
    assert torch.equal(ctrl.integral, frozen)


def test_integral_needs_dt():
    with pytest.raises(ValueError):
        LeeVelocityController(params("crazyflie"), integral_gain=[0.0, 0.0, 1.0])
