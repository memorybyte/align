"""
Wiring test for `formation_nav/env.py` without Isaac Sim.

Isaac Sim / OmniDrones modules are replaced by small fakes that mirror the real call flow
of `omni_drones.envs.isaac_env.IsaacEnv` (init -> _design_scene -> _set_specs, _reset ->
_reset_idx -> _compute_state_and_obs, _step -> substeps of _pre_sim_step + physics ->
_post_sim_step -> obs -> reward). The fake drone is the quadrotor model of `quadrotor.py`
(OmniDrones' rotor, force and downwash models, the Crazyflie asset's mass) driven by the real
controller of env.py through the same attributes OmniDrones' MultirotorBase has (`params`,
`forces`, `_view.get_body_masses`, ...). This checks the Hydra configs, every tensordict key
and shape, the controller / prim-view calls, that a formation holds under the downwash, and a
short MAPPO-LSTM training + evaluation through env.py. It does not check the Isaac Sim API.
"""

import os
import sys
import types

import math

import pytest
import torch
import yaml
from tensordict import TensorDict
from torchrl.envs import EnvBase

from formation_nav.controller import quat_rotate
from formation_nav.quadrotor import OMNIDRONES_ASSETS, QuadrotorModel

CFG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, "cfg")
ASSET_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, "formation_nav", "assets")


# ----------------------------------------------------------------------------------------
# fakes
# ----------------------------------------------------------------------------------------


class FakeArticulationView:
    def __init__(self, drone):
        self.drone = drone

    def get_body_masses(self):
        # base link + 4 rotor links, as in OmniDrones' cf2x_pybullet.usd
        E, n = self.drone.num_envs, self.drone.n
        return torch.tensor([0.027, 1e-4, 1e-4, 1e-4, 1e-4]).expand(E, n, 5).clone()


class FakeDrone:
    """OmniDrones' MultirotorBase interface on top of the pure-PyTorch quadrotor model."""

    num_envs = None
    device = "cpu"

    def __init__(self, name):
        with open(os.path.join(ASSET_DIR, f"{name.lower()}.yaml")) as f:
            self.params = yaml.safe_load(f)
        self.n = 0
        self.apply_calls = 0

    def spawn(self, translations):
        self.n = len(translations)

    def initialize(self):
        E, n = self.num_envs, self.n
        env = FakeIsaacEnv.instance
        asset = OMNIDRONES_ASSETS[self.params["name"]]
        scale = float(env.cfg.task.get("downwash_scale", 1.0))
        self.quad = QuadrotorModel(self.params, (E, n), env.dt, downwash=scale > 0, downwash_scale=scale, **asset)
        self._view = FakeArticulationView(self)
        self.forces = torch.zeros(E, n, 3)
        self.cmds = torch.zeros(E, n, 4)
        self._sync()

    def _sync(self):
        q = self.quad
        self.pos, self.rot = q.pos.clone(), q.rot.clone()
        self.vel_w = torch.cat([q.vel, quat_rotate(q.rot, q.omega)], -1)
        self.heading, self.up = q.axes()

    def _reset_idx(self, env_ids, train=True):
        # like OmniDrones: rotors at the hover throttle of the base-link mass
        self.quad.throttle[env_ids] = math.sqrt(0.027 * 9.81 / float(self.quad.max_thrust.sum()))

    def set_world_poses(self, pos, rot, env_ids):
        assert pos.shape == (len(env_ids), self.n, 3) and rot.shape == (len(env_ids), self.n, 4)
        throttle = self.quad.throttle[env_ids].clone()
        self.quad.reset(env_ids, pos - FakeIsaacEnv.instance.envs_positions[env_ids].unsqueeze(1), rot)
        self.quad.throttle[env_ids] = throttle
        self._sync()

    def set_velocities(self, vel, env_ids):
        assert vel.shape == (len(env_ids), self.n, 6)
        self.quad.vel[env_ids] = vel[..., :3]
        self._sync()

    def get_state(self):
        self._sync()
        return torch.cat([self.pos, self.rot, self.vel_w, self.heading, self.up], -1)

    def apply_action(self, cmds):
        assert cmds.shape == (self.num_envs, self.n, 4)
        assert torch.isfinite(cmds).all()
        self.cmds = cmds
        self.apply_calls += 1
        return self.quad.throttle.sum(-1)

    def physics(self, dt):
        self.quad.step(self.cmds)
        self.forces = self.quad.external_force.clone()  # OmniDrones: downwash (+ drag, zero)


class MultirotorBase:
    @staticmethod
    def make(name, controller=None, device="cpu"):
        assert controller is None  # env.py builds its own velocity controller
        return FakeDrone(name), None

    @staticmethod
    def downwash(p0, p1, p1_t, kr=2, kz=1):
        return torch.ones(p0.shape[0], p0.shape[0] - 1, 3)


class RigidPrimView:
    views = []

    def __init__(self, expr, reset_xform_properties=False, shape=None):
        self.expr, self.shape = expr, shape
        self.poses = None
        RigidPrimView.views.append(self)

    def initialize(self):
        pass

    def set_world_poses(self, positions=None, orientations=None, env_indices=None):
        k = FakeIsaacEnv.instance.num_envs if env_indices is None else len(env_indices)
        assert positions.shape[0] == k and positions.shape[-1] == 3
        self.poses = positions


class FakePrim:
    def __init__(self, path):
        self.path = path

    def GetPath(self):
        return types.SimpleNamespace(pathString=self.path)


created = []


def create_obstacle(prim_path, prim_type, translation, attributes):
    created.append((prim_path, prim_type, attributes))
    return FakePrim(prim_path)


class AgentSpec:
    def __init__(self, name, n, observation_key, action_key, reward_key, state_key):
        self.name, self.n = name, n


class _Draw:
    def draw_points(self, *a):
        pass


class _DebugDraw:
    _draw = _Draw()

    def clear(self):
        pass

    def plot(self, *a, **k):
        pass


class FakeIsaacEnv(EnvBase):
    """Same control flow as omni_drones.envs.isaac_env.IsaacEnv."""

    instance = None

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        FakeIsaacEnv.REGISTRY[cls.__name__] = cls

    REGISTRY = {}

    def __init__(self, cfg, headless):
        super().__init__(device=cfg.sim.device, batch_size=[cfg.env.num_envs], run_type_checks=False)
        FakeIsaacEnv.instance = self
        self.cfg = cfg
        self.enable_render(not headless)
        self.num_envs = cfg.env.num_envs
        self.max_episode_length = cfg.env.max_episode_length
        self.substeps = cfg.sim.substeps
        self.dt = cfg.sim.dt
        self.agent_spec = {}
        FakeDrone.num_envs = self.num_envs
        self.envs_positions = torch.arange(self.num_envs).float().unsqueeze(-1) * torch.tensor([40.0, 0, 0])
        self._design_scene()
        self.central_env_idx = torch.tensor(0)
        self.debug_draw = _DebugDraw()
        self.progress_buf = torch.zeros(self.num_envs)
        from torchrl.data import CompositeSpec, DiscreteTensorSpec

        self.done_spec = CompositeSpec({
            "done": DiscreteTensorSpec(2, (1,), dtype=torch.bool),
            "terminated": DiscreteTensorSpec(2, (1,), dtype=torch.bool),
            "truncated": DiscreteTensorSpec(2, (1,), dtype=torch.bool),
        }).expand(self.num_envs)
        self._set_specs()

    def enable_render(self, enable=True):
        self._should_render = lambda substep: enable

    def _reset(self, tensordict, **kwargs):
        if tensordict is not None:
            env_mask = tensordict.get("_reset").reshape(self.num_envs)
        else:
            env_mask = torch.ones(self.num_envs, dtype=bool)
        env_ids = env_mask.nonzero().squeeze(-1)
        self._reset_idx(env_ids)
        self.progress_buf[env_ids] = 0.0
        td = TensorDict({}, self.batch_size)
        td.update(self._compute_state_and_obs())
        td.set("truncated", (self.progress_buf > self.max_episode_length).unsqueeze(1))
        return td

    def _step(self, tensordict):
        for substep in range(self.substeps):
            self._pre_sim_step(tensordict)
            self.drone.physics(self.dt)
        self._post_sim_step(tensordict)
        self.progress_buf += 1
        td = TensorDict({}, self.batch_size)
        td.update(self._compute_state_and_obs())
        td.update(self._compute_reward_and_done())
        return td

    def _set_seed(self, seed):
        torch.manual_seed(seed)

    def render(self, mode="human"):
        return torch.zeros(8, 8, 3, dtype=torch.uint8).numpy()


FAKE_MODULES = []


def install_fakes():
    def module(name, **attrs):
        m = types.ModuleType(name)
        m.__dict__.update(attrs)
        sys.modules[name] = m
        FAKE_MODULES.append(name)
        return m

    for name in ["omni", "omni.isaac", "omni.isaac.core", "omni.isaac.core.utils"]:
        module(name)
    module("omni.isaac.core.utils.viewports", set_camera_view=lambda **k: None)
    gprim = types.SimpleNamespace(CreateDisplayColorAttr=lambda: types.SimpleNamespace(Set=lambda v: None))
    module("pxr", Gf=types.SimpleNamespace(Vec3f=lambda *a: a), UsdGeom=types.SimpleNamespace(Gprim=lambda p: gprim))
    module("omni_drones")
    module("omni_drones.utils")
    module("omni_drones.utils.kit", set_collision_properties=lambda path, **k: None)
    module("omni_drones.utils.scene", design_scene=lambda: None)
    module("omni_drones.envs")
    module("omni_drones.envs.isaac_env", AgentSpec=AgentSpec, IsaacEnv=FakeIsaacEnv)
    module("omni_drones.envs.utils", create_obstacle=create_obstacle)
    module("omni_drones.robots")
    module("omni_drones.robots.drone", MultirotorBase=MultirotorBase)
    module("omni_drones.views", RigidPrimView=RigidPrimView)


@pytest.fixture(scope="module")
def fn_env_module():
    install_fakes()
    sys.modules.pop("formation_nav.env", None)
    import formation_nav.env as env_module

    yield env_module
    # remove only the fakes (never torch & co: re-importing those breaks torch)
    for name in FAKE_MODULES + ["formation_nav.env"]:
        sys.modules.pop(name, None)


def compose(config_name, overrides):
    from hydra import compose as hydra_compose, initialize_config_dir
    from omegaconf import OmegaConf

    with initialize_config_dir(config_dir=os.path.abspath(CFG_DIR), version_base=None):
        cfg = hydra_compose(config_name=config_name, overrides=overrides)
    OmegaConf.resolve(cfg)
    OmegaConf.set_struct(cfg, False)
    return cfg


SMALL = [
    "task.env.num_envs=4",
    "task.env.max_episode_length=40",
    "task.sim.device=cpu",
    "task.num_drones=4",
    "task.scenario=mixed",
    "algo.hidden_size=32",
    "algo.train_every=16",
    "algo.seq_len=8",
    "algo.num_minibatches=2",
    "algo.ppo_epochs=1",
]


# ----------------------------------------------------------------------------------------
# tests
# ----------------------------------------------------------------------------------------


def test_task_yaml_keys_match_the_config_dataclass():
    from dataclasses import fields

    from formation_nav.core import FormationNavConfig

    cfg = compose("train", [])
    extra = {"name", "env", "sim", "drone_model", "controller", "downwash_scale", "obstacle_physics_collision",
             "follow_camera"}
    names = {f.name for f in fields(FormationNavConfig)}
    unknown = set(cfg.task.keys()) - names - extra
    assert not unknown, f"unknown task keys (typo?): {unknown}"
    # yaml defaults equal the dataclass defaults (single source of truth)
    from omegaconf import OmegaConf

    loaded = FormationNavConfig.from_dict(OmegaConf.to_container(cfg.task))
    default = FormationNavConfig()
    for f in fields(FormationNavConfig):
        if f.name in ("dt",):
            continue
        assert getattr(loaded, f.name) == getattr(default, f.name), f.name
    assert cfg.sim.dt == 0.016 and cfg.sim.substeps == 2
    assert cfg.task.drone_model.name == "Crazyflie"


def test_controller_block_matches_env_defaults(fn_env_module):
    cfg = compose("train", [])
    assert set(cfg.task.controller.keys()) == set(fn_env_module.CONTROLLER_DEFAULTS)


def test_hummingbird_preset_composes():
    from omegaconf import OmegaConf

    from formation_nav.core import FormationNavConfig

    cfg = compose("train", ["task=FormationNavHummingbird"])
    assert cfg.task.name == "FormationNav" and cfg.task.drone_model.name == "Hummingbird"
    task = FormationNavConfig.from_dict(OmegaConf.to_container(cfg.task))
    assert task.formation_spacing == 1.2 and task.drone_radius == 0.25 and task.num_drones == 8
    assert cfg.task.controller.downwash_feedforward is True


def test_smoke_config_composes():
    cfg = compose("smoke", ["lite=true"])
    assert cfg.lite is True and cfg.task.env.num_envs == 16 and "cube" in cfg.formations


def test_eval_config_composes():
    cfg = compose("eval", ["checkpoint_path=x.pt"])
    assert cfg.task.env.num_envs == 64
    assert list(cfg.eval_scenarios) == ["none", "static", "dynamic", "mixed"]


def test_env_specs_reset_step_and_obstacle_prims(fn_env_module):
    cfg = compose("train", SMALL)
    env = fn_env_module.FormationNav(cfg, headless=True)
    n, core = 4, env.core
    assert len(created) >= core.M
    assert {p[1] for p in created} == {"Cylinder", "Sphere"}
    # the controller uses the simulated mass (all bodies), not the yaml's 0.028 kg
    assert abs(env.controller.mass.item() - 0.0274) < 1e-6
    td = env.reset()
    assert td["agents", "observation"].shape == (4, n, core.obs_dim)
    assert td["agents", "observation_central"].shape == (4, core.state_dim)
    # drones start on the ground
    assert torch.allclose(env.drone.pos[..., 2], torch.full((4, n), cfg.task.spawn_height))
    static_view = next(v for v in RigidPrimView.views if "pillar" in v.expr)
    assert static_view.poses.shape == (4, core.M_s, 3)
    td.set(("agents", "action"), torch.zeros(4, n, 4))
    td["agents", "action"][..., 2] = 1.0
    td["agents", "action"][..., 3] = 1.0  # straight up at max speed
    calls = env.drone.apply_calls
    for _ in range(10):
        out = env.step(td)
        td = out["next"].set(("agents", "action"), td["agents", "action"])
    assert env.drone.apply_calls - calls == 10 * cfg.sim.substeps  # controller runs every physics step
    assert out["next", "agents", "reward"].shape == (4, n, 1)
    assert out["next", "done"].shape == (4, 1)
    assert (env.drone.pos[..., 2] > cfg.task.spawn_height + 0.1).all()
    for k in ("success", "formation_error", "collisions_obstacle"):
        assert out["next", "stats", k].shape == (4, 1)


@pytest.mark.parametrize("feedforward", [True, False])
def test_stacked_formation_holds_only_with_the_downwash_feedforward(fn_env_module, feedforward):
    """Scripted slot seeking through env.py: take-off and a cube of 8 Crazyflies under the downwash."""
    from torchrl.envs.utils import step_mdp

    from formation_nav.scripted import SlotSeeker

    cfg = compose("train", [
        "task.env.num_envs=2", "task.sim.device=cpu", "task.num_drones=8", "task.scenario=none",
        "task.formation=cube", f"task.controller.downwash_feedforward={feedforward}",
    ])
    env = fn_env_module.FormationNav(cfg, headless=True)
    policy = SlotSeeker(env.core.cfg.max_speed)
    td = env.reset()
    crashed = torch.zeros(2, dtype=torch.bool)
    for _ in range(200):  # 6.4 s: staged take-off (2.5 s), forming, first waypoints
        out = env.step(policy(td))
        crashed |= out["next", "terminated"].squeeze(-1)
        td = step_mdp(out)
    formed = env.core.stats["time_to_form"].squeeze(-1) > 0
    if feedforward:
        assert formed.all() and not crashed.any()
    else:
        assert crashed.all() and not formed.any()  # the lower layer sinks to the ground


def test_downwash_scale_wraps_omnidrones_model(fn_env_module):
    scaled = fn_env_module._scaled_downwash(0.25)
    p = torch.zeros(3, 3)
    assert torch.allclose(scaled(p, p, p, kz=0.3), torch.full((3, 2, 3), 0.25))


def test_training_and_evaluation_run_through_env_py(fn_env_module, tmp_path):
    from torchrl.collectors import SyncDataCollector
    from torchrl.envs.transforms import InitTracker, TransformedEnv

    from formation_nav.evaluation import evaluate_scenarios
    from formation_nav.mappo_lstm import MAPPOLSTM

    cfg = compose("train", SMALL)
    base = fn_env_module.FormationNav(cfg, headless=True)
    env = TransformedEnv(base, InitTracker())
    policy = MAPPOLSTM(cfg.algo, env.observation_spec, env.action_spec, env.reward_spec)
    collector = SyncDataCollector(env, policy, frames_per_batch=4 * 16, total_frames=-1, device="cpu",
                                  return_same_td=True)
    for i, data in enumerate(collector):
        info = policy.train_op(data.to_tensordict())
        assert all(v == v for v in info.values())  # no NaN
        if i == 1:
            break
    results = evaluate_scenarios(env, base, policy, max_steps=40, scenarios=["none", "dynamic"],
                                 plot_dir=str(tmp_path))
    assert set(results) == {"none", "dynamic"}
    assert (tmp_path / "trajectory_none.png").exists()
    # the dynamic spheres were moved by env.py every step
    dyn_view = next(v for v in RigidPrimView.views if "mover" in v.expr)
    assert dyn_view.poses.shape == (4, base.core.M_d, 3)


def test_render_path_draws_route_slots_and_follows_the_swarm(fn_env_module):
    cfg = compose("train", SMALL + ["task.waypoint_planner=dp"])
    env = fn_env_module.FormationNav(cfg, headless=False)  # rendering on -> _debug_vis runs every step
    td = env.reset()
    td.set(("agents", "action"), torch.zeros(4, 4, 4))
    out = env.step(td)
    assert out["next", "info", "route"].shape == (4, env.core.Kp, 3)
