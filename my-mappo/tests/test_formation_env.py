"""
Tests for formation placement and the formation error metric.

    PYTHONPATH=. python -m pytest tests/test_formation_env.py -q
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from onpolicy.envs.pybullet_drone_env import PyBulletDroneWrapper  # noqa: E402
from onpolicy.utils.formation import compute_formation_error  # noqa: E402


def _pairwise(x):
    return np.linalg.norm(x[:, None, :] - x[None, :, :], axis=-1)


@pytest.mark.parametrize("formation", ["cube", "sphere", "pyramid", "plane"])
def test_targets_keep_the_3d_shape_and_stay_above_ground(formation):
    np.random.seed(0)
    env = PyBulletDroneWrapper(num_drones=8, formation_type=formation)
    template = env._env._formation_template
    try:
        for _ in range(5):
            env.reset()
            aviary = env._env
            assert aviary.TARGET_POS[:, 2].min() >= aviary._arena_z_range[0] - 1e-6
            # Initial positions carry Gaussian perturbation noise, but the formation
            # itself is lifted, so no drone is pushed into the ground clamp.
            init_center_offset = aviary.INIT_XYZS - aviary.INIT_XYZS.mean(0)
            np.testing.assert_allclose(
                _pairwise(init_center_offset), _pairwise(template), atol=0.5
            )
            assert aviary.INIT_XYZS[:, 2].min() >= 0.05
            # Targets are the template translated (no perturbation): identical shape.
            np.testing.assert_allclose(
                _pairwise(aviary.TARGET_POS), _pairwise(template), atol=1e-5
            )
    finally:
        env.close()


def test_formation_error_is_mean_squared_and_rigid_invariant():
    rng = np.random.default_rng(0)
    target = rng.normal(size=(8, 3))
    yaw = 0.7
    R = np.array([[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]])
    moved = target @ R.T + np.array([3.0, -2.0, 1.0])
    E, _ = compute_formation_error(moved, target)
    assert E == pytest.approx(0.0, abs=1e-10)

    # One drone displaced by d (orthogonal to everything else) gives a per-drone mean.
    shifted = target.copy()
    shifted[0] += np.array([0.0, 0.0, 0.8])
    E_one, _ = compute_formation_error(shifted, target)
    E_sum_like, _ = compute_formation_error(np.tile(shifted, (2, 1)), np.tile(target, (2, 1)))
    # Duplicating the swarm must not change a *mean* error.
    assert E_sum_like == pytest.approx(E_one, rel=1e-6)
