"""
Scenario evaluation and trajectory plots, shared by the Isaac Sim and point-mass scripts.

`evaluate_scenarios` runs one deterministic episode per parallel environment for each
scenario (none / static / dynamic / mixed) and reports the mean and std of the episode stats,
so the same policy can be compared with and without obstacles.
"""

import json
import math
from typing import Dict, Iterable, List, Optional

import torch
from tensordict import TensorDictBase
from torchrl.envs.utils import ExplorationType, set_exploration_type

from .core import SCENARIOS

REPORT_KEYS = (
    "success",
    "hold_ratio",
    "formation_error",
    "slot_error",
    "progress",
    "collisions_drone",
    "collisions_obstacle",
    "min_obstacle_clearance",
    "min_separation",
    "smoothness",
    "time_to_form",
    "time_to_goal",
    "crashed",
    "crash_ground",
    "crash_flip",
    "crash_bounds",
    "crash_drone_collision",
    "crash_obstacle_collision",
    "return",
    "episode_len",
)


def _first_episode(traj: TensorDictBase) -> TensorDictBase:
    """Stats of each env's first episode (the step at which it ended)."""
    done = traj.get(("next", "done")).squeeze(-1)  # (E, T)
    T = done.shape[1]
    ended = done.any(dim=1)
    first = torch.where(ended, done.float().argmax(dim=1), torch.full_like(ended, T - 1, dtype=torch.long))
    idx = torch.arange(done.shape[0], device=done.device)
    return traj.get(("next", "stats"))[idx, first]


@torch.no_grad()
def rollout_scenario(env, base_env, policy, scenario: str, max_steps: int, formation: Optional[str] = None,
                     callback=None) -> TensorDictBase:
    base_env.set_scenario(scenario)
    base_env.set_formation(formation)
    with set_exploration_type(ExplorationType.MODE):
        traj = env.rollout(
            max_steps=max_steps,
            policy=policy,
            callback=callback,
            auto_reset=True,
            break_when_any_done=False,
            return_contiguous=False,
        )
    return traj


@torch.no_grad()
def evaluate_scenarios(
    env,
    base_env,
    policy,
    max_steps: int,
    scenarios: Iterable[str] = SCENARIOS,
    formation: Optional[str] = None,
    plot_dir: Optional[str] = None,
    callback_factory=None,
) -> Dict[str, Dict[str, float]]:
    """
    Returns {scenario: {metric_mean, metric_std, ...}} and optionally writes a trajectory plot
    of env 0 per scenario to `plot_dir`.
    """
    results = {}
    core = getattr(base_env, "core", None)
    if core is not None:
        core.freeze_difficulty(1.0)  # evaluate at full obstacle difficulty
    for scenario in scenarios:
        callback = callback_factory(scenario) if callback_factory is not None else None
        traj = rollout_scenario(env, base_env, policy, scenario, max_steps, formation, callback)
        stats = _first_episode(traj)
        res = {}
        for k in REPORT_KEYS:
            v = stats.get(k).float().squeeze(-1)
            if k in ("time_to_form", "time_to_goal"):
                v = v[v >= 0]  # only episodes that got there
            res[f"{k}_mean"] = v.mean().item() if v.numel() else float("nan")
            res[f"{k}_std"] = v.std().item() if v.numel() > 1 else 0.0
        res["episodes"] = int(stats.shape[0])
        results[scenario] = res
        if plot_dir is not None:
            plot_episode(traj, 0, f"{plot_dir}/trajectory_{scenario}.png", title=f"scenario: {scenario}")
    base_env.set_scenario(getattr(getattr(core, "cfg", None), "scenario", "train_mix"))
    base_env.set_formation(None)
    if core is not None:
        core.freeze_difficulty(None)
    return results


def format_table(results: Dict[str, Dict[str, float]], keys: Iterable[str] = REPORT_KEYS) -> str:
    keys = list(keys)
    header = "| metric | " + " | ".join(results) + " |"
    sep = "|---" * (len(results) + 1) + "|"
    rows = [header, sep]
    for k in keys:
        cells = []
        for r in results.values():
            m, s = r.get(f"{k}_mean", math.nan), r.get(f"{k}_std", math.nan)
            cells.append(f"{m:.3f} ± {s:.3f}")
        rows.append(f"| {k} | " + " | ".join(cells) + " |")
    return "\n".join(rows)


def save_results(results, path_json: str, path_md: Optional[str] = None):
    with open(path_json, "w") as f:
        json.dump(results, f, indent=2)
    if path_md is not None:
        with open(path_md, "w") as f:
            f.write(format_table(results) + "\n")


def plot_episode(traj: TensorDictBase, env_idx: int, path: str, title: str = ""):
    """Top-down and side view of env `env_idx`'s first episode (uses the "info" entries)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle

    info = traj.get(("next", "info"))[env_idx]
    done = traj.get(("next", "done"))[env_idx].squeeze(-1)
    T = int(done.float().argmax().item()) + 1 if done.any() else done.shape[0]
    pos = info.get("drone_pos")[:T].cpu()  # (T, n, 3)
    slots = info.get("slot_pos")[:T].cpu()
    goal = info.get("goal_pos")[0].cpu()
    obs_pos = info.get("obstacle_pos")[:T].cpu()
    radius = info.get("obstacle_radius")[0].cpu()
    active = info.get("obstacle_active")[0].cpu() > 0.5
    phase = info.get("phase")[:T, 0].cpu()
    route = info.get("route")[0].cpu() if "route" in info.keys() else None
    n = pos.shape[1]
    # pillars are static (their trajectory is constant), spheres move
    moving = (obs_pos[-1] - obs_pos[0]).norm(dim=-1) > 1e-4

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(14, 6), gridspec_kw={"width_ratios": [1.3, 1]})
    colors = plt.cm.viridis(torch.linspace(0, 1, n).numpy())
    for j in range(obs_pos.shape[1]):
        if not active[j]:
            continue
        if moving[j]:
            ax.plot(obs_pos[:, j, 0], obs_pos[:, j, 1], color="tab:orange", lw=1, alpha=0.5)
            ax.add_patch(Circle(obs_pos[-1, j, :2].tolist(), radius[j].item(), color="tab:orange", alpha=0.6))
        else:
            ax.add_patch(Circle(obs_pos[0, j, :2].tolist(), radius[j].item(), color="dimgray", alpha=0.8))
    for i in range(n):
        ax.plot(pos[:, i, 0], pos[:, i, 1], color=colors[i], lw=1.2)
        ax.plot(pos[0, i, 0], pos[0, i, 1], "o", color=colors[i], ms=4)
    if route is not None:
        ax.plot(route[:, 0], route[:, 1], "--", color="tab:green", lw=1, label="formation route")
    ax.plot(slots[-1, :, 0], slots[-1, :, 1], "x", color="tab:blue", ms=7, label="final slots")
    ax.plot(pos[-1, :, 0], pos[-1, :, 1], "o", mfc="none", color="k", ms=8, label="final drones")
    ax.plot(goal[0], goal[1], "*", color="tab:red", ms=14, label="goal")
    ax.set_aspect("equal")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.legend(loc="best", fontsize=8)
    ax.set_title(f"{title} (top view, {T} steps)")

    t = torch.arange(T)
    for i in range(n):
        ax2.plot(t, pos[:, i, 2], color=colors[i], lw=1)
    ax2.plot(t, slots[:, :, 2].min(1).values, "--", color="tab:blue", lw=1, label="lowest slot")
    for p, name in ((1, "NAV"), (2, "HOLD")):
        idx = (phase >= p).nonzero()
        if len(idx):
            ax2.axvline(idx[0].item(), color="gray", ls=":", lw=1)
            ax2.text(idx[0].item(), ax2.get_ylim()[1] * 0.95, name, fontsize=8)
    ax2.set_xlabel("step")
    ax2.set_ylabel("altitude [m]")
    ax2.set_title("take-off, formation and hold")
    ax2.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
