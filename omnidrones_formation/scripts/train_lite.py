"""
Train MAPPO-LSTM on the point-mass FormationNav (no Isaac Sim, runs on CPU or GPU).

Useful to debug rewards / observations quickly and to sanity-check that the task is learnable
before launching Isaac Sim. The same checkpoint format is used by the Isaac Sim scripts, and
because the observation / action spaces are identical a lite-trained actor can warm-start
Isaac Sim training (`algo.checkpoint_path=...`).

Example:
    python scripts/train_lite.py --num_envs 128 --iters 300 --scenario train_mix --out runs/lite
"""

import argparse
import json
import os
import sys
import time

import torch
from torchrl.collectors import SyncDataCollector
from torchrl.envs.transforms import InitTracker, TransformedEnv

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir))

from formation_nav import FormationNavConfig, MAPPOLSTM  # noqa: E402
from formation_nav.evaluation import evaluate_scenarios, format_table, save_results  # noqa: E402
from formation_nav.pointmass_env import FormationNavLite  # noqa: E402


def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--num_envs", type=int, default=128)
    p.add_argument("--num_drones", type=int, default=8)
    p.add_argument("--iters", type=int, default=300)
    p.add_argument("--max_episode_length", type=int, default=800)
    p.add_argument("--scenario", default="train_mix", help="none|static|dynamic|mixed|train_mix")
    p.add_argument("--formation", default="dynamic")
    p.add_argument("--goal_min", type=float, default=8.0)
    p.add_argument("--goal_max", type=float, default=14.0)
    p.add_argument("--train_every", type=int, default=64)
    p.add_argument("--hidden_size", type=int, default=256)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out", default="runs/lite")
    p.add_argument("--eval_envs", type=int, default=64)
    return p.parse_args()


def main():
    args = parse()
    os.makedirs(args.out, exist_ok=True)
    torch.manual_seed(args.seed)
    task = FormationNavConfig(
        num_drones=args.num_drones,
        formation=args.formation,
        scenario=args.scenario,
        goal_distance=(args.goal_min, args.goal_max),
    )
    base = FormationNavLite(task, args.num_envs, args.max_episode_length, device=args.device, seed=args.seed)
    env = TransformedEnv(base, InitTracker())
    algo = dict(train_every=args.train_every, hidden_size=args.hidden_size, lr_decay_iters=args.iters)
    policy = MAPPOLSTM(algo, env.observation_spec, env.action_spec, env.reward_spec, device=args.device)
    collector = SyncDataCollector(
        env, policy, frames_per_batch=args.num_envs * args.train_every, total_frames=-1,
        device=args.device, return_same_td=True,
    )

    log = open(os.path.join(args.out, "log.jsonl"), "w")
    start = time.time()
    finished = []
    best = -float("inf")
    for i, data in enumerate(collector):
        if i >= args.iters:
            break
        done = data.get(("next", "done")).squeeze(-1)
        if done.any():
            finished.append(data.get(("next", "stats"))[done].to_tensordict().cpu())
        info = policy.train_op(data.to_tensordict())
        info["iter"] = i
        info["frames"] = (i + 1) * data.numel()
        info["time"] = time.time() - start
        if finished and sum(f.shape[0] for f in finished) >= args.num_envs // 2:
            stats = torch.cat(finished)
            for k in ("return", "success", "formation_error", "slot_error", "progress", "crashed",
                      "collisions_obstacle", "episode_len", "hold_ratio", "crash_ground", "crash_flip",
                      "crash_bounds", "crash_drone_collision", "crash_obstacle_collision"):
                info[f"train/{k}"] = stats.get(k).float().mean().item()
            finished.clear()
            if info["train/return"] > best:
                best = info["train/return"]
                torch.save(policy.checkpoint(), os.path.join(args.out, "checkpoint_best.pt"))
            print(
                f"[{i:4d}] frames {info['frames']:>9d}  return {info['train/return']:8.2f}  "
                f"success {info['train/success']:.2f}  progress {info['train/progress']:.2f}  "
                f"form_err {info['train/formation_error']:.3f}  slot_err {info['train/slot_error']:.2f}  "
                f"crashed {info['train/crashed']:.2f}  len {info['train/episode_len']:.0f}",
                flush=True,
            )
        log.write(json.dumps(info) + "\n")
        log.flush()
        if (i + 1) % 50 == 0:
            torch.save(policy.checkpoint(), os.path.join(args.out, "checkpoint.pt"))
    torch.save(policy.checkpoint(), os.path.join(args.out, "checkpoint.pt"))

    # evaluate the final policy with and without obstacles
    eval_base = FormationNavLite(task, args.eval_envs, args.max_episode_length, device=args.device, seed=1234)
    eval_env = TransformedEnv(eval_base, InitTracker())
    results = evaluate_scenarios(eval_env, eval_base, policy, args.max_episode_length, plot_dir=args.out)
    save_results(results, os.path.join(args.out, "eval.json"), os.path.join(args.out, "eval.md"))
    print(format_table(results))


if __name__ == "__main__":
    main()
