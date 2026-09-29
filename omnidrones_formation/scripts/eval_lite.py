"""
Evaluate a checkpoint on the point-mass FormationNav for every obstacle scenario.

    python scripts/eval_lite.py --checkpoint runs/lite/checkpoint.pt --num_drones 8
    python scripts/eval_lite.py --checkpoint runs/lite/checkpoint.pt --num_drones 16   # scale-up

Writes eval.json / eval.md and one trajectory plot per scenario to --out.
"""

import argparse
import os
import sys

import torch
from torchrl.envs.transforms import InitTracker, TransformedEnv

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir))

from formation_nav import FormationNavConfig, MAPPOLSTM  # noqa: E402
from formation_nav.evaluation import evaluate_scenarios, format_table, save_results  # noqa: E402
from formation_nav.pointmass_env import FormationNavLite  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--num_drones", type=int, default=8)
    p.add_argument("--num_envs", type=int, default=64)
    p.add_argument("--max_episode_length", type=int, default=800)
    p.add_argument("--scenarios", nargs="+", default=["none", "static", "dynamic", "mixed"])
    p.add_argument("--formation", default=None, help="force one formation of the pool")
    p.add_argument("--formation_pool", nargs="+", default=["cube", "sphere", "pyramid", "plane"])
    p.add_argument("--goal_min", type=float, default=8.0)
    p.add_argument("--goal_max", type=float, default=14.0)
    p.add_argument("--hidden_size", type=int, default=256)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--out", default="runs/lite_eval")
    args = p.parse_args()
    os.makedirs(args.out, exist_ok=True)

    task = FormationNavConfig(
        num_drones=args.num_drones,
        formation_pool=tuple(args.formation_pool),
        goal_distance=(args.goal_min, args.goal_max),
    )
    base = FormationNavLite(task, args.num_envs, args.max_episode_length, seed=args.seed)
    env = TransformedEnv(base, InitTracker())
    policy = MAPPOLSTM(dict(hidden_size=args.hidden_size), env.observation_spec, env.action_spec, env.reward_spec)
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    try:
        policy.load_checkpoint(ckpt)
    except RuntimeError:
        print("[info] swarm size differs from training: loading the actor only")
        policy.load_checkpoint(ckpt, actor_only=True)

    results = evaluate_scenarios(
        env, base, policy, args.max_episode_length, scenarios=args.scenarios, formation=args.formation,
        plot_dir=args.out,
    )
    save_results(results, os.path.join(args.out, "eval.json"), os.path.join(args.out, "eval.md"))
    print(format_table(results))


if __name__ == "__main__":
    main()
