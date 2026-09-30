"""
Train FormationNav in Isaac Sim with OmniDrones.

    python scripts/train.py                                   # MAPPO-LSTM, mixed obstacle scenarios
    python scripts/train.py task.scenario=none                # no obstacles
    python scripts/train.py task.formation=cube task.num_drones=8 algo=mappo   # MLP baseline
    python scripts/train.py checkpoint_path=runs/lite/checkpoint.pt            # warm start

Follows OmniDrones' scripts/train.py; additionally registers the MAPPO-LSTM algorithm and
evaluates every scenario (none / static / dynamic / mixed) at each evaluation.
"""

import logging
import os
import sys

import hydra
import torch
from omegaconf import OmegaConf
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir))

from omni_drones import init_simulation_app  # noqa: E402

CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, "cfg")


def load_policy_checkpoint(policy, path: str):
    ckpt = torch.load(path, map_location="cpu")
    if hasattr(policy, "load_checkpoint"):
        policy.load_checkpoint(ckpt)
    else:
        policy.load_state_dict(ckpt)
    logging.info(f"Loaded checkpoint {path}")


def save_policy_checkpoint(policy, path: str):
    state = policy.checkpoint() if hasattr(policy, "checkpoint") else policy.state_dict()
    torch.save(state, path)
    logging.info(f"Saved checkpoint to {path}")


@hydra.main(version_base=None, config_path=CONFIG_PATH, config_name="train")
def main(cfg):
    OmegaConf.register_new_resolver("eval", eval)
    OmegaConf.resolve(cfg)
    OmegaConf.set_struct(cfg, False)
    simulation_app = init_simulation_app(cfg)

    # everything below needs the running simulation app
    import wandb
    from torchrl.envs.transforms import Compose, InitTracker, TransformedEnv

    from omni_drones.envs.isaac_env import IsaacEnv
    from omni_drones.learning import ALGOS
    from omni_drones.utils.torchrl import EpisodeStats, RenderCallback, SyncDataCollector
    from omni_drones.utils.wandb import init_wandb

    import formation_nav.env  # noqa: F401  (registers FormationNav)
    from formation_nav.evaluation import evaluate_scenarios, format_table, save_results, write_video
    from formation_nav.mappo_lstm import MAPPOLSTM

    ALGOS["mappo_lstm"] = MAPPOLSTM

    run = init_wandb(cfg)
    print(OmegaConf.to_yaml(cfg))
    # everything is also written here, whatever the wandb mode
    out = os.path.abspath(cfg.get("output_dir") or "runs/FormationNav")
    os.makedirs(out, exist_ok=True)
    OmegaConf.save(cfg, os.path.join(out, "config.yaml"))
    print(f"[train] checkpoints and evaluation results go to {out}")

    base_env = IsaacEnv.REGISTRY[cfg.task.name](cfg, headless=cfg.headless)
    env = TransformedEnv(base_env, Compose(InitTracker())).train()
    env.set_seed(cfg.seed)

    frames_per_batch = env.num_envs * int(cfg.algo.train_every)
    total_frames = cfg.get("total_frames", -1) // frames_per_batch * frames_per_batch
    max_iters = cfg.get("max_iters", -1)
    if cfg.algo.get("lr_decay_iters", 0) == -1:
        cfg.algo.lr_decay_iters = max_iters if max_iters > 0 else total_frames // frames_per_batch

    policy = ALGOS[cfg.algo.name.lower()](
        cfg.algo, env.observation_spec, env.action_spec, env.reward_spec, device=base_env.device
    )
    if cfg.get("checkpoint_path"):
        load_policy_checkpoint(policy, cfg.checkpoint_path)
    eval_interval = cfg.get("eval_interval", -1)
    save_interval = cfg.get("save_interval", -1)

    stats_keys = [k for k in base_env.observation_spec.keys(True, True) if isinstance(k, tuple) and k[0] == "stats"]
    episode_stats = EpisodeStats(stats_keys)
    collector = SyncDataCollector(
        env,
        policy=policy,
        frames_per_batch=frames_per_batch,
        total_frames=total_frames,
        device=cfg.sim.device,
        return_same_td=True,
    )

    @torch.no_grad()
    def evaluate(tag: str):
        eval_dir = os.path.join(out, f"eval_{tag}")
        os.makedirs(eval_dir, exist_ok=True)
        base_env.enable_render(True)
        base_env.eval()
        env.eval()
        callbacks = {}

        def callback_factory(scenario):
            if not cfg.get("eval_record_video", True):
                return None
            callbacks[scenario] = RenderCallback(interval=2)
            return callbacks[scenario]

        results = evaluate_scenarios(
            env, base_env, policy, base_env.max_episode_length,
            scenarios=cfg.eval_scenarios, plot_dir=eval_dir, callback_factory=callback_factory,
        )
        print(format_table(results))
        save_results(results, os.path.join(eval_dir, "results.json"), os.path.join(eval_dir, "results.md"))
        info = {f"eval/{s}/{k}": v for s, r in results.items() for k, v in r.items()}
        fps = 0.5 / (cfg.sim.dt * cfg.sim.substeps)  # every 2nd step -> real time
        for scenario, cb in callbacks.items():
            if not cb.frames:
                continue
            write_video(cb.frames, os.path.join(eval_dir, f"video_{scenario}.mp4"), fps)
            if cfg.wandb.mode != "disabled":
                info[f"eval/{scenario}/recording"] = wandb.Video(
                    cb.get_video_array(axes="t c h w"), fps=fps, format="mp4"
                )
        for scenario in results:
            png = os.path.join(eval_dir, f"trajectory_{scenario}.png")
            if os.path.exists(png):
                info[f"eval/{scenario}/trajectory"] = wandb.Image(png)
        base_env.enable_render(not cfg.headless)
        env.train()
        base_env.train()
        collector.reset()  # the evaluation rollouts moved the simulation
        return info

    pbar = tqdm(collector, total=total_frames // frames_per_batch)
    env.train()
    for i, data in enumerate(pbar):
        info = {"env_frames": collector._frames, "rollout_fps": collector._fps}
        episode_stats.add(data.to_tensordict())

        if len(episode_stats) >= base_env.num_envs:
            stats = {
                "train/" + (".".join(k) if isinstance(k, tuple) else k): torch.mean(v.float()).item()
                for k, v in episode_stats.pop().items(True, True)
            }
            info.update(stats)

        info.update(policy.train_op(data.to_tensordict()))

        if eval_interval > 0 and i % eval_interval == 0 and i > 0:
            logging.info(f"Eval at {collector._frames} steps.")
            info.update(evaluate(str(collector._frames)))

        if save_interval > 0 and i % save_interval == 0:
            save_policy_checkpoint(policy, os.path.join(out, f"checkpoint_{collector._frames}.pt"))

        run.log(info)
        pbar.set_postfix({"rollout_fps": collector._fps, "frames": collector._frames})

        if max_iters > 0 and i >= max_iters - 1:
            break

    logging.info(f"Final Eval at {collector._frames} steps.")
    info = {"env_frames": collector._frames}
    info.update(evaluate("final"))
    run.log(info)

    save_policy_checkpoint(policy, os.path.join(out, "checkpoint_final.pt"))
    if cfg.wandb.mode != "disabled":
        wandb.save(os.path.join(out, "checkpoint_final.pt"), base_path=out)
    print(f"[train] done: {os.path.join(out, 'checkpoint_final.pt')}")
    wandb.finish()
    simulation_app.close()


if __name__ == "__main__":
    main()
