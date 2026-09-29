"""
Evaluate a trained FormationNav policy in Isaac Sim, with and without obstacles.

    python scripts/evaluate.py checkpoint_path=/path/checkpoint_final.pt
    python scripts/evaluate.py checkpoint_path=... headless=false eval_scenarios=[none]      # watch it
    python scripts/evaluate.py checkpoint_path=... eval_formations=[cube,sphere,pyramid,plane]
    python scripts/evaluate.py checkpoint_path=... task.num_drones=16    # scale-up (same policy)

For every (formation, scenario) pair it writes to `output_dir`:
    results_<formation>.json / .md   mean ± std of success, hold ratio, formation error,
                                     collisions, clearance, smoothness, time to form / goal, ...
    trajectory_<formation>_<scenario>.png   top view + altitude of env 0
    video_<formation>_<scenario>.mp4        viewport recording of the central env
"""

import logging
import os
import sys

import hydra
import torch
from omegaconf import OmegaConf

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir))

from omni_drones import init_simulation_app  # noqa: E402

CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, "cfg")


def write_video(frames, path, fps):
    import imageio

    try:
        imageio.mimsave(path, frames, fps=fps)
    except Exception as e:  # no ffmpeg backend: fall back to GIF
        gif = os.path.splitext(path)[0] + ".gif"
        logging.warning(f"mp4 writing failed ({e}); writing {gif}")
        imageio.mimsave(gif, frames, duration=1.0 / fps)


@hydra.main(version_base=None, config_path=CONFIG_PATH, config_name="eval")
def main(cfg):
    OmegaConf.register_new_resolver("eval", eval)
    OmegaConf.resolve(cfg)
    OmegaConf.set_struct(cfg, False)
    simulation_app = init_simulation_app(cfg)

    from torchrl.envs.transforms import InitTracker, TransformedEnv

    from omni_drones.envs.isaac_env import IsaacEnv
    from omni_drones.learning import ALGOS
    from omni_drones.utils.torchrl import RenderCallback

    import formation_nav.env  # noqa: F401  (registers FormationNav)
    from formation_nav.evaluation import evaluate_scenarios, format_table, save_results
    from formation_nav.mappo_lstm import MAPPOLSTM

    ALGOS["mappo_lstm"] = MAPPOLSTM
    out = os.path.abspath(cfg.output_dir)
    os.makedirs(out, exist_ok=True)

    base_env = IsaacEnv.REGISTRY[cfg.task.name](cfg, headless=cfg.headless)
    env = TransformedEnv(base_env, InitTracker())
    env.set_seed(cfg.seed)
    base_env.enable_render(True)
    base_env.eval()
    env.eval()

    policy = ALGOS[cfg.algo.name.lower()](
        cfg.algo, env.observation_spec, env.action_spec, env.reward_spec, device=base_env.device
    )
    ckpt = torch.load(cfg.checkpoint_path, map_location=base_env.device)
    if hasattr(policy, "load_checkpoint"):
        try:
            policy.load_checkpoint(ckpt)
        except RuntimeError:
            # different swarm size than in training: the critic does not fit, the actor does
            logging.warning("Checkpoint critic does not match this swarm size; loading the actor only.")
            policy.load_checkpoint(ckpt, actor_only=True)
    else:
        policy.load_state_dict(ckpt)

    fps = 1.0 / (cfg.sim.dt * cfg.sim.substeps * cfg.video_interval)
    for formation in cfg.eval_formations:
        tag = formation or "pool"
        callbacks = {}

        def callback_factory(scenario):
            if not cfg.eval_record_video:
                return None
            callbacks[scenario] = RenderCallback(interval=cfg.video_interval)
            return callbacks[scenario]

        results = evaluate_scenarios(
            env, base_env, policy, base_env.max_episode_length,
            scenarios=cfg.eval_scenarios, formation=formation, plot_dir=out, callback_factory=callback_factory,
        )
        for scenario in results:
            src = os.path.join(out, f"trajectory_{scenario}.png")
            if os.path.exists(src):
                os.replace(src, os.path.join(out, f"trajectory_{tag}_{scenario}.png"))
        for scenario, cb in callbacks.items():
            if cb.frames:
                write_video(cb.frames, os.path.join(out, f"video_{tag}_{scenario}.mp4"), fps)
        save_results(results, os.path.join(out, f"results_{tag}.json"), os.path.join(out, f"results_{tag}.md"))
        print(f"\n## formation: {tag}\n")
        print(format_table(results))

    simulation_app.close()


if __name__ == "__main__":
    main()
