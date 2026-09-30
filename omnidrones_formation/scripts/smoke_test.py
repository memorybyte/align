"""
First-flight check of FormationNav with a scripted policy (no training needed).

Every drone flies straight to its own formation slot (formation_nav/scripted.py). The table
shows whether the drones take off, build the formation (the stacked 3-D ones only hold with the
downwash feed-forward), follow the route to the goal and hold there, and why episodes ended.
Run it once after installing OmniDrones, and after changing the drone or controller settings.

    python scripts/smoke_test.py                                # Isaac Sim, Crazyflie, headless
    python scripts/smoke_test.py headless=false                 # watch it
    python scripts/smoke_test.py record_video=false             # faster, no mp4
    python scripts/smoke_test.py lite=true                      # pure-PyTorch quadrotor model, no Isaac Sim
    python scripts/smoke_test.py task=FormationNavHummingbird
    python scripts/smoke_test.py task.controller.downwash_feedforward=false   # see the stacked drones sink

Expected with the defaults (lite=true gives these numbers; Isaac Sim should be close): success
and hold_ratio near 1 in `none` for every formation, no crashes; in `static` the planned route
avoids the pillars but the drones do not, so some obstacle collisions are normal.
"""

import os
import sys

import hydra
import torch
from omegaconf import OmegaConf

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir))

CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, "cfg")
KEYS = (
    "success", "hold_ratio", "slot_error", "formation_error", "time_to_form", "time_to_goal",
    "crashed", "crash_ground", "crash_flip", "crash_drone_collision", "crash_obstacle_collision",
    "min_separation", "episode_len",
)


def make_lite_env(cfg):
    from formation_nav.core import FormationNavConfig
    from formation_nav.pointmass_env import FormationNavLite

    task = FormationNavConfig.from_dict(OmegaConf.to_container(cfg.task, resolve=True))
    task.dt = cfg.sim.dt * cfg.sim.substeps
    controller = cfg.task.get("controller") or {}
    return FormationNavLite(
        task,
        num_envs=cfg.env.num_envs,
        max_episode_length=cfg.env.max_episode_length,
        seed=cfg.seed,
        dynamics="quadrotor",
        drone=cfg.task.drone_model.name,
        substeps=cfg.sim.substeps,
        downwash_scale=float(cfg.task.get("downwash_scale", 1.0)),
        downwash_feedforward=bool(controller.get("downwash_feedforward", True)),
    )


@hydra.main(version_base=None, config_path=CONFIG_PATH, config_name="smoke")
def main(cfg):
    OmegaConf.resolve(cfg)
    OmegaConf.set_struct(cfg, False)
    torch.manual_seed(cfg.seed)
    # any formation can be checked: add the requested ones to the pool the env samples from
    cfg.task.formation_pool = list(dict.fromkeys(list(cfg.task.formation_pool) + list(cfg.formations)))

    simulation_app = None
    if cfg.lite:
        base_env = make_lite_env(cfg)
    else:
        from formation_nav.app import start_simulation_app

        simulation_app = start_simulation_app(cfg)
        from omni_drones.envs.isaac_env import IsaacEnv

        import formation_nav.env  # noqa: F401  (registers FormationNav)

        base_env = IsaacEnv.REGISTRY[cfg.task.name](cfg, headless=cfg.headless)
        base_env.set_seed(cfg.seed)

    from formation_nav.evaluation import evaluate_scenarios, format_table, save_results, write_video
    from formation_nav.scripted import SlotSeeker

    policy = SlotSeeker(base_env.core.cfg.max_speed, base_env.core.cfg.action_mode)
    backend = "pure-PyTorch quadrotor model" if cfg.lite else "Isaac Sim"
    print(f"[smoke test] {cfg.task.drone_model.name} on {backend}, {cfg.env.num_envs} envs x "
          f"{base_env.core.cfg.num_drones} drones, {base_env.max_episode_length} steps")

    os.makedirs(cfg.output_dir, exist_ok=True)
    record = bool(cfg.get("record_video", True)) and not cfg.lite  # the lite model has no renderer
    if record:
        base_env.enable_render(True)
    all_results = {}
    for formation in cfg.formations:
        plot_dir = os.path.join(cfg.output_dir, formation)
        os.makedirs(plot_dir, exist_ok=True)
        callbacks = {}

        def record_scenario(scenario):
            from omni_drones.utils.torchrl import RenderCallback

            callbacks[scenario] = RenderCallback(interval=cfg.video_interval)
            return callbacks[scenario]

        results = evaluate_scenarios(
            base_env, base_env, policy, base_env.max_episode_length,
            scenarios=cfg.scenarios, formation=formation, plot_dir=plot_dir,
            callback_factory=record_scenario if record else None,
        )
        for scenario, cb in callbacks.items():
            if cb.frames:
                fps = 1.0 / (cfg.sim.dt * cfg.sim.substeps * cfg.video_interval)
                print("[smoke test] video:", write_video(cb.frames, os.path.join(plot_dir, f"video_{scenario}.mp4"), fps))
        print(f"\n### formation: {formation}\n" + format_table(results, KEYS))
        save_results(results, os.path.join(plot_dir, "results.json"), os.path.join(plot_dir, "results.md"))
        all_results[formation] = results

    failed = [f"{f}/{s}" for f, r in all_results.items() for s, v in r.items()
              if s == "none" and (v["success_mean"] < 0.9 or v["crashed_mean"] > 0.0)]
    print("\n[smoke test] " + ("OK: every formation took off, formed, reached the goal and held it without obstacles."
                              if not failed else f"CHECK: {failed} (see the tables and {cfg.output_dir}/*/trajectory_*.png)"))
    if simulation_app is not None:
        simulation_app.close()


if __name__ == "__main__":
    main()
