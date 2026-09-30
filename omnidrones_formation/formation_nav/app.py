"""
Start Isaac Sim for the FormationNav scripts.

Same as OmniDrones' `init_simulation_app`, but multi-GPU rendering is off by default: on
machines with several GPUs, Isaac Sim otherwise splits rendering across all of them, which only
adds GPU-to-GPU copies for our small render product. Config keys (all optional):
  headless: bool
  multi_gpu: bool          (default False)
  active_gpu: int | null   GPU used for rendering (null: Isaac Sim's choice, normally GPU 0)
"""

import os


def start_simulation_app(cfg):
    from isaacsim import SimulationApp

    config = {
        "headless": bool(cfg.get("headless", True)),
        "anti_aliasing": 1,
        "multi_gpu": bool(cfg.get("multi_gpu", False)),
    }
    if cfg.get("active_gpu") is not None:
        config["active_gpu"] = int(cfg.get("active_gpu"))
    if "EXP_PATH" not in os.environ:
        raise RuntimeError(
            "EXP_PATH is not set: point it at the folder that contains omni.isaac.sim.python.kit "
            "(docs/INSTALL.md, section 2)."
        )
    experience = os.path.join(os.environ["EXP_PATH"], "omni.isaac.sim.python.kit")
    return SimulationApp(config, experience=experience)
