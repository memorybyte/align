#!/usr/bin/env python
import sys
import os
import socket
import setproctitle
import numpy as np
from pathlib import Path
import torch

from onpolicy.config import get_config
from onpolicy.envs.pybullet_drone_env import make_pybullet_drone_env
from onpolicy.envs.env_wrappers import ShareDummyVecEnv
from onpolicy.scripts.train.train_pybullet_drones import EPISODE_LENGTH

def make_render_env(all_args):
    def get_env_fn(rank):
        def init_env():
            # Same env construction as training (obs size, formation pool, reward),
            # with the GUI enabled for visual rendering.
            env = make_pybullet_drone_env(all_args, gui=True)
            env.seed(all_args.seed + rank * 1000)
            return env
        return init_env
    
    if all_args.n_rollout_threads == 1:
        return ShareDummyVecEnv([get_env_fn(0)])
    else:
        raise NotImplementedError("Only support 1 thread for rendering")

def parse_args(args, parser):
    # Environment specific arguments (num_drones already defined in config.py)
    parser.add_argument('--scenario_name', type=str, default='drones_3',
                        help="Scenario name for organizing results")
    
    all_args = parser.parse_known_args(args)[0]
    
    # Set num_agents based on num_drones
    all_args.num_agents = all_args.num_drones
    
    return all_args


def main(args):
    parser = get_config()
    all_args = parse_args(args, parser)

    # Set algorithm configurations
    print("Using recurrent MAPPO with MA-LSTM policy for rendering")
    all_args.model = 'ma_lstm'  # Use MA-LSTM model architecture
    all_args.use_recurrent_policy = True
    all_args.use_naive_recurrent_policy = False
    all_args.use_wandb = False  # Disable wandb for rendering
    all_args.algorithm_name = "rmappo"
    all_args.env_name = "pybullet-drones"
    all_args.recurrent_N = 2  # LSTM stores both h and c states
    all_args.hidden_size = 256  # must match training (train_pybullet_drones.py)
    # Episode length must match env: 8s * 30Hz ctrl_freq = 240 steps
    all_args.episode_length = EPISODE_LENGTH

    # Validation
    assert all_args.use_render, ("You need to set use_render to True")
    assert all_args.model_dir is not None and all_args.model_dir != "", ("Set model_dir first")
    assert all_args.n_rollout_threads == 1, ("Only support 1 env for rendering")
    
    # Device setup
    if all_args.cuda and torch.cuda.is_available():
        print("Using GPU...")
        device = torch.device("cuda:0")
        torch.set_num_threads(all_args.n_training_threads)
        if all_args.cuda_deterministic:
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
    else:
        print("Using CPU...")
        device = torch.device("cpu")
        torch.set_num_threads(all_args.n_training_threads)

    # Setup directories
    run_dir = Path(os.path.split(os.path.dirname(os.path.abspath(__file__)))[0] + "/results") / "pybullet-drones" / all_args.scenario_name / all_args.algorithm_name / all_args.experiment_name
    
    if not run_dir.exists():
        curr_run = 'run1'
    else:
        exst_run_nums = [int(str(folder.name).split('run')[1]) for folder in run_dir.iterdir() if str(folder.name).startswith('run')]
        if len(exst_run_nums) == 0:
            curr_run = 'run1'
        else:
            curr_run = 'run%i' % (max(exst_run_nums) + 1)
    
    run_dir = run_dir / curr_run
    if not run_dir.exists():
        os.makedirs(str(run_dir))

    # Set process title
    setproctitle.setproctitle(
        str(all_args.algorithm_name) + "-" + 
        str(all_args.env_name) + "-" + 
        str(all_args.experiment_name) + "@" + 
        str(all_args.user_name)
    )

    # Seed
    torch.manual_seed(all_args.seed)
    torch.cuda.manual_seed_all(all_args.seed)
    np.random.seed(all_args.seed)

    # Create environment
    envs = make_render_env(all_args)
    num_agents = all_args.num_agents

    config = {
        "all_args": all_args,
        "envs": envs,
        "eval_envs": None,
        "num_agents": num_agents,
        "device": device,
        "run_dir": run_dir
    }

    # Run rendering
    from onpolicy.runner.shared.pybullet_drone_runner import PyBulletDroneRunner as Runner
    
    runner = Runner(config)
    runner.render()
    
    # Cleanup
    envs.close()
    
    if all_args.save_gifs:
        print(f"\nRendering complete! GIF saved to: {runner.gif_dir}/render.gif")
    else:
        print("\nRendering complete!")


if __name__ == "__main__":
    main(sys.argv[1:])
