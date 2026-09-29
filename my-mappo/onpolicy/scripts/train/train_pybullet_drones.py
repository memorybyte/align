import sys
import os
import socket
import setproctitle
import numpy as np
from pathlib import Path
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from onpolicy.config import get_config
from onpolicy.envs.pybullet_drone_env import make_pybullet_drone_env
from onpolicy.envs.env_wrappers import ShareDummyVecEnv, ShareSubprocVecEnv


# ctrl_freq (30 Hz) * EPISODE_LEN_SEC (8 s) in PyBulletDroneWrapper / MultiHoverAviary
EPISODE_LENGTH = 240


def make_train_env(all_args):
    """
    Create training environments.
    
    Args:
        all_args: Argument namespace
        
    Returns:
        Vectorized environment
    """
    def get_env_fn(rank):
        def init_env():
            env = make_pybullet_drone_env(all_args, gui=False)  # No GUI for training
            env.seed(all_args.seed + rank * 1000)
            return env
        return init_env
    
    if all_args.n_rollout_threads == 1:
        return ShareDummyVecEnv([get_env_fn(0)])
    else:
        return ShareSubprocVecEnv([get_env_fn(i) for i in range(all_args.n_rollout_threads)])


def make_eval_env(all_args):
    """
    Create evaluation environments.
    
    Args:
        all_args: Argument namespace
        
    Returns:
        Vectorized environment
    """
    def get_env_fn(rank):
        def init_env():
            env = make_pybullet_drone_env(all_args, gui=False)
            env.seed(all_args.seed * 50000 + rank * 10000)
            return env
        return init_env
    
    if all_args.n_eval_rollout_threads == 1:
        return ShareDummyVecEnv([get_env_fn(0)])
    else:
        return ShareSubprocVecEnv([get_env_fn(i) for i in range(all_args.n_eval_rollout_threads)])


def parse_args(args, parser):
    """
    Parse command line arguments for PyBullet drones training.
    
    Adds environment-specific arguments to the base MAPPO config.

    """
    parser.add_argument('--resume_checkpoint', type=str, default=None,
                        help="Path to full-state resume checkpoint (.pt). If set, resumes optimizer/RNG/progress too")
    parser.add_argument('--resume_partial_checkpoint', type=str, default=None,
                        help="Path to models folder containing actor.pt and critic.pt. Loads weights only and starts fresh optimizer/RNG/progress")
    
    all_args = parser.parse_known_args(args)[0]
    if not hasattr(all_args, 'model') or all_args.model is None:
        all_args.model = 'ma_lstm'
    
    return all_args


def main(args):
    """Main training function."""
    parser = get_config()
    all_args = parse_args(args, parser)
    
    all_args.env_name = "pybullet-drones"
    print("Using MA-LSTM-PPO model with LSTM actor and centralized critic")

    if all_args.resume_partial_checkpoint and all_args.resume_checkpoint:
        raise ValueError("Use either --resume_partial_checkpoint or --resume_checkpoint, not both.")

    if all_args.resume_partial_checkpoint:
        partial_dir = all_args.resume_partial_checkpoint
        actor_path = os.path.join(partial_dir, "actor.pt")
        critic_path = os.path.join(partial_dir, "critic.pt")
        if not os.path.isfile(actor_path) or not os.path.isfile(critic_path):
            raise FileNotFoundError(
                f"resume_partial_checkpoint must point to a models folder containing actor.pt and critic.pt: {partial_dir}"
            )
        # Reuse existing base-runner restore path (weights only) while keeping a clearer CLI.
        all_args.model_dir = partial_dir
        print(f"Loading partial checkpoint weights from: {partial_dir}")

    if all_args.resume_checkpoint:
        print(f"Resuming from checkpoint: {all_args.resume_checkpoint}")
    all_args.use_recurrent_policy = True
    all_args.use_naive_recurrent_policy = False
    all_args.algorithm_name = "rmappo"
    all_args.recurrent_N = 2
    
    all_args.lr = 5e-4
    all_args.use_linear_lr_decay = True
    all_args.gamma = 0.99
    all_args.gae_lambda = 0.95
    all_args.clip_param = 0.2
    # With ctrl_freq=30 (default) and 8s episodes: 8*30 = 240 steps
    all_args.episode_length = EPISODE_LENGTH
    all_args.hidden_size = 256
    
    # CUDA setup
    if all_args.cuda and torch.cuda.is_available():
        print("Using GPU for training...")
        device = torch.device("cuda:0")
        torch.set_num_threads(all_args.n_training_threads)
        if all_args.cuda_deterministic:
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
    else:
        print("Using CPU for training...")
        device = torch.device("cpu")
        torch.set_num_threads(all_args.n_training_threads)
    
    run_dir = Path(os.path.split(os.path.dirname(os.path.abspath(__file__)))[0] + "/results") \
        / all_args.env_name / f"drones_{all_args.num_drones}" / all_args.algorithm_name / all_args.experiment_name
    if not run_dir.exists():
        os.makedirs(str(run_dir))
    
    if not run_dir.exists():
        curr_run = 'run1'
    else:
        exst_run_nums = [int(str(folder.name).split('run')[1]) 
                        for folder in run_dir.iterdir() 
                        if str(folder.name).startswith('run')]
        if len(exst_run_nums) == 0:
            curr_run = 'run1'
        else:
            curr_run = f'run{max(exst_run_nums) + 1}'
    run_dir = run_dir / curr_run
    if not run_dir.exists():
        os.makedirs(str(run_dir))
    
    setproctitle.setproctitle(
        f"{all_args.model}-{all_args.env_name}-{all_args.experiment_name}@{all_args.user_name}"
    )
    
    torch.manual_seed(all_args.seed)
    torch.cuda.manual_seed_all(all_args.seed)
    np.random.seed(all_args.seed)
    
    # Create environments
    envs = make_train_env(all_args)
    eval_envs = make_eval_env(all_args) if all_args.use_eval else None
    num_agents = all_args.num_drones
    
    config = {
        "all_args": all_args,
        "envs": envs,
        "eval_envs": eval_envs,
        "num_agents": num_agents,
        "device": device,
        "run_dir": run_dir
    }
    
    # Import and create runner
    # Use shared runner for MA-LSTM (all agents share policy)
    from onpolicy.runner.shared.pybullet_drone_runner import PyBulletDroneRunner as Runner
    
    runner = Runner(config)
    runner.run()
    
    # Cleanup
    envs.close()
    if all_args.use_eval and eval_envs is not envs:
        eval_envs.close()
    
    runner.writter.export_scalars_to_json(str(runner.log_dir + '/summary.json'))
    runner.writter.close()


if __name__ == "__main__":
    main(sys.argv[1:])
