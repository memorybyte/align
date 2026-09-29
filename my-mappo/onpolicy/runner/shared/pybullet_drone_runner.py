"""
PyBullet Drone Runner for MA-LSTM-PPO.

This runner handles training, evaluation, and rendering for the PyBullet
drone formation flying task with MA-LSTM-PPO.

"""

import os
import time
import json
import random
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for saving plots
import matplotlib.pyplot as plt
from onpolicy.runner.shared.base_runner import Runner


def _t2n(x):
    """Convert torch tensor to numpy array."""
    return x.detach().cpu().numpy()


class PyBulletDroneRunner(Runner):
    """
    Runner for MA-LSTM-PPO training with PyBullet drones.
    
    Handles the training loop, data collection, and evaluation
    for formation flying with multiple drones.
    
    """
    
    def __init__(self, config):
        """
        Initialize the runner.
        
        Overrides parent to handle MA-LSTM policy selection.
        
        Args:
            config: Configuration dictionary
        """
        # Get model type from args
        all_args = config['all_args']
        model_type = getattr(all_args, 'model', 'ma_lstm')
        
        # Store for later use in policy creation
        self._model_type = model_type
        
        # Call parent init (this sets up policy, trainer, buffer)
        super(PyBulletDroneRunner, self).__init__(config)
        
        # Training history for detailed logging and plotting
        self.training_history = {
            'episodes': [],           # Episode numbers
            'timesteps': [],          # Total env steps
            'avg_rewards': [],        # Average episodic reward (logged points)
            'formation_errors': [],   # Formation errors
            'episodic_sum_r_form': [],  # Episodic sum formation reward component
            'episodic_sum_r_nav': [],   # Episodic sum navigation reward component
            'episodic_sum_r_dist': [],  # Episodic sum distance penalty component
            'episodic_sum_r_avoid': [],  # Episodic sum collision penalty component
            'episodic_sum_r_tilt': [],  # Episodic sum tilt penalty component
            'episodic_sum_r_reached': [],  # Episodic sum reaching bonus component
            'episodic_sum_r_vel_penalty': [],  # Episodic sum velocity penalty component
            'episodic_sum_r_smooth': [],  # Episodic sum action smoothness penalty component
            'avg_distances': [],      # Avg distance to target (all drones)
            'per_drone_distances': [],  # Per-drone final distances
            'per_drone_reached': [],  # Per-drone reached target (bool)
            'episode_summaries': [],  # Detailed episode summaries
        }
        
        # Track completed episodes during each training cycle
        self.recent_episode_completions = []

        # Resume state (for checkpoint-based continuation)
        self.start_episode = 0
        self.start_total_num_steps = 0

        resume_ckpt = getattr(self.all_args, 'resume_checkpoint', None)
        if resume_ckpt:
            self._load_training_checkpoint(resume_ckpt)
    
    def _init_policy(self):
        """
        Initialize policy based on model type.
        
        This method is called by parent __init__ to create the policy.
        Override to support MA-LSTM policy.
        """
        # Already handled in parent __init__ based on algorithm_name
        pass
    
    def run(self):
        """
        Main training loop.
        
        Pseudocode:
        for iteration in range(max_iters):
            for t in range(T):
                actions = policy.act(obs, rnn_states, deterministic=False)
                next_obs, rewards, dones, infos = env.step(actions)
                buffer.add(obs, actions, rewards, dones, rnn_states, values, logp)
            advantages, returns = compute_gae(buffer)
            for epoch in range(K):
                for minibatch in buffer.minibatches():
                    loss = ppo_loss(minibatch, clip=eps)
                    optimizer.step(loss)
        """
        self.warmup()
        
        start = time.time()
        episodes = int(self.num_env_steps) // self.episode_length // self.n_rollout_threads
        
        for episode in range(self.start_episode, episodes):
            cnt = 0
            # Learning rate decay
            if self.use_linear_lr_decay:
                self.trainer.policy.lr_decay(episode, episodes)
            
            # Collect rollouts
            for step in range(self.episode_length):
                # Sample actions from policy
                values, actions, action_log_probs, rnn_states, rnn_states_critic, actions_env = \
                    self.collect(step)
                
                # Environment step
                obs, share_obs, rewards, dones, infos, available_actions = self.envs.step(actions_env)
                cnt += 1
                
                data = (obs, share_obs, rewards, dones, infos, values, actions, 
                       action_log_probs, rnn_states, rnn_states_critic)
                
                # Insert data into buffer
                self.insert(data)
            
            print(f"Episode {episode} collected with {cnt} steps.")
            # Compute returns and update network
            self.compute()
            train_infos = self.train()
            
            # Post processing
            total_num_steps = (episode + 1) * self.episode_length * self.n_rollout_threads
            
            # Save model
            if episode % self.save_interval == 0 or episode == episodes - 1:
                self.save()
                self._save_training_checkpoint(episode, total_num_steps)
            
            # Log information
            if episode % self.log_interval == 0:
                end = time.time()
                print(f"\n{'='*80}")
                print(f" PyBullet Drones | Algo {self.algorithm_name} | Exp {self.experiment_name}")
                print(f" Episode {episode}/{episodes} | Steps {total_num_steps}/{self.num_env_steps} | "
                      f"FPS {int(total_num_steps / (end - start))}")
                print(f"{'='*80}")
                
                # Log environment-specific info
                env_infos = self._collect_env_infos(infos)

                # Prefer true completed-episode returns from this logging window.
                completed_episode_rewards = [
                    float(c['episode_cumulative_reward'])
                    for c in self.recent_episode_completions
                    if c.get('episode_cumulative_reward') is not None
                ]
                if completed_episode_rewards:
                    avg_reward = float(np.mean(completed_episode_rewards))
                else:
                    # Fallback when no episode completed in this logging window.
                    avg_reward = float(np.mean(self.buffer.rewards) * self.episode_length)
                train_infos["average_episode_rewards"] = avg_reward
                print(f"\n  Average episodic reward: {avg_reward:.4f}")
                
                # Log formation metrics
                formation_error = None
                if env_infos.get('formation_error'):
                    formation_error = np.mean(env_infos['formation_error'])
                    print(f"  Formation error: {formation_error:.4f}")
                
                # Collect detailed episode summaries from infos
                self._log_episode_details(infos, episode, total_num_steps, avg_reward, formation_error, env_infos)
                
                self.log_train(train_infos, total_num_steps)
                self.log_env(env_infos, total_num_steps)
                print(f"{'='*80}\n")
            
            # Evaluation
            if episode % self.eval_interval == 0 and self.use_eval:
                self.eval(total_num_steps)
        
        self._save_training_history()
        self._generate_training_plots()
        self._save_last20_metrics()

    def _save_training_checkpoint(self, episode, total_num_steps):
        """Save full training state for resume (weights, optimizers, RNG, counters)."""
        ckpt = {
            'episode': int(episode + 1),
            'total_num_steps': int(total_num_steps),
            'actor_state_dict': self.trainer.policy.actor.state_dict(),
            'critic_state_dict': self.trainer.policy.critic.state_dict(),
            'actor_optimizer_state_dict': self.trainer.policy.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.trainer.policy.critic_optimizer.state_dict(),
            'python_rng_state': random.getstate(),
            'numpy_rng_state': np.random.get_state(),
            'torch_rng_state': torch.get_rng_state(),
        }

        if self.trainer.value_normalizer is not None:
            ckpt['value_normalizer_state_dict'] = self.trainer.value_normalizer.state_dict()

        if torch.cuda.is_available():
            ckpt['torch_cuda_rng_state_all'] = torch.cuda.get_rng_state_all()

        ckpt_path = os.path.join(self.save_dir, 'resume_checkpoint.pt')
        torch.save(ckpt, ckpt_path)

    def _load_training_checkpoint(self, checkpoint_path):
        """Load full training state for resume."""
        if not os.path.isfile(checkpoint_path):
            raise FileNotFoundError(f"Resume checkpoint not found: {checkpoint_path}")

        # Full-state resume checkpoint contains Python/NumPy RNG states, not only tensors.
        # PyTorch 2.6 defaults torch.load(..., weights_only=True), which rejects these objects.
        # Explicitly set weights_only=False for trusted local checkpoints.
        try:
            ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        except TypeError:
            # Backward compatibility with older PyTorch versions without weights_only arg.
            ckpt = torch.load(checkpoint_path, map_location=self.device)

        # Networks
        self.trainer.policy.actor.load_state_dict(ckpt['actor_state_dict'])
        self.trainer.policy.critic.load_state_dict(ckpt['critic_state_dict'])

        # Optimizers
        self.trainer.policy.actor_optimizer.load_state_dict(ckpt['actor_optimizer_state_dict'])
        self.trainer.policy.critic_optimizer.load_state_dict(ckpt['critic_optimizer_state_dict'])

        # Value normalizer (if enabled in config and present in checkpoint)
        if self.trainer.value_normalizer is not None and 'value_normalizer_state_dict' in ckpt:
            self.trainer.value_normalizer.load_state_dict(ckpt['value_normalizer_state_dict'])

        # RNG states
        if 'python_rng_state' in ckpt:
            random.setstate(ckpt['python_rng_state'])
        if 'numpy_rng_state' in ckpt:
            np.random.set_state(ckpt['numpy_rng_state'])
        if 'torch_rng_state' in ckpt:
            torch.set_rng_state(ckpt['torch_rng_state'])
        if torch.cuda.is_available() and 'torch_cuda_rng_state_all' in ckpt:
            torch.cuda.set_rng_state_all(ckpt['torch_cuda_rng_state_all'])

        # Progress counters
        self.start_episode = int(ckpt.get('episode', 0))
        self.start_total_num_steps = int(ckpt.get('total_num_steps', 0))

        print(f"[INFO] Resumed full training state from: {checkpoint_path}")
        print(f"[INFO] Resume episode: {self.start_episode}, total steps: {self.start_total_num_steps}")
    
    def warmup(self):
        """
        Initialize buffer with first observation.
        
        Reset environments and store initial observations.
        """
        # Reset env and get initial observations
        obs, share_obs, available_actions = self.envs.reset()
        
        # Handle observation shapes
        if not isinstance(obs, np.ndarray):
            obs = np.array(obs)
        if not isinstance(share_obs, np.ndarray):
            share_obs = np.array(share_obs)
        
        # Shape: (n_rollout_threads, num_agents, obs_dim)
        if len(obs.shape) == 2:
            obs = np.expand_dims(obs, 0)
        if len(share_obs.shape) == 2:
            share_obs = np.expand_dims(share_obs, 0)
        
        # Store in buffer
        self.buffer.share_obs[0] = share_obs.copy()
        self.buffer.obs[0] = obs.copy()
    
    @torch.no_grad()
    def collect(self, step):
        """
        Collect experience for one step.
        
        Sample actions from policy and prepare for environment step.
        
        Args:
            step: Current step in episode
            
        Returns:
            Tuple of (values, actions, action_log_probs, rnn_states, rnn_states_critic, actions_env)
        """
        self.trainer.prep_rollout()
        
        # Get actions from policy
        value, action, action_log_prob, rnn_states, rnn_states_critic = \
            self.trainer.policy.get_actions(
                np.concatenate(self.buffer.share_obs[step]),
                np.concatenate(self.buffer.obs[step]),
                np.concatenate(self.buffer.rnn_states[step]),
                np.concatenate(self.buffer.rnn_states_critic[step]),
                np.concatenate(self.buffer.masks[step])
            )
        
        # Split by environment
        values = np.array(np.split(_t2n(value), self.n_rollout_threads))
        actions = np.array(np.split(_t2n(action), self.n_rollout_threads))
        action_log_probs = np.array(np.split(_t2n(action_log_prob), self.n_rollout_threads))
        
        # Policy returns (n_rollout * n_agents, recurrent_N, hidden_size), agent-major
        # like every other batched tensor here. Buffer expects
        # (n_rollout, n_agents, recurrent_N, hidden_size).
        rnn_states = np.array(np.split(_t2n(rnn_states), self.n_rollout_threads))
        rnn_states_critic = np.array(np.split(_t2n(rnn_states_critic), self.n_rollout_threads))
        
        # Actions are v_des = [vx, vy, vz, throttle] in [-1, 1]
        actions_env = actions.copy()
        
        return values, actions, action_log_probs, rnn_states, rnn_states_critic, actions_env
    
    def insert(self, data):
        """
        Insert collected data into buffer.
        
        Args:
            data: Tuple of (obs, share_obs, rewards, dones, infos, values, 
                           actions, action_log_probs, rnn_states, rnn_states_critic)
        """
        obs, share_obs, rewards, dones, infos, values, actions, \
            action_log_probs, rnn_states, rnn_states_critic = data

        # ShareDummyVecEnv can return infos as a numpy object array.
        if isinstance(infos, np.ndarray):
            infos = infos.tolist()
        
        # Convert to numpy if needed
        if not isinstance(obs, np.ndarray):
            obs = np.array(obs)
        if not isinstance(share_obs, np.ndarray):
            share_obs = np.array(share_obs)
        if not isinstance(rewards, np.ndarray):
            rewards = np.array(rewards)
        if not isinstance(dones, np.ndarray):
            dones = np.array(dones)
        
        # Handle shapes
        if len(obs.shape) == 2:
            obs = np.expand_dims(obs, 0)
        if len(share_obs.shape) == 2:
            share_obs = np.expand_dims(share_obs, 0)
        if len(rewards.shape) == 2:
            rewards = np.expand_dims(rewards, 0)
        if len(dones.shape) == 1:
            dones = np.expand_dims(dones, 0)
        if len(dones.shape) == 2:
            dones = np.expand_dims(dones, -1)
        
        # Reset RNN states where episode ended
        done_mask = dones.squeeze(-1) == True
        for i in range(self.n_rollout_threads):
            for j in range(self.num_agents):
                if done_mask[i, j]:
                    rnn_states[i, j] = np.zeros((self.recurrent_N, self.hidden_size), dtype=np.float32)
                    rnn_states_critic[i, j] = np.zeros(self.buffer.rnn_states_critic.shape[3:], dtype=np.float32)
                    
                    # Capture episode summary when episode completes
                    if isinstance(infos, (list, tuple)) and i < len(infos):
                        env_info = infos[i]
                        if isinstance(env_info, np.ndarray):
                            env_info = env_info.tolist()
                        if isinstance(env_info, (list, tuple)) and j < len(env_info):
                            agent_info = env_info[j]
                            if isinstance(agent_info, dict) and 'episode_summary' in agent_info:
                                self.recent_episode_completions.append({
                                    'env_idx': i,
                                    'agent_id': j,
                                    'episode_cumulative_reward': agent_info.get('episode_cumulative_reward', None),
                                    'summary': {
                                        **agent_info['episode_summary'],
                                        'episode_steps': agent_info.get('episode_steps', 0),
                                        'episode_reward': agent_info.get('episode_cumulative_reward', 0.0),
                                    }
                                })
        
        # Create masks
        masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
        masks[done_mask] = 0.0
        
        # Insert into buffer
        self.buffer.insert(share_obs, obs, rnn_states, rnn_states_critic, 
                          actions, action_log_probs, values, rewards, masks)
    
    def _collect_env_infos(self, infos):
        """
        Collect environment-specific information for logging.
        
        Args:
            infos: List of info dicts from environment
            
        Returns:
            Dict of aggregated info
        """
        env_infos = {}

        if isinstance(infos, np.ndarray):
            infos = infos.tolist()
        
        # Collect individual rewards
        for agent_id in range(self.num_agents):
            idv_rews = []
            form_errors = []
            form_rewards = []
            nav_rewards = []
            dist_rewards = []
            avoid_rewards = []
            tilt_rewards = []
            reached_rewards = []
            vel_penalty_rewards = []
            smooth_rewards = []
            for info in infos:
                if isinstance(info, dict):
                    if 'individual_reward' in info:
                        idv_rews.append(info['individual_reward'])
                    if 'formation_error' in info:
                        form_errors.append(info['formation_error'])
                    if 'r_form' in info:
                        form_rewards.append(info['r_form'])
                    if 'r_nav' in info:
                        nav_rewards.append(info['r_nav'])
                    if 'r_dist' in info:
                        dist_rewards.append(info['r_dist'])
                    if 'r_avoid' in info:
                        avoid_rewards.append(info['r_avoid'])
                    if 'r_tilt' in info:
                        tilt_rewards.append(info['r_tilt'])
                    if 'r_reached' in info:
                        reached_rewards.append(info['r_reached'])
                    if 'r_vel_penalty' in info:
                        vel_penalty_rewards.append(info['r_vel_penalty'])
                    if 'r_smooth' in info:
                        smooth_rewards.append(info['r_smooth'])
                elif isinstance(info, np.ndarray):
                    info = info.tolist()
                    if len(info) > agent_id:
                        if 'individual_reward' in info[agent_id]:
                            idv_rews.append(info[agent_id]['individual_reward'])
                        if 'formation_error' in info[agent_id]:
                            form_errors.append(info[agent_id]['formation_error'])
                        if 'r_form' in info[agent_id]:
                            form_rewards.append(info[agent_id]['r_form'])
                        if 'r_nav' in info[agent_id]:
                            nav_rewards.append(info[agent_id]['r_nav'])
                        if 'r_dist' in info[agent_id]:
                            dist_rewards.append(info[agent_id]['r_dist'])
                        if 'r_avoid' in info[agent_id]:
                            avoid_rewards.append(info[agent_id]['r_avoid'])
                        if 'r_tilt' in info[agent_id]:
                            tilt_rewards.append(info[agent_id]['r_tilt'])
                        if 'r_reached' in info[agent_id]:
                            reached_rewards.append(info[agent_id]['r_reached'])
                        if 'r_vel_penalty' in info[agent_id]:
                            vel_penalty_rewards.append(info[agent_id]['r_vel_penalty'])
                        if 'r_smooth' in info[agent_id]:
                            smooth_rewards.append(info[agent_id]['r_smooth'])
                elif isinstance(info, list) and len(info) > agent_id:
                    if 'individual_reward' in info[agent_id]:
                        idv_rews.append(info[agent_id]['individual_reward'])
                    if 'formation_error' in info[agent_id]:
                        form_errors.append(info[agent_id]['formation_error'])
                    if 'r_form' in info[agent_id]:
                        form_rewards.append(info[agent_id]['r_form'])
                    if 'r_nav' in info[agent_id]:
                        nav_rewards.append(info[agent_id]['r_nav'])
                    if 'r_dist' in info[agent_id]:
                        dist_rewards.append(info[agent_id]['r_dist'])
                    if 'r_avoid' in info[agent_id]:
                        avoid_rewards.append(info[agent_id]['r_avoid'])
                    if 'r_tilt' in info[agent_id]:
                        tilt_rewards.append(info[agent_id]['r_tilt'])
                    if 'r_reached' in info[agent_id]:
                        reached_rewards.append(info[agent_id]['r_reached'])
                    if 'r_vel_penalty' in info[agent_id]:
                        vel_penalty_rewards.append(info[agent_id]['r_vel_penalty'])
                    if 'r_smooth' in info[agent_id]:
                        smooth_rewards.append(info[agent_id]['r_smooth'])
            
            if idv_rews:
                env_infos[f'agent{agent_id}/individual_rewards'] = idv_rews
            if form_errors:
                env_infos[f'agent{agent_id}/formation_error'] = form_errors
            if form_rewards:
                env_infos[f'agent{agent_id}/r_form'] = form_rewards
            if nav_rewards:
                env_infos[f'agent{agent_id}/r_nav'] = nav_rewards
            if dist_rewards:
                env_infos[f'agent{agent_id}/r_dist'] = dist_rewards
            if avoid_rewards:
                env_infos[f'agent{agent_id}/r_avoid'] = avoid_rewards
            if tilt_rewards:
                env_infos[f'agent{agent_id}/r_tilt'] = tilt_rewards
            if reached_rewards:
                env_infos[f'agent{agent_id}/r_reached'] = reached_rewards
            if vel_penalty_rewards:
                env_infos[f'agent{agent_id}/r_vel_penalty'] = vel_penalty_rewards
            if smooth_rewards:
                env_infos[f'agent{agent_id}/r_smooth'] = smooth_rewards
        
        # Aggregate formation error
        if env_infos.get('agent0/formation_error'):
            env_infos['formation_error'] = env_infos['agent0/formation_error']

        # Aggregate formation reward across all agents (r_form is typically shared)
        all_r_form = []
        for agent_id in range(self.num_agents):
            all_r_form.extend(env_infos.get(f'agent{agent_id}/r_form', []))
        if all_r_form:
            env_infos['r_form'] = all_r_form

        # Aggregate navigation reward across all agents
        all_r_nav = []
        for agent_id in range(self.num_agents):
            all_r_nav.extend(env_infos.get(f'agent{agent_id}/r_nav', []))
        if all_r_nav:
            env_infos['r_nav'] = all_r_nav

        # Aggregate distance penalty across all agents
        all_r_dist = []
        for agent_id in range(self.num_agents):
            all_r_dist.extend(env_infos.get(f'agent{agent_id}/r_dist', []))
        if all_r_dist:
            env_infos['r_dist'] = all_r_dist

        # Aggregate collision avoidance reward across all agents
        all_r_avoid = []
        for agent_id in range(self.num_agents):
            all_r_avoid.extend(env_infos.get(f'agent{agent_id}/r_avoid', []))
        if all_r_avoid:
            env_infos['r_avoid'] = all_r_avoid

        # Aggregate tilt penalty across all agents
        all_r_tilt = []
        for agent_id in range(self.num_agents):
            all_r_tilt.extend(env_infos.get(f'agent{agent_id}/r_tilt', []))
        if all_r_tilt:
            env_infos['r_tilt'] = all_r_tilt

        # Aggregate reaching bonus across all agents
        all_r_reached = []
        for agent_id in range(self.num_agents):
            all_r_reached.extend(env_infos.get(f'agent{agent_id}/r_reached', []))
        if all_r_reached:
            env_infos['r_reached'] = all_r_reached

        # Aggregate velocity penalty across all agents
        all_r_vel_penalty = []
        for agent_id in range(self.num_agents):
            all_r_vel_penalty.extend(env_infos.get(f'agent{agent_id}/r_vel_penalty', []))
        if all_r_vel_penalty:
            env_infos['r_vel_penalty'] = all_r_vel_penalty

        # Aggregate action smoothness penalty across all agents
        all_r_smooth = []
        for agent_id in range(self.num_agents):
            all_r_smooth.extend(env_infos.get(f'agent{agent_id}/r_smooth', []))
        if all_r_smooth:
            env_infos['r_smooth'] = all_r_smooth

        return env_infos

    @torch.no_grad()
    def eval(self, total_num_steps):
        """
        Evaluate the current policy.
        
        Args:
            total_num_steps: Current total training steps (for logging)
        """
        eval_episode_rewards = []
        eval_formation_errors = []
        eval_obs, eval_share_obs, _ = self.eval_envs.reset()
        
        eval_rnn_states = np.zeros(
            (self.n_eval_rollout_threads, *self.buffer.rnn_states.shape[2:]), 
            dtype=np.float32
        )
        eval_masks = np.ones(
            (self.n_eval_rollout_threads, self.num_agents, 1), 
            dtype=np.float32
        )
        
        for eval_step in range(self.episode_length):
            self.trainer.prep_rollout()
            
            # Get deterministic actions
            eval_action, eval_rnn_states = self.trainer.policy.act(
                np.concatenate(eval_obs),
                np.concatenate(eval_rnn_states),
                np.concatenate(eval_masks),
                deterministic=True
            )
            eval_actions = np.array(np.split(_t2n(eval_action), self.n_eval_rollout_threads))
            eval_rnn_states = np.array(np.split(_t2n(eval_rnn_states), self.n_eval_rollout_threads))
            
            # Step environment
            eval_obs, eval_share_obs, eval_rewards, eval_dones, eval_infos, _ = \
                self.eval_envs.step(eval_actions)
            eval_episode_rewards.append(eval_rewards)
            for env_info in eval_infos:
                if len(env_info) > 0 and 'formation_error' in env_info[0]:
                    eval_formation_errors.append(env_info[0]['formation_error'])
            
            # Reset RNN states for done episodes
            eval_dones_arr = np.array(eval_dones).reshape(self.n_eval_rollout_threads, self.num_agents)
            eval_rnn_states[eval_dones_arr] = 0.0
            eval_masks = np.ones(
                (self.n_eval_rollout_threads, self.num_agents, 1), 
                dtype=np.float32
            )
            eval_masks[eval_dones_arr] = 0.0
        
        # Log evaluation results
        eval_episode_rewards = np.array(eval_episode_rewards)
        eval_env_infos = {
            'eval_average_episode_rewards': np.sum(eval_episode_rewards, axis=0)
        }
        if eval_formation_errors:
            eval_env_infos['eval_formation_error'] = eval_formation_errors
        eval_average_episode_rewards = np.mean(eval_env_infos['eval_average_episode_rewards'])
        print(f"Eval average episode rewards: {eval_average_episode_rewards}")
        self.log_env(eval_env_infos, total_num_steps)
    
    @torch.no_grad()
    def render(self):
        """
        Render episodes with the trained policy.
        Loops continuously until user closes the window or presses Ctrl+C.
        """
        envs = self.envs
        
        # Create gif directory if saving gifs
        if self.all_args.save_gifs:
            print("\n[INFO] GIF generation is not currently supported for PyBullet drones.")
            print("[INFO] The PyBullet environment's render() method returns text output only.")
            print("[INFO] To visualize your trained model, run without --save_gifs flag.")
            print("[INFO] The GUI window will show the drones flying in real-time.\n")
            return
        
        print("\n[INFO] Starting visualization. Press Ctrl+C to stop and close the window.\n")
        
        all_frames = []
        episode_count = 0
        
        try:
            # Loop indefinitely until user interrupts
            while True:
                # Run the specified number of episodes, then repeat
                for episode in range(self.all_args.render_episodes):
                    episode_num = episode_count + episode
                    obs, share_obs, available_actions = envs.reset()
                    
                    # Draw visual markers for initial and target positions
                    try:
                        # Access the actual environment from the vectorized wrapper
                        actual_env = envs.envs[0] if hasattr(envs, 'envs') else envs
                        actual_env.draw_position_markers()
                        print(f"\n[Episode {episode_num}] Visual markers drawn:")
                        print(f"  Green spheres = Initial positions")
                        print(f"  Red spheres = Target positions")
                        print(f"  Gray lines = Initial → Target path\n")
                    except Exception as e:
                        print(f"[WARNING] Could not draw markers: {e}")
                    
                    if self.all_args.save_gifs:
                        try:
                            render_output = envs.render('rgb_array')
                            if render_output is not None:
                                image = render_output[0][0] if isinstance(render_output, list) else render_output
                                all_frames.append(image)
                        except Exception as e:
                            print(f"Warning: Failed to capture initial frame for episode {episode_num}: {e}")
                    else:
                        envs.render('human')
                    
                    rnn_states = np.zeros(
                        (self.n_rollout_threads, self.num_agents, self.recurrent_N, self.hidden_size), 
                        dtype=np.float32
                    )
                    masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
                    
                    episode_rewards = []
                    
                    for step in range(self.episode_length):
                        calc_start = time.time()
                        
                        self.trainer.prep_rollout()
                        action, rnn_states = self.trainer.policy.act(
                            np.concatenate(obs),
                            np.concatenate(rnn_states),
                            np.concatenate(masks),
                            deterministic=True
                        )
                        actions = np.array(np.split(_t2n(action), self.n_rollout_threads))
                        rnn_states = np.array(np.split(_t2n(rnn_states), self.n_rollout_threads))
                        
                        # Step environment
                        obs, share_obs, rewards, dones, infos, available_actions = envs.step(actions)
                        episode_rewards.append(rewards)
                        
                        # Reset RNN states for done episodes
                        dones_arr = np.array(dones)
                        if len(dones_arr.shape) == 1:
                            dones_arr = np.expand_dims(dones_arr, 0)
                        
                        rnn_states[dones_arr == True] = np.zeros(
                            ((dones_arr == True).sum(), self.recurrent_N, self.hidden_size), 
                            dtype=np.float32
                        )
                        masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
                        masks[dones_arr == True] = np.zeros(((dones_arr == True).sum(), 1), dtype=np.float32)
                        
                        if self.all_args.save_gifs:
                            try:
                                render_output = envs.render('rgb_array')
                                if render_output is not None:
                                    image = render_output[0][0] if isinstance(render_output, list) else render_output
                                    all_frames.append(image)
                            except Exception as e:
                                print(f"Warning: Failed to capture frame at step {step}: {e}")
                            calc_end = time.time()
                            elapsed = calc_end - calc_start
                            if elapsed < self.all_args.ifi:
                                time.sleep(self.all_args.ifi - elapsed)
                        else:
                            envs.render('human')
                    
                    print(f"Episode {episode_num} average rewards: {np.mean(np.sum(np.array(episode_rewards), axis=0))}")
                
                episode_count += self.all_args.render_episodes
                
        except KeyboardInterrupt:
            print(f"\n\n[INFO] Visualization stopped by user after {episode_count} episodes.")
            print("[INFO] Closing PyBullet window...")
    
    def _log_episode_details(self, infos, episode, total_num_steps, avg_reward, formation_error, env_infos=None):
        """
        Log detailed episode information including positions, distances, and directions.
        
        Args:
            infos: Info dicts from the last step (tuple of env infos across rollout threads)
            episode: Current episode number
            total_num_steps: Total training steps so far
            avg_reward: Average episodic reward at this logging point
            formation_error: Formation error (or None)
            env_infos: Aggregated environment infos from _collect_env_infos()
        """
        # First check if we have any episode completions from this rollout period
        episode_details_to_show = None

        if isinstance(infos, np.ndarray):
            infos = infos.tolist()
        
        # CRITICAL: Extract components from recent_episode_completions BEFORE clearing it
        episodic_sum_r_form = None
        episodic_sum_r_nav = None
        episodic_sum_r_dist = None
        episodic_sum_r_avoid = None
        episodic_sum_r_tilt = None
        episodic_sum_r_reached = None
        episodic_sum_r_vel_penalty = None
        episodic_sum_r_smooth = None
        episodic_form_totals = []
        episodic_nav_totals = []
        episodic_dist_totals = []
        episodic_avoid_totals = []
        episodic_tilt_totals = []
        episodic_reached_totals = []
        episodic_vel_penalty_totals = []
        episodic_smooth_totals = []
        
        if self.recent_episode_completions:
            for completion in self.recent_episode_completions:
                summary = completion.get('summary', {})
                episode_steps = int(summary.get('episode_steps', self.episode_length))
                episode_steps = max(1, episode_steps)

                r_form = summary.get('episode_avg_r_form', None)
                r_nav = summary.get('episode_avg_r_nav', None)
                r_dist = summary.get('episode_avg_r_dist', None)
                r_avoid = summary.get('episode_avg_r_avoid', None)
                r_tilt = summary.get('episode_avg_r_tilt', None)
                r_reached = summary.get('episode_avg_r_reached', None)
                r_vel_penalty = summary.get('episode_avg_r_vel_penalty', None)
                r_smooth = summary.get('episode_avg_r_smooth', None)

                if r_form is not None:
                    episodic_form_totals.append(float(r_form) * episode_steps)
                if r_nav is not None:
                    episodic_nav_totals.append(float(r_nav) * episode_steps)
                if r_dist is not None:
                    episodic_dist_totals.append(float(r_dist) * episode_steps)
                if r_avoid is not None:
                    episodic_avoid_totals.append(float(r_avoid) * episode_steps)
                if r_tilt is not None:
                    episodic_tilt_totals.append(float(r_tilt) * episode_steps)
                if r_reached is not None:
                    episodic_reached_totals.append(float(r_reached) * episode_steps)
                if r_vel_penalty is not None:
                    episodic_vel_penalty_totals.append(float(r_vel_penalty) * episode_steps)
                if r_smooth is not None:
                    episodic_smooth_totals.append(float(r_smooth) * episode_steps)
            
            # Compute means from aggregated totals
            if episodic_form_totals:
                episodic_sum_r_form = float(np.mean(episodic_form_totals))
            if episodic_nav_totals:
                episodic_sum_r_nav = float(np.mean(episodic_nav_totals))
            if episodic_dist_totals:
                episodic_sum_r_dist = float(np.mean(episodic_dist_totals))
            if episodic_avoid_totals:
                episodic_sum_r_avoid = float(np.mean(episodic_avoid_totals))
            if episodic_tilt_totals:
                episodic_sum_r_tilt = float(np.mean(episodic_tilt_totals))
            if episodic_reached_totals:
                episodic_sum_r_reached = float(np.mean(episodic_reached_totals))
            if episodic_vel_penalty_totals:
                episodic_sum_r_vel_penalty = float(np.mean(episodic_vel_penalty_totals))
            if episodic_smooth_totals:
                episodic_sum_r_smooth = float(np.mean(episodic_smooth_totals))
        else:
            pass
        
        if self.recent_episode_completions:
            # Group completions by environment
            env_completions = {}
            for completion in self.recent_episode_completions:
                env_idx = completion['env_idx']
                if env_idx not in env_completions:
                    env_completions[env_idx] = []
                env_completions[env_idx].append(completion)
            
            # Use completions from environment 0 if available
            # Only show the most recent episode (last num_agents completions)
            if 0 in env_completions:
                recent_completions = env_completions[0][-self.num_agents:]
                episode_details_to_show = {
                    'env_idx': 0,
                    'drones': [
                        {
                            'drone_id': c['agent_id'],
                            **c['summary']
                        }
                        for c in sorted(recent_completions, key=lambda x: x['agent_id'])
                    ]
                }
            
            # Clear the completions for next logging period
            self.recent_episode_completions.clear()
        
        # Collect episode summaries from all rollout threads
        all_final_distances = []
        all_reached = []
        episode_details = []
        current_positions_log = []  # For logging current positions even if episode not done
        
        for env_idx, info in enumerate(infos):
            if isinstance(info, np.ndarray):
                info = info.tolist()
            if not isinstance(info, (list, tuple)):
                continue
            
            env_summary = {'env_idx': env_idx, 'drones': []}
            env_current = {'env_idx': env_idx, 'drones': []}
            has_summary = False
            
            for agent_id in range(min(len(info), self.num_agents)):
                agent_info = info[agent_id] if isinstance(info[agent_id], dict) else {}
                
                # Get current distances
                dist = agent_info.get('dist_to_target', None)
                if dist is not None:
                    all_final_distances.append(dist)
                
                # Try to extract current position info from agent_info first
                current_pos_info = agent_info.get('current_pos', None)
                target_pos_info = agent_info.get('target_pos', None)
                
                if current_pos_info is not None and target_pos_info is not None:
                    # Got positions directly from info
                    env_current['drones'].append({
                        'drone_id': agent_id,
                        'current_position': list(current_pos_info),
                        'target_position': list(target_pos_info),
                        'distance': dist if dist is not None else np.linalg.norm(np.array(target_pos_info) - np.array(current_pos_info)),
                    })
                else:
                    # Extract current position and target from observation (available every step)
                    # Observation structure: [own_pos(3), own_vel(3), rel_target(3), neighbors...]
                    # Get the last observation for this agent from buffer
                    if hasattr(self, 'buffer') and self.buffer.obs is not None:
                        try:
                            # Buffer shape: [episode_length+1, n_rollout_threads, num_agents, obs_dim]
                            # Access the most recent complete observation (step -2 before warmup overwrites)
                            obs_step = min(self.episode_length - 1, self.buffer.obs.shape[0] - 1)
                            current_obs = self.buffer.obs[obs_step][env_idx][agent_id]
                            current_pos = current_obs[0:3]  # First 3 elements are position
                            rel_target = current_obs[6:9]   # Elements 6-8 are relative target
                            target_pos = current_pos + rel_target
                            
                            env_current['drones'].append({
                                'drone_id': agent_id,
                                'current_position': current_pos.tolist() if hasattr(current_pos, 'tolist') else list(current_pos),
                                'target_position': target_pos.tolist() if hasattr(target_pos, 'tolist') else list(target_pos),
                                'distance': dist if dist is not None else np.linalg.norm(rel_target),
                            })
                        except (IndexError, AttributeError, TypeError) as e:
                            # Fallback: try to construct from agent_info if available
                            pass
                
                # Check for episode summary (only present on done steps)
                summary = agent_info.get('episode_summary', None)
                if summary is not None:
                    has_summary = True
                    all_reached.append(summary['reached_target'])
                    env_summary['drones'].append({
                        'drone_id': agent_id,
                        **summary,
                        'episode_steps': agent_info.get('episode_steps', 0),
                        'episode_reward': agent_info.get('episode_cumulative_reward', 0),
                    })
            
            if has_summary:
                episode_details.append(env_summary)
            if env_current['drones']:
                current_positions_log.append(env_current)
        
        # Compute aggregate metrics
        # Prioritize episode_details_to_show if available (from accumulated recent completions)
        if episode_details_to_show and episode_details_to_show['drones']:
            # Use accumulated completions as source of truth - replace previous data
            all_reached = []
            all_final_distances = []
            for drone in episode_details_to_show['drones']:
                all_reached.append(drone.get('reached_target', False))
                all_final_distances.append(drone.get('final_distance', 0))
        elif episode_details:
            # Use data from current step's infos (already populated from loop above)
            pass
        
        avg_distance = np.mean(all_final_distances) if all_final_distances else None
        
        # Components already computed above from recent_episode_completions
        # Now finalize the aggregates
        num_reached = sum(all_reached) if all_reached else 0
        total_drones = len(all_reached) if all_reached else 0
        
        # Store in history
        self.training_history['episodes'].append(episode)
        self.training_history['timesteps'].append(total_num_steps)
        self.training_history['avg_rewards'].append(avg_reward)
        self.training_history['formation_errors'].append(formation_error)
        self.training_history['episodic_sum_r_form'].append(episodic_sum_r_form)
        self.training_history['episodic_sum_r_nav'].append(episodic_sum_r_nav)
        self.training_history['episodic_sum_r_dist'].append(episodic_sum_r_dist)
        self.training_history['episodic_sum_r_avoid'].append(episodic_sum_r_avoid)
        self.training_history['episodic_sum_r_tilt'].append(episodic_sum_r_tilt)
        self.training_history['episodic_sum_r_reached'].append(episodic_sum_r_reached)
        self.training_history['episodic_sum_r_vel_penalty'].append(episodic_sum_r_vel_penalty)
        self.training_history['episodic_sum_r_smooth'].append(episodic_sum_r_smooth)
        self.training_history['avg_distances'].append(avg_distance)
        self.training_history['per_drone_distances'].append(all_final_distances)
        self.training_history['per_drone_reached'].append(all_reached)
        # Store episode_details_to_show if available, otherwise episode_details
        self.training_history['episode_summaries'].append([episode_details_to_show] if episode_details_to_show else episode_details)
        
        # Print detailed per-drone info
        if avg_distance is not None:
            print(f"  Avg distance to target: {avg_distance:.4f}m")
        if episodic_sum_r_form is not None:
            print(f"  Episodic formation reward (sum r_form over episode): {episodic_sum_r_form:.4f}")
        if episodic_sum_r_nav is not None:
            print(f"  Episodic navigation reward (sum r_nav over episode): {episodic_sum_r_nav:.4f}")
        if episodic_sum_r_dist is not None:
            print(f"  Episodic distance penalty (sum r_dist over episode): {episodic_sum_r_dist:.4f}")
        if episodic_sum_r_avoid is not None:
            print(f"  Episodic collision penalty (sum r_avoid over episode): {episodic_sum_r_avoid:.4f}")
        if episodic_sum_r_tilt is not None:
            print(f"  Episodic tilt penalty (sum r_tilt over episode): {episodic_sum_r_tilt:.4f}")
        if episodic_sum_r_reached is not None:
            print(f"  Episodic reaching bonus (sum r_reached over episode): {episodic_sum_r_reached:.4f}")
        if episodic_sum_r_vel_penalty is not None:
            print(f"  Episodic velocity penalty (sum r_vel_penalty over episode): {episodic_sum_r_vel_penalty:.4f}")
        if episodic_sum_r_smooth is not None:
            print(f"  Episodic smoothness penalty (sum r_smooth over episode): {episodic_sum_r_smooth:.4f}")
        
        if total_drones > 0:
            print(f"  Targets reached: {num_reached}/{total_drones} drones "
                  f"({100*num_reached/total_drones:.1f}%)")
        
        # Always print positions for the first environment every 5th episode
        if episode_details_to_show:
            # Episode completed during this rollout - show initial, final, and target
            env_idx = episode_details_to_show['env_idx']
            print(f"\n  --- Episode Completed in Env {env_idx} ---")
            for drone in episode_details_to_show['drones']:
                d = drone
                init = d['initial_position']
                final = d['final_position']
                target = d['target_position']
                print(f"    Drone {d['drone_id']}:")
                print(f"      Started at:       [{init[0]:+.2f}, {init[1]:+.2f}, {init[2]:+.2f}]")
                print(f"      Ended at:         [{final[0]:+.2f}, {final[1]:+.2f}, {final[2]:+.2f}]")
                print(f"      Should reach:     [{target[0]:+.2f}, {target[1]:+.2f}, {target[2]:+.2f}]")
                status = 'REACHED ✓' if d['reached_target'] else f"Distance remaining: {d['final_distance']:.3f}m"
                print(f"      Status: {status}")
        elif episode_details:
            # Episode completed at the last step - show initial, final, and target
            env_summary = episode_details[0]
            env_idx = env_summary['env_idx']
            print(f"\n  --- Episode Completed in Env {env_idx} ---")
            for drone in env_summary['drones']:
                d = drone
                init = d['initial_position']
                final = d['final_position']
                target = d['target_position']
                print(f"    Drone {d['drone_id']}:")
                print(f"      Started at:       [{init[0]:+.2f}, {init[1]:+.2f}, {init[2]:+.2f}]")
                print(f"      Ended at:         [{final[0]:+.2f}, {final[1]:+.2f}, {final[2]:+.2f}]")
                print(f"      Should reach:     [{target[0]:+.2f}, {target[1]:+.2f}, {target[2]:+.2f}]")
                status = 'REACHED ✓' if d['reached_target'] else f"Distance remaining: {d['final_distance']:.3f}m"
                print(f"      Status: {status}")
        elif current_positions_log:
            # Episode in progress - show current position and target
            env_current = current_positions_log[0]
            env_idx = env_current['env_idx']
            print(f"\n  --- Current Positions in Env {env_idx} (Episode In Progress) ---")
            for drone in env_current['drones']:
                curr = drone['current_position']
                target = drone['target_position']
                dist = drone['distance']
                print(f"    Drone {drone['drone_id']}:")
                print(f"      Current position: [{curr[0]:+.2f}, {curr[1]:+.2f}, {curr[2]:+.2f}]")
                print(f"      Should reach:     [{target[0]:+.2f}, {target[1]:+.2f}, {target[2]:+.2f}]")
                print(f"      Distance to target: {dist:.3f}m")
        else:
            print(f"\n  --- No position data available for logging ---")
    
    def _save_training_history(self):
        """Save training history to JSON file."""
        save_path = os.path.join(self.save_dir if hasattr(self, 'save_dir') else '.', 'training_history.json')
        
        # Convert numpy types to Python native types for JSON
        serializable_history = {}
        for key, values in self.training_history.items():
            if key == 'episode_summaries':
                serializable_history[key] = values  # Already serializable
            else:
                serializable_history[key] = []
                for v in values:
                    if v is None:
                        serializable_history[key].append(None)
                    elif isinstance(v, (list, tuple)):
                        serializable_history[key].append([float(x) if isinstance(x, (int, float, np.floating)) else bool(x) for x in v])
                    elif isinstance(v, (np.floating, np.integer)):
                        serializable_history[key].append(float(v))
                    else:
                        serializable_history[key].append(v)
        
        try:
            with open(save_path, 'w') as f:
                json.dump(serializable_history, f, indent=2, default=str)
            print(f"\n[INFO] Training history saved to: {save_path}")
        except Exception as e:
            print(f"[WARNING] Failed to save training history: {e}")
    
    def _generate_training_plots(self):
        """Generate comprehensive training plots at the end of training."""
        plot_dir = os.path.join(self.save_dir if hasattr(self, 'save_dir') else '.', 'plots')
        os.makedirs(plot_dir, exist_ok=True)
        individual_plot_dir = os.path.join(plot_dir, 'individual_plots')
        os.makedirs(individual_plot_dir, exist_ok=True)
        
        history = self.training_history
        if len(history['episodes']) < 2:
            print("[INFO] Not enough data points for plotting.")
            return
        
        episodes = history['episodes']
        timesteps = history['timesteps']
        use_normalized_plot_components = bool(getattr(self.all_args, 'normalize_plot_components', False))
        window = max(1, len(history['avg_rewards']) // 20)

        # Scale factors for optional visualization normalization of episodic reward components.
        reward_plot_scales = {
            'episodic_sum_r_form': 50.0,
            'episodic_sum_r_nav': 1.25,
            'episodic_sum_r_dist': 400.0,
            'episodic_sum_r_avoid': 0.5,
            'episodic_sum_r_tilt': 1.2,
            'episodic_sum_r_vel_penalty': 50.0,
            'episodic_sum_r_reached': 140.0,
            'episodic_sum_r_smooth': 500.0,
        }

        def _save_individual_line_plot(file_name, x_vals, y_vals, color, xlabel, ylabel, title,
                                       y_limits=None, draw_zero_line=False):
            """Save one line-plot panel into plots/individual_plots."""
            fig_i, ax_i = plt.subplots(1, 1, figsize=(8, 5))
            y_arr = np.array(y_vals, dtype=float)
            x_arr = np.array(x_vals[:len(y_arr)])
            mask = ~np.isnan(y_arr)

            if mask.any():
                ax_i.plot(x_arr[mask], y_arr[mask], color=color, alpha=0.3, linewidth=0.8)
                if mask.sum() >= window:
                    smoothed = np.convolve(y_arr[mask], np.ones(window)/window, mode='valid')
                    ax_i.plot(x_arr[mask][window-1:], smoothed, color=color, linewidth=2)
            else:
                ax_i.text(0.5, 0.5, 'No data', transform=ax_i.transAxes,
                          ha='center', va='center', color='gray')

            if draw_zero_line:
                ax_i.axhline(y=0, color='k', linestyle='--', alpha=0.3)

            ax_i.set_xlabel(xlabel)
            ax_i.set_ylabel(ylabel)
            ax_i.set_title(title)
            ax_i.grid(True, alpha=0.3)
            if y_limits is not None:
                ax_i.set_ylim(*y_limits)

            plt.tight_layout()
            save_path = os.path.join(individual_plot_dir, file_name)
            fig_i.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close(fig_i)

        
        # ---- Figure 1: Rewards over time ----
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        fig.suptitle(f'Training Summary: {self.experiment_name}', fontsize=16, fontweight='bold')
        
        # 1a: Average episodic reward
        ax = axes[0]
        rewards = history['avg_rewards']
        ax.plot(timesteps, rewards, 'b-', alpha=0.3, linewidth=0.8)
        # Smoothed curve (moving average)
        if len(rewards) >= window:
            smoothed = np.convolve(rewards, np.ones(window)/window, mode='valid')
            ax.plot(timesteps[window-1:], smoothed, 'b-', linewidth=2)
        ax.set_xlabel('Training Steps')
        ax.set_ylabel('Average Episodic Reward')
        ax.set_title('Episodic Reward')
        ax.grid(True, alpha=0.3)
        
        # 1b: Formation Error
        ax = axes[1]
        form_errors = [fe if fe is not None else np.nan for fe in history['formation_errors']]
        ax.plot(timesteps, form_errors, 'r-', alpha=0.3, linewidth=0.8)
        if len(form_errors) >= window:
            valid_fe = np.array(form_errors, dtype=float)
            # Handle NaNs for smoothing
            mask = ~np.isnan(valid_fe)
            if mask.sum() >= window:
                smoothed_fe = np.convolve(valid_fe[mask], np.ones(window)/window, mode='valid')
                ax.plot(np.array(timesteps)[mask][window-1:], smoothed_fe, 'r-', linewidth=2, label=f'Smoothed')
        ax.set_xlabel('Training Steps')
        ax.set_ylabel('Error')
        ax.set_title('Formation Error')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        fig_path = os.path.join(plot_dir, 'training_summary.png')
        fig.savefig(fig_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"[INFO] Training summary plot saved to: {fig_path}")

        _save_individual_line_plot(
            file_name='avg_episodic_reward.png',
            x_vals=timesteps,
            y_vals=rewards,
            color='b',
            xlabel='Training Steps',
            ylabel='Average Episodic Reward',
            title='Episodic Reward',
        )
        _save_individual_line_plot(
            file_name='formation_error.png',
            x_vals=timesteps,
            y_vals=form_errors,
            color='r',
            xlabel='Training Steps',
            ylabel='Error',
            title='Formation Error',
        )
        
        # ---- Figure 2: Per-drone distance box plots at intervals ----
        fig2, ax2 = plt.subplots(1, 1, figsize=(14, 6))
        # Sample 10 evenly spaced episodes for box plot
        n_samples = min(10, len(history['per_drone_distances']))
        sample_indices = np.linspace(0, len(history['per_drone_distances'])-1, n_samples, dtype=int)
        box_data = []
        box_labels = []
        for idx in sample_indices:
            dists = history['per_drone_distances'][idx]
            if dists and len(dists) > 0:
                box_data.append(dists)
                ep = history['episodes'][idx]
                box_labels.append(f'Ep {ep}')
        
        if box_data:
            # matplotlib >= 3.9 renamed `labels` to `tick_labels` (the old name was removed in 3.11)
            try:
                bp = ax2.boxplot(box_data, tick_labels=box_labels, patch_artist=True)
            except TypeError:
                bp = ax2.boxplot(box_data, labels=box_labels, patch_artist=True)
            for patch in bp['boxes']:
                patch.set_facecolor('lightblue')
            ax2.axhline(y=0.05, color='r', linestyle='--', alpha=0.5, label='Goal threshold (5cm)')
            ax2.set_xlabel('Training Episode')
            ax2.set_ylabel('Distance to Target (m)')
            ax2.set_title('Per-Drone Distance Distribution Over Training')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        fig2_path = os.path.join(plot_dir, 'distance_distribution.png')
        fig2.savefig(fig2_path, dpi=150, bbox_inches='tight')
        plt.close(fig2)
        print(f"[INFO] Distance distribution plot saved to: {fig2_path}")

        # ---- Figure 3: Reward components breakdown (from logged environment infos) ----
        # Extract reward components from episode summaries
        fig3, axes3 = plt.subplots(1, 2, figsize=(12, 5))
        fig3.suptitle('Reward Components Over Training', fontsize=14, fontweight='bold')
        
        # Episodic total formation reward component over time
        form_rewards = [r if r is not None else np.nan for r in history.get('episodic_sum_r_form', [])]
        form_ts = np.array(timesteps[:len(form_rewards)])
        form_vals_all = np.array(form_rewards, dtype=float)
        if use_normalized_plot_components:
            form_vals_all = form_vals_all / reward_plot_scales['episodic_sum_r_form']
            form_vals_all = np.clip(form_vals_all, -1.0, 1.0)
        form_mask = ~np.isnan(form_vals_all)
        if form_mask.any():
            ax = axes3[0]
            form_vals = form_vals_all[form_mask]
            form_t = form_ts[form_mask]
            ax.plot(form_t, form_vals, 'b-', alpha=0.3, linewidth=0.8)
            if len(form_vals) >= window:
                form_smoothed = np.convolve(form_vals, np.ones(window)/window, mode='valid')
                ax.plot(form_t[window-1:], form_smoothed, 'b-', linewidth=2)
            ax.set_xlabel('Training Steps')
            ax.set_ylabel('Reward')
            ax.set_title('Formation Reward')
            ax.grid(True, alpha=0.3)
            if use_normalized_plot_components:
                ax.set_ylim(-1.05, 0.05)
        
        # Episodic total navigation reward component over time
        ax = axes3[1]
        nav_rewards = [r if r is not None else np.nan for r in history.get('episodic_sum_r_nav', [])]
        nav_ts = np.array(timesteps[:len(nav_rewards)])
        nav_vals_all = np.array(nav_rewards, dtype=float)
        if use_normalized_plot_components:
            nav_vals_all = nav_vals_all / reward_plot_scales['episodic_sum_r_nav']
            nav_vals_all = np.clip(nav_vals_all, -1.0, 1.0)
        valid_mask = ~np.isnan(nav_vals_all)
        if valid_mask.any():
            nav_vals = nav_vals_all[valid_mask]
            nav_t = nav_ts[valid_mask]
            ax.plot(nav_t, nav_vals, 'g-', alpha=0.3, linewidth=0.8)
            if len(nav_vals) >= window:
                nav_smoothed = np.convolve(nav_vals, np.ones(window)/window, mode='valid')
                ax.plot(nav_t[window-1:], nav_smoothed, 'g-', linewidth=2)
            ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)
            ax.set_xlabel('Training Steps')
            ax.set_ylabel('Reward')
            ax.set_title('Navigation Reward')
            ax.grid(True, alpha=0.3)
            if use_normalized_plot_components:
                ax.set_ylim(-0.2, 1.05)
        
        plt.tight_layout()
        fig3_path = os.path.join(plot_dir, 'reward_components.png')
        fig3.savefig(fig3_path, dpi=150, bbox_inches='tight')
        plt.close(fig3)
        print(f"[INFO] Reward components plot saved to: {fig3_path}")

        form_ylim = (-1.05, 0.05) if use_normalized_plot_components else None
        nav_ylim = (-0.2, 1.05) if use_normalized_plot_components else None
        _save_individual_line_plot(
            file_name='formation_reward.png',
            x_vals=form_ts,
            y_vals=form_vals_all,
            color='b',
            xlabel='Training Steps',
            ylabel='Reward',
            title='Formation Reward',
            y_limits=form_ylim,
            draw_zero_line=False,
        )
        _save_individual_line_plot(
            file_name='navigation_reward.png',
            x_vals=nav_ts,
            y_vals=nav_vals_all,
            color='g',
            xlabel='Training Steps',
            ylabel='Reward',
            title='Navigation Reward',
            y_limits=nav_ylim,
            draw_zero_line=True,
        )

        # ---- Figure 4: Penalty/bonus components (r_dist, r_avoid, r_tilt, r_reached, r_vel_penalty) ----
        fig4, axes4 = plt.subplots(3, 2, figsize=(14, 15))
        fig4.suptitle('Penalty & Bonus Components Over Training', fontsize=14, fontweight='bold')

        component_specs = [
            ('episodic_sum_r_dist',        axes4[0, 0], 'Distance Penalty',     'purple', 'distance_penalty.png'),
            ('episodic_sum_r_avoid',       axes4[0, 1], 'Collision Penalty',    'red',    'collision_penalty.png'),
            ('episodic_sum_r_tilt',        axes4[1, 0], 'Tilt Penalty',         'orange', 'tilt_penalty.png'),
            ('episodic_sum_r_reached',     axes4[1, 1], 'Reaching Bonus',       'green',  'reaching_bonus.png'),
            ('episodic_sum_r_vel_penalty', axes4[2, 0], 'Velocity Penalty',     'brown',  'velocity_penalty.png'),
            ('episodic_sum_r_smooth',      axes4[2, 1], 'Smoothness Penalty',   'teal',   'smoothness_penalty.png'),
        ]

        for key, ax, title, color, individual_name in component_specs:
            vals_raw = [v if v is not None else np.nan for v in history.get(key, [])]
            ts_arr = np.array(timesteps[:len(vals_raw)])
            vals_arr = np.array(vals_raw, dtype=float)
            if use_normalized_plot_components:
                vals_arr = vals_arr / reward_plot_scales[key]
                vals_arr = np.clip(vals_arr, -1.0, 1.0)
            mask = ~np.isnan(vals_arr)
            ax.set_xlabel('Training Steps')
            ax.set_ylabel(f'Reward')
            ax.set_title(title)
            ax.grid(True, alpha=0.3)
            if use_normalized_plot_components:
                if key == 'episodic_sum_r_reached':
                    ax.set_ylim(-0.05, 1.05)
                else:
                    ax.set_ylim(-1.05, 0.05)
            if mask.any():
                ax.plot(ts_arr[mask], vals_arr[mask], color=color, alpha=0.3, linewidth=0.8)
                if mask.sum() >= window:
                    smoothed = np.convolve(vals_arr[mask], np.ones(window)/window, mode='valid')
                    ax.plot(ts_arr[mask][window-1:], smoothed, color=color, linewidth=2)
                ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)
            else:
                ax.text(0.5, 0.5, 'No data', transform=ax.transAxes,
                        ha='center', va='center', color='gray')

            if use_normalized_plot_components:
                component_ylim = (-0.05, 1.05) if key == 'episodic_sum_r_reached' else (-1.05, 0.05)
            else:
                component_ylim = None
            _save_individual_line_plot(
                file_name=individual_name,
                x_vals=ts_arr,
                y_vals=vals_arr,
                color=color,
                xlabel='Training Steps',
                ylabel='Reward',
                title=title,
                y_limits=component_ylim,
                draw_zero_line=True,
            )

        plt.tight_layout()
        fig4_path = os.path.join(plot_dir, 'penalty_bonus_components.png')
        fig4.savefig(fig4_path, dpi=150, bbox_inches='tight')
        plt.close(fig4)
        print(f"[INFO] Penalty/bonus components plot saved to: {fig4_path}")
        print(f"[INFO] Individual plots saved to: {individual_plot_dir}")

    def _save_last20_metrics(self):
        """
        Compute and save FAR/FS/NAR/NS over the last 20% of logged episodic points.

        Metrics:
        - FAR: mean of r_form over window
        - FS:  1 / (1 + std(r_form))
        - NAR: mean of r_nav over window
        - NS:  1 / (1 + std(r_nav))
        """
        history = self.training_history
        timesteps = history.get('timesteps', [])
        r_form_series = np.array([v if v is not None else np.nan for v in history.get('episodic_sum_r_form', [])], dtype=float)
        r_nav_series = np.array([v if v is not None else np.nan for v in history.get('episodic_sum_r_nav', [])], dtype=float)

        # Require enough logged points to form a meaningful tail window.
        n_points = min(len(timesteps), len(r_form_series), len(r_nav_series))
        if n_points == 0:
            print("[INFO] No reward-component history available for FAR/FS/NAR/NS metrics.")
            return

        start_idx = int(np.floor(0.8 * n_points))
        if start_idx >= n_points:
            start_idx = max(0, n_points - 1)

        tail_timesteps = timesteps[start_idx:n_points]
        tail_r_form = r_form_series[start_idx:n_points]
        tail_r_nav = r_nav_series[start_idx:n_points]

        valid_form = tail_r_form[~np.isnan(tail_r_form)]
        valid_nav = tail_r_nav[~np.isnan(tail_r_nav)]

        far = float(np.mean(valid_form)) if valid_form.size > 0 else None
        fs = float(1.0 / (1.0 + np.std(valid_form))) if valid_form.size > 0 else None
        nar = float(np.mean(valid_nav)) if valid_nav.size > 0 else None
        ns = float(1.0 / (1.0 + np.std(valid_nav))) if valid_nav.size > 0 else None

        metrics_payload = {
            'window': {
                'definition': 'last_20_percent_of_logged_episodic_points',
                'source_series': ['episodic_sum_r_form', 'episodic_sum_r_nav'],
                'start_index': int(start_idx),
                'end_index_exclusive': int(n_points),
                'num_points_in_window': int(max(0, n_points - start_idx)),
                'first_timestep_in_window': int(tail_timesteps[0]) if tail_timesteps else None,
                'last_timestep_in_window': int(tail_timesteps[-1]) if tail_timesteps else None,
            },
            'metrics': {
                'FAR': far,
                'FS': fs,
                'NAR': nar,
                'NS': ns,
            },
            'valid_counts': {
                'r_form_points': int(valid_form.size),
                'r_nav_points': int(valid_nav.size),
            }
        }

        output_dir = os.path.join(self.save_dir if hasattr(self, 'save_dir') else '.', 'plots')
        os.makedirs(output_dir, exist_ok=True)
        metrics_path = os.path.join(output_dir, 'last20_metrics.json')

        try:
            with open(metrics_path, 'w') as f:
                json.dump(metrics_payload, f, indent=2)
            print(f"[INFO] Last-20% FAR/FS/NAR/NS metrics saved to: {metrics_path}")
        except Exception as e:
            print(f"[WARNING] Failed to save last-20% FAR/FS/NAR/NS metrics: {e}")
        
        print(f"\n[INFO] All training plots and metrics saved to: {output_dir}/")


