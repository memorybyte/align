"""
PyBullet Drone Environment Wrapper for MAPPO.

This module wraps gym-pybullet-drones MultiHoverAviary to provide a MAPPO-compatible
interface with proper observation/share_obs structure.

- reset() -> (per_agent_obs, share_obs)
- step(actions) -> (per_agent_obs, share_obs, rewards, dones, infos)
- share_obs: concatenation of all agents' positions and velocities, tiled per agent
"""

import numpy as np
from gym import spaces
from typing import List, Tuple, Dict, Any, Optional

# Import from local gym_pybullet_drones copy
from onpolicy.envs.gym_pybullet_drones.envs.MultiHoverAviary import MultiHoverAviary
from onpolicy.envs.gym_pybullet_drones.utils.enums import DroneModel, Physics, ActionType, ObservationType

# Formation utilities for Procrustes/SVD-based reward
from onpolicy.utils.formation import compute_formation_error


class PyBulletDroneWrapper:
    def __init__(
        self,
        num_drones: int = 3,
        gui: bool = False,
        record: bool = False,
        obs_type: ObservationType = ObservationType.KIN,
        act_type: ActionType = ActionType.VEL,  # VEL uses internal PID, accepts v_des
        pyb_freq: int = 240,
        ctrl_freq: int = 30,
        episode_len_sec: float = 8.0,
        collision_dist: float = 0.1,
        w_form: float = 0,
        w_nav: float = 10.0,
        w_dist: float = 0,
        w_avoid: float = 2.0,
        w_tilt: float = 5.0,
        w_vel_penalty: float = 0.0,
        w_smooth: float = 0.5,
        tilt_soft_threshold: float = 0.35,
        collision_C: float = 1.0,
        formation_spacing: float = 1.0,
        perturbation_std: float = 0.05,
        formation_type: str = 'polygon',
        max_neighbors: int = 3,
        neighbour_radius: float = 0.0,
        min_dynamic_neighbours: int = 1,
        max_dynamic_neighbours: int = 8,
        target_distance_range: Tuple[float, float] = (0.5, 2.0),
        success_dist: float = 0.2,
    ):
        """
        Initialize the PyBullet drone wrapper.
        
        Args:
            num_drones: Number of drones in the environment
            gui: Whether to use PyBullet GUI
            record: Whether to record video
            obs_type: Observation type (KIN for kinematic)
            act_type: Action type (VEL for velocity-based with internal PID)
            pyb_freq: PyBullet simulation frequency
            ctrl_freq: Control frequency (48Hz to match simulation)
            episode_len_sec: Episode length in seconds
            collision_dist: Distance threshold for collision detection
            w_form: Weight for formation reward (downweighted for navigation tasks)
            w_nav: Weight for navigation reward — primary signal
            w_dist: Weight for persistent distance penalty
            w_avoid: Weight for collision avoidance reward
            w_tilt: Weight for soft tilt penalty (applied when |roll|/|pitch| exceeds threshold)
            tilt_soft_threshold: Soft threshold (radians) before tilt penalty activates
            collision_C: Collision penalty constant C ~1
            formation_spacing: Inter-drone distance in formation template (meters)
            formation_type: one of polygon/line/plane/cube/sphere/pyramid/dynamic
                            (dynamic samples cube/sphere/pyramid/plane each episode)
            perturbation_std: Std-dev of Gaussian noise on initial positions (meters)
            max_neighbors: Maximum nearest neighbors per drone used in KIN observation
                           (used when dynamic neighbors is disabled, i.e. neighbour_radius=0)
            neighbour_radius: Radius (meters) for dynamic neighbor selection.
                              0 disables dynamic neighbors and falls back to fixed k-nearest.
            min_dynamic_neighbours: Minimum neighbor slots always filled (closest are included
                                    even if outside radius)
            max_dynamic_neighbours: Maximum neighbor slots (defines obs_dim when dynamic
                                    neighbors is enabled). Remaining slots are zero-padded.
            target_distance_range: (min, max) distance (m) from initial to target formation center
            success_dist: Distance (m) to target counted as reached. Used for the reaching
                          bonus, the success termination and the logged `reached_target`.
        """
        self.num_drones = num_drones
        self.max_neighbors = max(0, int(max_neighbors))
        self.neighbour_radius = float(neighbour_radius)
        self.min_dynamic_neighbours = max(0, int(min_dynamic_neighbours))
        self.max_dynamic_neighbours = max(1, int(max_dynamic_neighbours))
        self.use_dynamic_neighbours = self.neighbour_radius > 0.0
        self.collision_dist = collision_dist
        self.w_form = w_form
        self.w_nav = w_nav
        self.w_dist = w_dist
        self.w_avoid = w_avoid
        self.w_tilt = w_tilt
        self.w_vel_penalty = w_vel_penalty
        self.w_smooth = w_smooth
        self.tilt_soft_threshold = tilt_soft_threshold
        self.collision_C = collision_C
        self.success_dist = float(success_dist)
        
        # Create the underlying environment with formation-based positioning
        # MultiHoverAviary handles formation template, random center, random yaw,
        # and perturbations internally (see FORMATION_CONTROL_TRAINING_STRATEGY.md)
        # ActionType.VEL accepts velocity commands and uses internal PID to compute RPMs
        self._env = MultiHoverAviary(
            drone_model=DroneModel.CF2X,
            num_drones=num_drones,
            initial_xyzs=None,  # Let MultiHoverAviary generate from formation template
            physics=Physics.PYB,
            pyb_freq=pyb_freq,
            ctrl_freq=ctrl_freq,
            gui=gui,
            record=record,
            obs=obs_type,
            act=act_type,
            formation_spacing=formation_spacing,
            perturbation_std=perturbation_std,
            formation_type=formation_type,
            max_neighbors=self.max_neighbors,
            neighbour_radius=self.neighbour_radius,
            min_dynamic_neighbours=self.min_dynamic_neighbours,
            max_dynamic_neighbours=self.max_dynamic_neighbours,
            target_distance_range=tuple(target_distance_range),
        )
        
        # Store for computing deltas
        self._prev_distances = None
        self._prev_actions = None  # For action smoothness reward
        
        # Visual markers for rendering
        self._visual_marker_ids = []  # Store IDs of visual markers for cleanup
        
        # Set up spaces for MAPPO
        self._setup_spaces()
    
    def _setup_spaces(self):
        """Set up observation, share_observation, and action spaces for MAPPO."""
        # Get base observation space from env
        base_obs_space = self._env.observation_space
        
        # Per-agent observation dimension
        # KIN obs: own_state(6) + rel_target(3) + neighbors(6*k)
        # When dynamic neighbors is enabled, k = max_dynamic_neighbours (zero-padded)
        # When disabled, k = min(max_neighbors, N-1) (fixed k-nearest)
        if hasattr(base_obs_space, 'shape'):
            self.obs_dim = base_obs_space.shape[1] if len(base_obs_space.shape) > 1 else base_obs_space.shape[0]
        else:
            if self.use_dynamic_neighbours:
                k = min(self.max_dynamic_neighbours, self.num_drones - 1)
            else:
                k = min(self.max_neighbors, self.num_drones - 1)
            self.obs_dim = 6 + 3 + 6 * k
        
        # Share observation: positions (3) + velocities (3) + target_pos (3) for all agents
        # Gives the centralized critic full information about where drones are and should go
        self.share_obs_dim_per_agent = 9  # pos(3) + vel(3) + target(3)
        self.share_obs_dim = self.num_drones * self.share_obs_dim_per_agent
        
        # Create spaces (lists for MAPPO compatibility)
        self.observation_space = [
            spaces.Box(low=-np.inf, high=np.inf, shape=(self.obs_dim,), dtype=np.float32)
            for _ in range(self.num_drones)
        ]
        
        self.share_observation_space = [
            spaces.Box(low=-np.inf, high=np.inf, shape=(self.share_obs_dim,), dtype=np.float32)
            for _ in range(self.num_drones)
        ]
        
        # Action space: 4D velocity command [vx, vy, vz, throttle_scale]
        self.action_space = [
            spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)
            for _ in range(self.num_drones)
        ]
        
        # For env_wrappers compatibility
        self.n = self.num_drones
    
    def seed(self, seed: int = None):
        """Set random seed."""
        if seed is not None:
            np.random.seed(seed)
    
    def _get_drone_states(self) -> np.ndarray:
        """
        Get states of all drones.
        
        Returns:
            Array of shape (num_drones, state_dim) containing drone states
        """
        states = np.array([
            self._env._getDroneStateVector(i) for i in range(self.num_drones)
        ])
        return states
    
    def _extract_positions_velocities(self, states: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract positions and velocities from drone states.
        
        Args:
            states: Drone states array (num_drones, state_dim)
            
        Returns:
            positions: (num_drones, 3)
            velocities: (num_drones, 3)
        """
        positions = states[:, 0:3]
        velocities = states[:, 10:13]
        return positions, velocities
    
    def _compute_share_obs(self, states: np.ndarray) -> np.ndarray:
        """
        Compute shared observation for centralized critic.
        
        - share_obs: concatenated positions and velocities of all agents
        - Tiled per agent (each agent sees the same global state)
        
        Args:
            states: Drone states array (num_drones, state_dim)
            
        Returns:
            share_obs: Array of shape (num_drones, share_obs_dim)
        """
        positions, velocities = self._extract_positions_velocities(states)
        
        # Get target positions from the underlying env
        target_pos = self._env.TARGET_POS  # (num_drones, 3)
        
        # Concatenate pos + vel + target for each agent, then flatten
        agent_share_info = np.concatenate([positions, velocities, target_pos], axis=1)  # (num_drones, 9)
        share_obs_flat = agent_share_info.flatten()  # (num_drones * 9,)
        
        # Tile for each agent (each agent gets the same share_obs)
        share_obs = np.tile(share_obs_flat, (self.num_drones, 1))  # (num_drones, share_obs_dim)
        
        return share_obs.astype(np.float32)
    
    def _compute_per_agent_obs(self, raw_obs: np.ndarray) -> List[np.ndarray]:
        """
        Convert raw observations to per-agent list format.
        
        Args:
            raw_obs: Raw observations from env (num_drones, obs_dim)
            
        Returns:
            List of observations, one per agent
        """
        return [raw_obs[i].astype(np.float32) for i in range(self.num_drones)]
    
    def _compute_reward(
        self, 
        states: np.ndarray,
        actions: np.ndarray = None,
    ) -> Tuple[List[float], List[Dict[str, float]]]:
        """
        Compute individual rewards for each drone.
        
        Total reward per agent i:
            r[i] = w_form * r_form + w_nav * r_nav[i] + w_dist * r_dist[i] + w_avoid * r_avoid[i] + r_reached[i] + r_vel_penalty[i] + w_smooth * r_smooth[i]
        
        Components:
        - Formation error via Procrustes/SVD alignment (shared across all agents):
            r_form = -E / (G + eps)
        - Navigation reward (per-agent delta distance improvement):
            r_nav[i] = d_prev[i] - d_curr[i]
        - Distance penalty (per-agent distance to target):
            r_dist[i] = -dist[i]
        - Collision avoidance penalty (per-agent):
            r_avoid[i] = -1 if agent i is in collision, else 0
        - Reaching bonus (per-agent): +1.0 if agent i within success_dist of its target
        - Velocity penalty (per-agent): Penalize high velocity when close to target
        - Tilt penalty (per-agent): Soft penalty when roll/pitch exceeds threshold
        
        """
        positions, velocities = self._extract_positions_velocities(states)
        target_pos = self._env.TARGET_POS  # (num_drones, 3)
        
        # --- 1. Formation reward (Procrustes/SVD) — shared across all agents ---
        E, G = compute_formation_error(positions, target_pos)
        r_form = -E / (G + 1e-8)
        
        # --- 2. Navigation reward (per-agent) — PRIMARY signal ---
        # r_nav[i] = d_prev[i] - d_curr[i] for each agent i
        current_distances = np.linalg.norm(target_pos - positions, axis=1)
        r_nav = self._prev_distances - current_distances
        self._prev_distances = current_distances.copy()
        
        # --- 3. Distance penalty (per-agent) — persistent signal ---
        # r_dist[i] = -current_distances[i] for each agent i
        r_dist = -current_distances
        
        # --- 4. Collision avoidance penalty (per-agent) ---
        # r_avoid[i] = -1 if agent i is involved in a collision, else 0
        pos_diff = positions[:, None, :] - positions[None, :, :]
        dist_sq = np.sum(pos_diff * pos_diff, axis=2)
        collision_pairs = np.triu(dist_sq < (self.collision_dist ** 2), k=1)
        pair_indices = np.argwhere(collision_pairs)
        num_collisions = int(pair_indices.shape[0])

        involved = np.zeros(self.num_drones, dtype=bool)
        if num_collisions > 0:
            involved[pair_indices[:, 0]] = True
            involved[pair_indices[:, 1]] = True
        r_avoid = np.where(involved, -1.0, 0.0)
        
        # --- 5. Reaching bonus (per-agent) ---
        # +1.0 if agent i is within success_dist of its target
        r_reached = np.where(current_distances < self.success_dist, 1.0, 0.0)
        
        # --- 6. Velocity penalty when close to target (per-agent) ---
        # Smooth ramp: penalty grows linearly as drone gets closer to target.
        # At distance=0 → full penalty weight; at distance=threshold → zero penalty.
        # This avoids a discontinuous reward at the boundary.
        r_vel_penalty = np.zeros(self.num_drones)
        velocity_threshold = 0.5  # Distance at which penalty starts ramping in
        velocity_magnitude = np.linalg.norm(velocities, axis=1)
        proximity_factor = np.clip(1.0 - (current_distances / velocity_threshold), 0.0, 1.0)
        r_vel_penalty = -proximity_factor * velocity_magnitude

        # --- 7. Soft tilt penalty (per-agent) ---
        # Uses max(|roll|, |pitch|) so either axis can trigger stabilization pressure.
        roll = states[:, 7]
        pitch = states[:, 8]
        tilt_mag = np.maximum(np.abs(roll), np.abs(pitch))
        tilt_excess = np.clip(tilt_mag - self.tilt_soft_threshold, 0.0, None)
        r_tilt = -(tilt_excess ** 2)
        
        # --- 8. Action smoothness penalty (per-agent) ---
        # Penalizes rapid changes in action between consecutive steps.
        # r_smooth[i] = -||a_t[i] - a_{t-1}[i]||^2
        if actions is not None and self._prev_actions is not None:
            action_diff = actions - self._prev_actions  # (num_drones, 4)
            r_smooth = -np.sum(action_diff ** 2, axis=1)  # (num_drones,)
        else:
            r_smooth = np.zeros(self.num_drones)
        # Update previous actions
        if actions is not None:
            self._prev_actions = actions.copy()
        
        # --- Total reward per agent ---
        rewards = []
        reward_infos = []
        for i in range(self.num_drones):
            total_reward_i = ( self.w_form * r_form 
                            + self.w_nav * r_nav[i] 
                            + self.w_dist * r_dist[i] 
                            + self.w_avoid * r_avoid[i] 
                            + self.w_tilt * r_tilt[i]
                            + r_reached[i]
                            + self.w_vel_penalty * r_vel_penalty[i]
                            + self.w_smooth * r_smooth[i]
                            )
            rewards.append(total_reward_i)
            
            # Create per-agent reward info
            reward_info_i = {
                'r_form': r_form,
                'r_nav': r_nav[i],
                'r_dist': r_dist[i],
                'r_avoid': r_avoid[i],
                'r_tilt': r_tilt[i],
                'r_reached': r_reached[i],
                'r_vel_penalty': r_vel_penalty[i],
                'r_smooth': r_smooth[i],
                'formation_error': E,
                'normalization_G': G,
                'num_collisions': num_collisions,
                'total_reward': total_reward_i,
                'dist_to_target': current_distances[i],
                'current_pos': positions[i].tolist() if hasattr(positions[i], 'tolist') else list(positions[i]),
                'target_pos': target_pos[i].tolist() if hasattr(target_pos[i], 'tolist') else list(target_pos[i]),
            }
            reward_infos.append(reward_info_i)
        
        return rewards, reward_infos
    
    def reset(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Reset the environment.
        
        Returns:
            obs_n: Per-agent observations as numpy array (num_drones, obs_dim)
            share_obs: Shared observation for centralized critic (num_drones, share_obs_dim)
            available_actions: Available actions mask (all available for continuous control)
        """
        raw_obs, info = self._env.reset()
        
        # Get states for share_obs computation
        states = self._get_drone_states()
        positions, _ = self._extract_positions_velocities(states)
        
        # Store initial positions and targets for logging
        self._episode_initial_positions = positions.copy()
        self._episode_target_positions = self._env.TARGET_POS.copy()
        self._episode_step_count = 0
        self._episode_cumulative_reward = np.zeros(self.num_drones, dtype=np.float32)
        self._episode_cumulative_r_form = 0.0
        self._episode_cumulative_r_nav = np.zeros(self.num_drones, dtype=np.float32)
        self._episode_cumulative_r_dist = np.zeros(self.num_drones, dtype=np.float32)
        self._episode_cumulative_r_avoid = np.zeros(self.num_drones, dtype=np.float32)
        self._episode_cumulative_r_tilt = np.zeros(self.num_drones, dtype=np.float32)
        self._episode_cumulative_r_reached = np.zeros(self.num_drones, dtype=np.float32)
        self._episode_cumulative_r_vel_penalty = np.zeros(self.num_drones, dtype=np.float32)
        self._episode_cumulative_r_smooth = np.zeros(self.num_drones, dtype=np.float32)
        self._episode_terminated_early = False
        
        # Initialize previous actions to zeros (no action change on first step)
        self._prev_actions = np.zeros((self.num_drones, 4), dtype=np.float32)
        
        # Initialize per-drone distances to TARGET_POS for shaped reward
        self._prev_distances = np.array([
            np.linalg.norm(self._env.TARGET_POS[i] - positions[i])
            for i in range(self.num_drones)
        ])
        self._episode_initial_distances = self._prev_distances.copy()
        
        # Compute observations - return as numpy arrays instead of lists
        obs_n = raw_obs.astype(np.float32)  # Already (num_drones, obs_dim)
        share_obs = self._compute_share_obs(states)  # (num_drones, share_obs_dim)
        
        # All actions available for continuous control
        available_actions = np.ones((self.num_drones, 4), dtype=np.float32)
        
        return obs_n, share_obs, available_actions
    
    def step(
        self, 
        actions: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, List[List[float]], List[bool], List[Dict], np.ndarray]:
        """
        Take a step in the environment.
        
        
        Args:
            actions: Actions for all agents (num_drones, action_dim)
                    For VEL action type: [vx, vy, vz, throttle_scale] in [-1, 1]
        
        Returns:
            obs_n: Per-agent observations as numpy array (num_drones, obs_dim)
            share_obs: Shared observation (num_drones, share_obs_dim)
            rewards_n: List of [reward] per agent
            dones_n: List of done flags per agent
            infos_n: List of info dicts per agent
            available_actions: Available actions mask (all available for continuous control)
        """
        # Ensure actions are numpy array with correct shape
        if isinstance(actions, list):
            actions = np.array(actions)
        
        if len(actions.shape) == 1:
            actions = actions.reshape(self.num_drones, -1)
        
        # Take step in environment
        # MultiHoverAviary with ActionType.VEL handles PID internally
        raw_obs, base_reward, terminated, truncated, info = self._env.step(actions)
        # if(truncated):
        #     print("Step:", self._episode_step_count + 1, "Episode truncated.")
        
        # Get states for reward computation and share_obs
        states = self._get_drone_states()
        positions, velocities = self._extract_positions_velocities(states)
        target_pos = self._env.TARGET_POS
        
        # Fix 3: Check if all drones reached target with low velocity (early success termination)
        all_reached = True
        success_distance_threshold = self.success_dist
        success_velocity_threshold = 0.1  # Velocity less than 0.1 m/s
        for i in range(self.num_drones):
            distance = np.linalg.norm(target_pos[i] - positions[i])
            velocity_magnitude = np.linalg.norm(velocities[i])
            if distance >= success_distance_threshold or velocity_magnitude >= success_velocity_threshold:
                all_reached = False
                break
        
        # Add success bonus if all drones reached target
        success_bonus = 100.0 if all_reached else 0.0
        
        # Early termination on success
        if all_reached:
            terminated = True
        
        # Combine terminated and truncated for done
        done = terminated or truncated
        
        # Compute reward per agent (pass actions for smoothness penalty)
        rewards_list, reward_infos = self._compute_reward(states, actions)
        
        # Add success bonus to all agents' rewards
        if success_bonus > 0:
            rewards_list = [r + success_bonus for r in rewards_list]
            for info in reward_infos:
                info['success_bonus'] = success_bonus
                info['total_reward'] += success_bonus
        
        # Compute observations - return as numpy arrays
        obs_n = raw_obs.astype(np.float32)  # Already (num_drones, obs_dim)
        share_obs = self._compute_share_obs(states)  # (num_drones, share_obs_dim)
        
        # MAPPO expects list format for rewards and dones
        rewards_n = [[r] for r in rewards_list]  # Per-drone individual rewards
        dones_n = [done for _ in range(self.num_drones)]
        
        # Track episode progress
        self._episode_step_count += 1
        self._episode_cumulative_reward += np.asarray(rewards_list, dtype=np.float32)
        self._episode_cumulative_r_form += float(reward_infos[0].get('r_form', 0.0))
        self._episode_cumulative_r_nav += np.asarray([ri.get('r_nav', 0.0) for ri in reward_infos], dtype=np.float32)
        self._episode_cumulative_r_dist += np.asarray([ri.get('r_dist', 0.0) for ri in reward_infos], dtype=np.float32)
        self._episode_cumulative_r_avoid += np.asarray([ri.get('r_avoid', 0.0) for ri in reward_infos], dtype=np.float32)
        self._episode_cumulative_r_tilt += np.asarray([ri.get('r_tilt', 0.0) for ri in reward_infos], dtype=np.float32)
        self._episode_cumulative_r_reached += np.asarray([ri.get('r_reached', 0.0) for ri in reward_infos], dtype=np.float32)
        self._episode_cumulative_r_vel_penalty += np.asarray([ri.get('r_vel_penalty', 0.0) for ri in reward_infos], dtype=np.float32)
        self._episode_cumulative_r_smooth += np.asarray([ri.get('r_smooth', 0.0) for ri in reward_infos], dtype=np.float32)
        if terminated:
            self._episode_terminated_early = True
        
        # Build info dicts (positions and velocities already extracted above)
        infos_n = []
        for i in range(self.num_drones):
            agent_info = {
                'individual_reward': rewards_list[i],
                'current_position': positions[i].tolist(),
                'current_velocity': velocities[i].tolist(),
                'success_reached': all_reached,
                **reward_infos[i]
            }
            # Add episode summary info on done (accessible by runner before auto-reset)
            if done:
                step_count = max(1, int(self._episode_step_count))
                agent_info['episode_summary'] = {
                    'initial_position': self._episode_initial_positions[i].tolist(),
                    'final_position': positions[i].tolist(),
                    'target_position': self._episode_target_positions[i].tolist(),
                    'initial_distance': float(self._episode_initial_distances[i]),
                    'final_distance': float(reward_infos[i]['dist_to_target']),
                    'distance_improvement': float(self._episode_initial_distances[i] - reward_infos[i]['dist_to_target']),
                    'reached_target': reward_infos[i]['dist_to_target'] < self.success_dist,
                    'direction_traveled': (positions[i] - self._episode_initial_positions[i]).tolist(),
                    'direction_to_target': (self._episode_target_positions[i] - self._episode_initial_positions[i]).tolist(),
                    'episode_avg_r_form': float(self._episode_cumulative_r_form / step_count),
                    'episode_avg_r_nav': float(self._episode_cumulative_r_nav[i] / step_count),
                    'episode_avg_r_dist': float(self._episode_cumulative_r_dist[i] / step_count),
                    'episode_avg_r_avoid': float(self._episode_cumulative_r_avoid[i] / step_count),
                    'episode_avg_r_tilt': float(self._episode_cumulative_r_tilt[i] / step_count),
                    'episode_avg_r_reached': float(self._episode_cumulative_r_reached[i] / step_count),
                    'episode_avg_r_vel_penalty': float(self._episode_cumulative_r_vel_penalty[i] / step_count),
                    'episode_avg_r_smooth': float(self._episode_cumulative_r_smooth[i] / step_count),
                }
                agent_info['episode_steps'] = self._episode_step_count
                agent_info['episode_cumulative_reward'] = float(self._episode_cumulative_reward[i])
                agent_info['episode_terminated_early'] = self._episode_terminated_early
                
                # print only cumulative formation and average formation reward for debugging
                # if i==0:
                #     print(f"Episode done in {self._episode_step_count} steps. Drone {i} cumulative formation reward: {self._episode_cumulative_r_form:.2f}, avg r_form: {agent_info['episode_summary']['episode_avg_r_form']:.4f}")
                
            infos_n.append(agent_info)
        
        # All actions available for continuous control
        available_actions = np.ones((self.num_drones, 4), dtype=np.float32)
        
        return obs_n, share_obs, rewards_n, dones_n, infos_n, available_actions
    
    def draw_position_markers(self):
        """Draw visual markers for initial and target positions in PyBullet GUI."""
        try:
            import pybullet as p
            
            # Remove old markers
            for marker_id in self._visual_marker_ids:
                try:
                    p.removeBody(marker_id)
                except:
                    pass
            self._visual_marker_ids.clear()
            
            # Draw markers for each drone
            for i in range(self.num_drones):
                # Green sphere for initial position
                initial_pos = self._episode_initial_positions[i]
                initial_visual = p.createVisualShape(
                    shapeType=p.GEOM_SPHERE,
                    radius=0.1,
                    rgbaColor=[0, 1, 0, 0.5]  # Green, semi-transparent
                )
                initial_marker = p.createMultiBody(
                    baseMass=0,
                    baseVisualShapeIndex=initial_visual,
                    basePosition=initial_pos
                )
                self._visual_marker_ids.append(initial_marker)
                
                # Red sphere for target position
                target_pos = self._episode_target_positions[i]
                target_visual = p.createVisualShape(
                    shapeType=p.GEOM_SPHERE,
                    radius=0.15,
                    rgbaColor=[1, 0, 0, 0.6]  # Red, semi-transparent
                )
                target_marker = p.createMultiBody(
                    baseMass=0,
                    baseVisualShapeIndex=target_visual,
                    basePosition=target_pos
                )
                self._visual_marker_ids.append(target_marker)
                
                # Draw a line from initial to target position
                p.addUserDebugLine(
                    lineFromXYZ=initial_pos,
                    lineToXYZ=target_pos,
                    lineColorRGB=[0.5, 0.5, 0.5],
                    lineWidth=2,
                    lifeTime=0  # Permanent until removed
                )
                
        except Exception as e:
            # Silently fail if PyBullet not available or GUI not active
            pass
    
    def render(self, mode: str = 'human'):
        """Render the environment."""
        return self._env.render()
    
    def close(self):
        """Close the environment."""
        # Clean up visual markers
        try:
            import pybullet as p
            for marker_id in self._visual_marker_ids:
                try:
                    p.removeBody(marker_id)
                except:
                    pass
        except:
            pass
        self._env.close()


def env_kwargs_from_args(all_args) -> Dict[str, Any]:
    """
    Build PyBulletDroneWrapper keyword arguments from the parsed CLI arguments.

    Training, evaluation and rendering must construct identical environments
    (same observation size, formation pool and reward), so they all go through here.
    """
    return dict(
        num_drones=all_args.num_drones,
        formation_spacing=all_args.formation_spacing,
        formation_type=all_args.formation_type,
        max_neighbors=all_args.max_neighbors,
        neighbour_radius=all_args.neighbour_radius,
        min_dynamic_neighbours=all_args.min_dynamic_neighbours,
        max_dynamic_neighbours=all_args.max_dynamic_neighbours,
        target_distance_range=(all_args.target_distance_min, all_args.target_distance_max),
        w_form=all_args.w_form,
        w_nav=all_args.w_nav,
        w_avoid=all_args.w_avoid,
        w_tilt=all_args.w_tilt,
        w_smooth=all_args.w_smooth,
        tilt_soft_threshold=all_args.tilt_soft_threshold,
        collision_dist=all_args.collision_dist,
        success_dist=all_args.success_dist,
    )


def make_pybullet_drone_env(all_args, gui: bool = False):
    """
    Factory function to create PyBulletDroneWrapper environment.
    
    Compatible with MAPPO's env creation pattern.
    
    Args:
        all_args: Namespace with environment arguments
        gui: Whether to open the PyBullet GUI
        
    Returns:
        PyBulletDroneWrapper environment
    """
    return PyBulletDroneWrapper(gui=gui, **env_kwargs_from_args(all_args))
