# Training code flow

Training starts from - [train_pybullet_drones.py](../onpolicy/scripts/train/train_pybullet_drones.py).

## Commands and Arguments

### Command to run training:  
```
python onpolicy/scripts/train/train_pybullet_drones.py --num_drones 8 --num_env_steps 8000000 --n_rollout_threads 8 --n_training_threads 6 --num_mini_batch 2 --save_interval 25 --experiment_name "malstm_gpu_10M"  --formation_type "dynamic" --neighbour_radius 1.0 --min_dynamic_neighbours 1 --max_dynamic_neighbours 7 
```
Arguments (see `config.py`): 
- `num_drones`: number of drones the model is trained on
- `num_env_steps`: total number of steps the model takes in the environment
- `n_rollout_threads`: the total number of processes that run in parallel in the system to execute an overall of  `num_env_steps` steps
- `n_training_threads`: number of PyTorch threads used during the neural network training phase
- `num_mini_batch`: number of mini-batches for PPO training updates
- `save_interval`: number of episodes between consecutive model checkpoints (.pt files updated after each such interval)
- `experiment_name`: identifier string to distinguish different experiments (used for logging and results organization)
- `formation_type`: formation geometry for training and evaluation episodes. Options: `polygon`, `line`, `plane`, `cube`, `sphere`, `pyramid`, or `dynamic` (which samples one of cube/sphere/pyramid/plane each episode)
- `neighbour_radius`: radius in meters for dynamic neighbor selection. Setting to 0 disables dynamic neighbors and uses fixed k-nearest neighbors instead
- `min_dynamic_neighbours`: minimum number of neighbor slots that should be filled (closest neighbors are always included even if outside the radius)
- `max_dynamic_neighbours`: maximum number of neighbor slots available (defines observation dimension; slots beyond actual neighbors are zero-padded) 

Other useful arguments:

- `resume_checkpoint`: Path to full-state checkpoint file (.pt) to resume training from. Restores the complete training state including optimizer, RNG, and training progress. Default: `None`
- `resume_partial_checkpoint`: Path to models folder containing `actor.pt` and `critic.pt`. Loads only the model weights and starts a fresh optimizer and training progress. Default: `None`

### Parameters used by the model:

Set in `main` function of `train_pybullet_drones.py`:

- **Learning rate**: Initial learning rate for actor and critic networks during gradient updates. Currently set to `5e-4`
- **Gamma**: Discount factor for future rewards in value function computation. Currently set to `0.99`
- **GAE Lambda**: Generalized Advantage Estimation (GAE) parameter for balancing bias and variance in advantage calculations. Currently set to `0.95`
- **Clip param**: PPO clipping parameter that limits the magnitude of policy updates per step. Currently set to `0.2`
- **Episode length**: Maximum number of control steps (at 30 Hz control frequency) before an episode is forcibly truncated. Currently set to `240` steps (~8 seconds)
- **Hidden_size**: Dimension of hidden layers in the actor and critic neural networks. Currently set to `256`

> Note: Training will automatically use cuda if you have torch with cuda installed.

### Model save location

The trained model is saved to: 

```
onpolicy\scripts\results\pybullet-drones\drones_{num_drones}\rmappo\{experiment_name}\run{run_number}\models
```

## Training environment initiation

As many training environments are created as number of rollout threads given in command line arguments.

Each environment is a Pybullet drone enviroment created using the `PybulletDroneWrapper` wrapper class:

```
env = PyBulletDroneWrapper(
                num_drones=all_args.num_drones,
                gui=False,  # No GUI for training
                formation_spacing=0.5,
                formation_type=all_args.formation_type,
                max_neighbors=all_args.max_neighbors,
                neighbour_radius=all_args.neighbour_radius,
                min_dynamic_neighbours=all_args.min_dynamic_neighbours,
                max_dynamic_neighbours=all_args.max_dynamic_neighbours,
            )
```

All the environments created, command line configurations and parameters are passed to the runner and `runner.run()` function starts the training process.

## Runner

File: `onpolicy\runner\shared\pybullet_drone_runner.py`

`PyBulletDroneRunner` class inherits from `Runner` class which initializes:
- policy - `MA_LSTM_Policy` in `onpolicy/models/ma_lstm_policy`
- training algorithm - `R_MAPPO` in `onpolicy/algorithms/r_mappo/r_mappo`
- a shared buffer across the processes - `SharedReplayBuffer` in `onpolicy/utils/shared_buffer.py`.

### `run` function

- Each process (rollout) runs this seprately and stores their data in the shared buffer.
- Number of episodes to be run by one process is calculated as follows:
```
episodes = int(self.num_env_steps) // self.episode_length // self.n_rollout_threads
```
- Next we have the main training loop.

## Main training loop

Pseudocode:

```
for episode in range(episodes):
    for step in range(episode_length): # episode_length is 240
        # 1. Get action from policy
        # 2. Take a step in the environment
        # 3. Collect rewards for that step from the environment and insert into the shared buffer
    
    # 4. Compute returns and update the network
    self.compute()
    train_infos = self.train()

    # 5. Post processing
    #   Save to .pt file after save_interval
    #   Print logs to console after each log_interval

self._save_training_history()
self._generate_training_plots()
self._save_last20_metrics()
```

### 1. Getting action from policy

```
values, actions, action_log_probs, rnn_states, rnn_states_critic, actions_env = self.collect(step)
```

What happens inside `collect(step)`:

1. `self.trainer.prep_rollout()` switches the policy to rollout/inference mode (no gradient updates in this stage).
2. The current step data is read from replay buffer:
    - `share_obs[step]`
    - `obs[step]`
    - `rnn_states[step]`
    - `rnn_states_critic[step]`
    - `masks[step]`
3. Data from all rollout threads is concatenated and passed into `policy.get_actions(...)`.
4. Policy returns tensors for value estimate, sampled action, action log-probability, and next RNN hidden states.
5. Outputs are converted to numpy and split back into per-thread layout.
6. RNN states are reshaped/transposed so they match buffer layout: `(n_rollout_threads, num_agents, recurrent_N, hidden_size)`.
7. `actions_env = actions.copy()` is created and sent to the environment in the next step.

Returned values:

- `values`: critic value predictions for each `(thread, agent)` at current step.
- `actions`: sampled policy actions for each `(thread, agent)`.
- `action_log_probs`: log-probabilities of sampled actions (used later in PPO objective).
- `rnn_states`: next actor RNN hidden states after action sampling.
- `rnn_states_critic`: next critic RNN hidden states.
- `actions_env`: action array passed directly to `envs.step(...)`.

For this drone setup, actions represent normalized velocity/throttle commands per drone:

- `actions_env[thread, agent] = [vx, vy, vz, throttle]` in range `[-1, 1]`.

### 2. Taking a step in the environment

```
obs, share_obs, rewards, dones, infos, available_actions = self.envs.step(actions_env)
```

**What this call does:**

1. `self.envs.step(actions_env)` sends one action per `(thread, agent)` to each rollout environment.
2. Inside each drone env (`PyBulletDroneWrapper.step`), those actions are passed to `MultiHoverAviary.step`.
3. One control step executes multiple physics steps:
     - `PYB_FREQ = 240 Hz`, `CTRL_FREQ = 30 Hz`
     - `PYB_STEPS_PER_CTRL = 240 / 30 = 8`
     - so each RL step advances the simulator by 8 PyBullet integration steps.
4. After physics updates, the env computes observation, termination/truncation flags, and info.
5. The wrapper computes shaped per-agent rewards (`compute_reward` function, more on this later) and, on global success, adds a success bonus.

**Termination and truncation flags:**

- `terminated`: set by `_computeTerminated` in `MultiHoverAviary.py`. True if all drones are within 5cm of their target position.
- `truncated`: set by `_computeTruncated` in `MultiHoverAviary.py`. True if:
    - one of the drones is too far away (any one of its absolute x,y,z position is more than 5m) or
    - one of the drones is tilted too much (any one of its absolute roll/pitch angles is more than 1.0 radian) or
    - if the episode time limit is reached (8 seconds).
- If all drones reach target (within 20cm) with low velocity (less than 0.1 m/s), wrapper sets `terminated` flag to `True` and adds `success_bonus = 100.0` to each drone reward for that final step.
- Wrapper combines the two flags as `done = terminated or truncated`.

**Sub-episodes and auto-reset behavior:**

- The runner loop still executes a fixed `episode_length` rollout horizon.
- If one env reaches `done` before that horizon, vector wrappers (`ShareSubprocVecEnv` / `ShareDummyVecEnv`) immediately call `reset()` for that env and continue collecting data.
- This creates multiple environment episodes inside one rollout horizon; these are effectively sub-episodes.
- Because of that, initial/target positions and per-episode counters reset whenever `done` occurs, not only at outer-loop boundaries.

**Role of `dones`:**

- `dones` is per-thread and per-agent.
- In this environment, `done` is shared across agents in the same thread (all agents in that env get the same done flag).
- `dones` is used by runner/buffer logic to reset RNN states and masks for environments that just ended.
- In the code, this happens in the runner's `insert(...)` step when the collected transition is written into the shared replay buffer.

**Exact reset flow:**

1. In [onpolicy/runner/shared/pybullet_drone_runner.py](../onpolicy/runner/shared/pybullet_drone_runner.py), `run()` calls `self.envs.step(actions_env)`.
2. `self.envs` is the vectorized wrapper created in the train script (either `ShareSubprocVecEnv` or `ShareDummyVecEnv` from [onpolicy/envs/env_wrappers.py](../onpolicy/envs/env_wrappers.py)).
3. In [onpolicy/envs/env_wrappers.py](../onpolicy/envs/env_wrappers.py), each vector env executes `env.step(action)` where `env` is `PyBulletDroneWrapper`.
4. In [onpolicy/envs/pybullet_drone_env.py](../onpolicy/envs/pybullet_drone_env.py), `PyBulletDroneWrapper.step()` computes `done = terminated or truncated` and returns `dones_n` (same done value for all drones in that env).
5. Back in [onpolicy/envs/env_wrappers.py](../onpolicy/envs/env_wrappers.py), the wrapper checks `done` (`np.all(done)` for multi-agent) and immediately calls `env.reset()` for that env.
6. That `env.reset()` call goes to [onpolicy/envs/pybullet_drone_env.py](../onpolicy/envs/pybullet_drone_env.py) `PyBulletDroneWrapper.reset()`.
7. Inside that reset, it first calls `self._env.reset()`, where `self._env` is `MultiHoverAviary` in [onpolicy/envs/gym_pybullet_drones/envs/MultiHoverAviary.py](../onpolicy/envs/gym_pybullet_drones/envs/MultiHoverAviary.py).
8. In `MultiHoverAviary.reset()`, formation state is rebuilt: dynamic formation type can be resampled, and `INIT_XYZS` and `TARGET_POS` are regenerated.
9. Control returns to `PyBulletDroneWrapper.reset()`, which resets per-episode counters and cumulative trackers, and stores fresh initial/target positions for logging.
10. The vector wrapper returns the (possibly reset) next observations plus the original `dones` from the just-finished step back to the runner.
11. Then `runner.insert(...)` in [onpolicy/runner/shared/pybullet_drone_runner.py](../onpolicy/runner/shared/pybullet_drone_runner.py) uses those `dones` to build `done_mask`, zero RNN states for ended env-agent entries, set masks to 0, and write into the shared buffer.

*Important distinction:*

- Environment reset call site: `env_wrappers.step_wait()`.
- RNN/mask reset call site: `pybullet_drone_runner.insert(...)`.

**Cumulative reward tracking (high level):**

- The wrapper keeps per-episode running totals such as:
    - cumulative total reward per drone
    - cumulative reward components (for diagnostics)
    - step count and early-termination marker
- When `done=True`, these summaries are attached into `infos` as episode metadata, then reset on the next auto-reset.

**What each returned value contains:**

- `obs`: per-agent local observations after the step.
    - shape: `(n_rollout_threads, num_drones, obs_dim)`.
- `share_obs`: centralized observation (same global state tiled per agent).
    - shape: `(n_rollout_threads, num_drones, share_obs_dim)`.
- `rewards`: per-agent scalar rewards for this step.
    - shape: `(n_rollout_threads, num_drones, 1)`.
- `dones`: done flags for this step.
    - shape: `(n_rollout_threads, num_drones)`.
- `infos`: per-agent dictionaries with diagnostics (position/velocity, reward terms, and episode summary fields when done).
- `available_actions`: action-availability mask.
    - continuous control here, so all actions are available (ones mask).

### 3. Collecting rewards and inserting into buffer

```
data = (obs, share_obs, rewards, dones, infos, values, actions, action_log_probs, rnn_states, rnn_states_critic)

self.insert(data)
```

**What this step does:**

1. The runner packs one transition into `data`:
    - observations after env step (`obs`, `share_obs`)
    - reward/done/info from env (`rewards`, `dones`, `infos`)
    - policy-side outputs from Step 1 (`values`, `actions`, `action_log_probs`, `rnn_states`, `rnn_states_critic`)
2. `insert(data)` unpacks this tuple in [onpolicy/runner/shared/pybullet_drone_runner.py](../onpolicy/runner/shared/pybullet_drone_runner.py).
3. It normalizes `infos` format first (important for `n_rollout_threads=1`):
    - if `infos` is a numpy object array (from `ShareDummyVecEnv`), it is converted to a Python list.
4. It converts `obs`, `share_obs`, `rewards`, `dones` to numpy arrays if needed.
5. It fixes dimensions to match buffer expectations:
    - adds rollout dimension when missing
    - ensures `dones` has trailing singleton dim before masking logic.
6. It builds `done_mask` from `dones`.
7. For every `(rollout_thread, agent)` where done is true:
    - actor and critic RNN states are zeroed (hidden-state reset at episode boundary)
    - episode summary is read from `infos` and appended to `recent_episode_completions` for logging.
8. It creates `masks` with shape `(n_rollout_threads, num_agents, 1)`:
    - `1.0` for ongoing trajectories
    - `0.0` where episode ended (used by return/advantage logic to cut sequence continuity).
9. It writes the transition to the shared replay buffer via `self.buffer.insert(...)`.

**Important clarification:**

- Reward computation is already finished in Step 2 (environment wrapper). Step 3 does not recompute rewards; it stores them and prepares correct recurrent/mask boundaries for training.
- Environment reset itself is handled in vector wrappers (`env_wrappers.py`) during `envs.step(...)`; Step 3 handles model-state reset (RNN + masks) for learning.

**What goes into the shared buffer at this step:**

- `share_obs`, `obs`
- `rnn_states`, `rnn_states_critic` (after done-based zeroing)
- `actions`, `action_log_probs`, `values`
- `rewards`
- `masks`

`infos` is not inserted into the replay buffer tensor storage; it is used for diagnostics/logging.

### 4. Compute returns and update the network

```
self.compute()
train_infos = self.train()
```

**`self.compute()`: Returns calculation**

1. Switches model to rollout/eval mode (`prep_rollout`) in [onpolicy/runner/shared/base_runner.py](../onpolicy/runner/shared/base_runner.py).
2. Gets critic value for the last buffer state (`buffer.share_obs[-1]`, `buffer.rnn_states_critic[-1]`, `buffer.masks[-1]`).
3. Splits the predicted values back by rollout thread.
4. Calls `buffer.compute_returns(next_values, value_normalizer)` in [onpolicy/utils/shared_buffer.py](../onpolicy/utils/shared_buffer.py).
5. Returns are computed backward through time using configured settings:
    - with GAE (default):
      - $\delta_t = r_t + \gamma V_{t+1} \cdot mask_{t+1} - V_t$
      - $GAE_t = \delta_t + \gamma \lambda \cdot mask_{t+1} \cdot GAE_{t+1}$
      - $return_t = GAE_t + V_t$
    - masks cut the return chain at done boundaries.
    - if value normalization is enabled, denormalization/normalization is applied around value targets.

**`self.train()`: Policy update**

1. Switches model to training mode (`prep_training`) in [onpolicy/runner/shared/base_runner.py](../onpolicy/runner/shared/base_runner.py).
2. Calls trainer update loop (`R_MAPPO.train`) in [onpolicy/algorithms/r_mappo/r_mappo.py](../onpolicy/algorithms/r_mappo/r_mappo.py).
3. Computes advantages as `returns - value_preds` (denormalized if needed), then normalizes advantages.
4. Runs PPO updates for `ppo_epoch x num_mini_batch`:
    - samples mini-batches from replay buffer generators
    - evaluates current policy log-probs and values
    - computes importance ratio $r_t = \exp(\log\pi_{new} - \log\pi_{old})$
    - computes clipped actor objective `min(r_t*A_t, clip(r_t, 1-eps, 1+eps)*A_t)`
    - adds entropy regularization
    - backprop + gradient clipping + optimizer step for actor
    - computes critic value loss (with value clipping option), then backprop + optimizer step for critic
5. Averages training metrics into `train_infos` (policy loss, value loss, entropy, grad norms, ratio).
6. Calls `buffer.after_update()` to carry final timestep state to index 0 for the next rollout window.


### 5. Post processing

**After each episode:**

- Compute `total_num_steps = (episode + 1) * episode_length * n_rollout_threads`.
- Save model weights and full resume checkpoint at `save_interval` (and on last episode).
- At `log_interval`, print training status (episode/steps/FPS), aggregate env metrics, and log train/env stats.

#### Metrics printed at each log interval

All episodic metrics below are computed from `recent_episode_completions` — the list of episodes that completed (across all rollout threads) since the last log point. Each metric averages episode sums from every drone-episode completion in that window, so with 8 rollout threads and `log_interval=5` episodes, up to 8 × 8 × 5 = 320 individual drone-episode completions can contribute to a single logged value.

| Printed label | Variable | How it is calculated |
|---|---|---|
| `Average episodic reward` | `avg_reward` | Mean of `episode_cumulative_reward` across all completions in the window. Fallback: `mean(buffer.rewards) × episode_length` when no episode completed. |
| `Formation error` | `formation_error` | Mean of `formation_error` values from the last step's `infos` across all rollout threads. Reflects the Procrustes error $E^t$ at that single step, not an episode average. |
| `Avg distance to target` | `avg_distance` | Mean of `dist_to_target` taken from `infos` at the final step of completed episodes. Units: metres. |
| `Episodic formation reward` | `episodic_sum_r_form` | For each completion: `episode_avg_r_form × episode_steps` (recovers the true episode sum). Then mean across all completions. Formula per step: $r_{\text{form}}^t = -E^t / (G + \varepsilon)$ where $E$ is Procrustes MSE and $G$ is max pairwise distance squared of the target formation. Shared across all drones. Range: $(-\infty, 0]$. |
| `Episodic navigation reward` | `episodic_sum_r_nav` | Same aggregation. Formula per step per drone $i$: $r_{\text{nav},i}^t = d_i^{t-1} - d_i^t$. Episode sum telescopes to $d_i^0 - d_i^T$ (net approach in metres). Positive = drone got closer overall. |
| `Episodic distance penalty` | `episodic_sum_r_dist` | Same aggregation. Formula: $r_{\text{dist},i}^t = -d_i^t$. Episode sum $= -\bar{d}_i \cdot T$ (integrates the full trajectory, not just start/end). More negative = drone spent the episode far from target. |
| `Episodic collision penalty` | `episodic_sum_r_avoid` | Same aggregation. Formula: $r_{\text{avoid},i}^t = -1$ if drone $i$ is within 0.1 m of any other drone, else 0. Episode sum = $-$(steps spent in collision). |
| `Episodic tilt penalty` | `episodic_sum_r_tilt` | Same aggregation. Formula: $r_{\text{tilt},i}^t = -\max(0,\; \max(\lvert \phi_i^t \rvert, \lvert \theta_i^t \rvert) - 0.35)^2$ where $\phi$, $\theta$ are roll and pitch in radians. Zero when tilt ≤ 0.35 rad (~20°), quadratic beyond. |
| `Episodic reaching bonus` | `episodic_sum_r_reached` | Same aggregation. Formula: $r_{\text{reached},i}^t = 3.0$ if $d_i^t < 0.20$ m, else 0. Episode sum $= 3.0 \times$ (steps within 20 cm of target). |
| `Targets reached` | `num_reached / total_drones` | Count of drones whose `final_distance < 0.50` m at episode end, shown as fraction and percentage. |

#### Plots generated at end of training


All plots are saved to `plots/` inside the model save directory. Each plot shows the raw per-log-point series (faint, `alpha=0.3`) and a smoothed moving average with `window = max(1, N // 20)` where N is the number of logged points.

| File | Content | Y-axis units |
|---|---|---|
| `training_summary.png` | Figure 1 left: `avg_rewards` over timesteps. Figure 1 right: `formation_errors` over timesteps. | Reward (unitless) / Formation error $E$ |
| `distance_distribution.png` | Figure 2: box plots of per-drone final distances at 10 evenly-spaced episodes. Red dashed line at 5 cm goal threshold. | Metres |
| `reward_components.png` | Figure 3 left: `episodic_sum_r_form`. Figure 3 right: `episodic_sum_r_nav`. | Episode sum (unitless) |
| `penalty_bonus_components.png` | Figure 4 (2×2 grid): top-left `episodic_sum_r_dist`, top-right `episodic_sum_r_avoid`, bottom-left `episodic_sum_r_tilt`, bottom-right `episodic_sum_r_reached`. | Episode sum (unitless) |



**After all episodes:**

- Save full training history (`_save_training_history()`).
- Generate training plots (`_generate_training_plots()`).
- Save last 20% episodes summary metrics (`_save_last20_metrics()`).

#### End-of-training summary metrics (`last20_metrics.json`)

Computed over the last 20% of logged episodic data points (i.e., `training_history` entries from index `floor(0.8 × N)` to `N`):

- `FAR` (Formation Accuracy Rate): mean of `episodic_sum_r_form` over the window. Higher (less negative) = better average formation quality in the final phase.
- `FS` (Formation Stability): $1 / (1 + \text{std}(\texttt{episodic\_sum\_r\_form}))$ over the window. Range $(0, 1]$; closer to 1 = more consistent formation performance.
- `NAR` (Navigation Accuracy Rate): mean of `episodic_sum_r_nav` over the window. Higher = drones ended closer to targets on average.
- `NS` (Navigation Stability): $1 / (1 + \text{std}(\texttt{episodic\_sum\_r\_nav}))$ over the window. Range $(0, 1]$; closer to 1 = more consistent navigation performance.

These are computed in [onpolicy/runner/shared/pybullet_drone_runner.py](../onpolicy/runner/shared/pybullet_drone_runner.py) inside `_save_last20_metrics()` and written to `plots/last20_metrics.json`.


## Model Architecture

Each drone is treated as an independent agent. Training is centralized and execution is decentralized (CTDE): the actor uses local observations, while the critic sees shared global state. The main policy and training logic are implemented in [onpolicy/models/ma_lstm_policy.py](../onpolicy/models/ma_lstm_policy.py) and [onpolicy/algorithms/r_mappo/r_mappo.py](../onpolicy/algorithms/r_mappo/r_mappo.py).

### Learning Objective

The policy is trained to learn three behaviors at the same time:

1. Formation maintenance: keep the swarm in the desired 3D formation.
2. Navigation: move the formation toward the target region.
3. Collision avoidance: avoid unsafe drone proximity and unstable flight.

### State / Observation

Each drone receives a local observation built from:

- its own position
- its own velocity
- relative target position
- neighbor positions
- neighbor velocities

The observation layout and dynamic-neighbor handling are implemented in [onpolicy/envs/pybullet_drone_env.py](../onpolicy/envs/pybullet_drone_env.py) and mapped into observation spaces in [onpolicy/envs/gym_pybullet_drones/envs/MultiHoverAviary.py](../onpolicy/envs/gym_pybullet_drones/envs/MultiHoverAviary.py).

The state is organized as:

- Own state: `[x, y, z, v_x, v_y, v_z]`
- Relative target position: `[x_t - x, y_t - y, z_t - z]`
- Neighbor information for `k` nearest neighbors:
    - relative position: 3 values
    - relative velocity: 3 values

So the observation dimension is:

$$
|State| = 9 + 6k, \quad \text{if } k \leq n - 1
$$

$$
|State| = 9 + 6(n - 1), \quad \text{if } k > n - 1
$$

### Actor Network

The actor is the policy network that outputs the drone action. It processes the local observation through normalization, a fully connected layer, and an LSTM before producing a continuous action. The action is a high-level command of the form:

$$
a_i = [v_x, v_y, v_z, thrust]
$$

The action is not sent directly to motors. Instead, the environment/controller stack converts it into the lower-level thrust or RPM commands used by the PyBullet quadrotor.

### Why LSTM Is Used

LSTM is used because drone motion is temporal. The network needs memory of what happened in earlier steps to learn stable behavior over a full trajectory. The recurrent state handling is coordinated through the runner and buffer in [onpolicy/runner/shared/pybullet_drone_runner.py](../onpolicy/runner/shared/pybullet_drone_runner.py).

### Critic Network

The critic is used only during training. It receives the shared state of the whole swarm and estimates how good the current state is in the long run. This is the centralized part of CTDE. The critic also uses recurrence so it can track state changes over time.

### Control Pipeline

The end-to-end pipeline is:

1. The drone observes its state.
2. The actor processes the local observation.
3. The actor outputs a continuous action.
4. The environment/controller converts the action into physical control.
5. The drone moves in PyBullet.
6. The environment returns the next observation and reward.
7. The critic evaluates the updated state during training.

## Reward Structure

At each timestep, the environment computes a shared reward for all agents. The reward logic is implemented in [onpolicy/envs/pybullet_drone_env.py](../onpolicy/envs/pybullet_drone_env.py) and uses formation utilities from [onpolicy/utils/formation.py](../onpolicy/utils/formation.py).

The reward is built from these components:

- `r_form`: formation reward based on Procrustes-aligned formation error.
- `r_nav`: navigation reward based on distance improvement from the previous step.
- `r_avoid`: collision-avoidance penalty when drones get too close.
- `r_reached`: bonus when a drone reaches its target region.
- `r_tilt`: soft tilt penalty when roll/pitch exceeds the stability threshold.
- `success_bonus`: extra bonus when all drones reach target with low velocity.

Current values in code ([onpolicy/envs/pybullet_drone_env.py](../onpolicy/envs/pybullet_drone_env.py)):

- Collision threshold (`d_avoid`) = `collision_dist = 0.1` m.
- `w_form = 0.5`
- `w_nav = 10.0`
- `w_avoid = 2.0`
- `w_tilt = 5.0`
- `w_dist = 0.5`
- `tilt_soft_threshold = 0.35` rad
- `r_reached = +3.0` when distance < `0.2` m
- `success_bonus = +100.0` (applied when all drones reach target with low velocity)


$$
r_i = w_{form} r_{form} + w_{nav} r_{nav,i} + w_{avoid} r_{avoid,i} + w_{tilt} r_{tilt,i} + r_{reached,i}
$$


The wrapper combines these components into the final step reward, and the runner stores the result in [onpolicy/runner/shared/pybullet_drone_runner.py](../onpolicy/runner/shared/pybullet_drone_runner.py).

### Formation Reward

The formation reward measures how closely the current drone configuration matches the desired formation, independent of global translation and rotation. It is computed using a Procrustes-style alignment:

$$
r_{form} = -\frac{E(F, F^*)}{G(F^*) + \epsilon}
$$

where:

- $E(F, F^*)$ is the formation error after optimal rigid-body alignment
- $G(F^*)$ is the normalization term based on the target formation
- $\epsilon$ is a small stability constant

### Navigation Reward

The navigation reward encourages drones to move toward their assigned targets:

$$
R_{nav} = D_i(t-1) - D_i(t)
$$

where $D_i(t)$ is the distance of drone $i$ from its target at time $t$.

This reward is positive when drones move closer to the target and negative when they move away.

### Collision Avoidance Penalty

To keep the swarm safe, a collision penalty is applied when drones get too close:

$$
R_{avoid} =
\begin{cases}
0, & d_{i,j} > d_{avoid} \\
-1, & d_{i,j} \le d_{avoid}
\end{cases}
$$

where $d_{avoid}$ is the minimum safe distance threshold.

### Tilt Penalty

To discourage unstable flight attitudes, the reward includes a soft tilt penalty based on roll and pitch. For each drone $i$ at timestep $t$:

$$
r_{tilt,i}^t = -\max\left(0,\; \max\left(\lvert \phi_i^t \rvert, \lvert \theta_i^t \rvert\right) - \theta_{soft}\right)^2
$$

where:

- $\phi_i^t$ is roll (rad), $\theta_i^t$ is pitch (rad)
- $\theta_{soft} = 0.35$ rad (about 20 degrees)

### Reaching Bonus

A bonus is given when drones get sufficiently close to their targets:

$$
r_{reached} = +1 \quad \text{if drones are within } 0.2\,\text{m of the target}
$$

### Dynamic Neignbourhood Design

To keep the observation size fixed as the swarm changes, each drone uses only local neighbor information. This dynamic-neighbor setup is configured in [onpolicy/envs/pybullet_drone_env.py](../onpolicy/envs/pybullet_drone_env.py) and the formation generation logic is in [onpolicy/envs/gym_pybullet_drones/envs/MultiHoverAviary.py](../onpolicy/envs/gym_pybullet_drones/envs/MultiHoverAviary.py).

- `max_dynamic_neighbours` fixes the observation size.
- `neighbour_radius` controls which neighbors are included.
- `min_dynamic_neighbours` ensures the closest slots are always filled.
- `formation_type="dynamic"` samples a new formation at the start of each episode.

This lets the same policy train on one swarm size and generalize to other swarm sizes as long as the observation layout stays compatible.

### Formation types

Configurable `formation_type` options (from [onpolicy/scripts/train/train_pybullet_drones.py](../onpolicy/scripts/train/train_pybullet_drones.py)):

- `polygon`: drones are arranged around a circle/polygon pattern.
- `line`: drones are arranged along a straight line.
- `plane`: drones are arranged on a 2D grid/plane.
- `cube`: drones are arranged on a 3D cubic lattice.
- `sphere`: drones are arranged over a sphere-like surface.
- `pyramid`: drones are arranged in a stepped pyramid-like structure.
- `dynamic`: formation type is resampled each episode from `{cube, sphere, pyramid, plane}` in [onpolicy/envs/gym_pybullet_drones/envs/MultiHoverAviary.py](../onpolicy/envs/gym_pybullet_drones/envs/MultiHoverAviary.py).