# Audit: ALiGn paper vs. `my-mappo` implementation

This compares the draft *"ALiGn: Adaptive Local Coordination for Global Harmony in IoT-Edge UAV
Swarms"* (`2026_Distributed_MARL_Submission.pdf`) with the code the previous student left in
`my-mappo.zip`.

* The code is imported unchanged in commit `47f202e` (baseline).
* Bug fixes are in the following commit. Line numbers below refer to the **original** files.

The code is a fork of the MAPPO `on-policy` repository (Yu et al.). It adds a PyBullet drone
environment (`gym-pybullet-drones` `MultiHoverAviary`), an FC-LSTM-FC policy and a drone runner.
Everything else (SMAC, Hanabi, Football, MPE, HAPPO, MAT, …) is upstream code that the drone
pipeline does not use.

---

## 1. What is implemented

| Paper claim | Where | Status |
|---|---|---|
| MAPPO with CTDE, GAE, advantage normalisation, grad clipping, value normalisation | `algorithms/r_mappo/r_mappo.py`, `utils/shared_buffer.py` | ✅ (upstream MAPPO) |
| Shared FC-LSTM-FC actor (256 → LSTM 256 → 4), LayerNorm on input and after the LSTM, Gaussian head | `models/ma_lstm_policy.py` | ✅ The paper says ReLU (Fig. 2); the code uses **Tanh**. Recurrent training was broken, see §3.1. |
| Centralised FC-LSTM-FC critic on the global state | `models/ma_lstm_policy.py`, `envs/pybullet_drone_env.py:_compute_share_obs` | ✅ Global state = position, velocity and target of every drone (9·N) |
| Observation `[p_i, v_i, Δp_i, {p_ij, v_ij}]`, max *k* neighbours, zero padding | `envs/gym_pybullet_drones/envs/BaseRLAviary.py:_computeObs` | ✅ Obs dim 51 for N=8, k=7, matching §V-D |
| Radius-based adaptive neighbour selection | `BaseRLAviary.py:383-396` | ✅ Also keeps `min_dynamic_neighbours` (default 1) even outside the radius, which the paper does not mention |
| 4-D action `[dir_x, dir_y, dir_z, speed]` → target velocity → PID | `BaseRLAviary.py:229-244` (DSLPIDControl) | ✅ Max speed 0.05·30 km/h ≈ 0.42 m/s |
| Rewards r_nav, r_avoid, r_tilt, r_smooth, r_reached | `envs/pybullet_drone_env.py:_compute_reward` | ✅ Also has an undocumented +100 all-reached bonus with early termination, plus r_dist and a velocity penalty (both weight 0) |
| Procrustes/Kabsch formation error E/G | `utils/formation.py` | ✅ Rotation maths are correct. It returned a sum, not the documented mean (fixed). |
| Formation pool (cube, sphere, pyramid, plane) sampled per episode | `MultiHoverAviary.py` (`formation_type=dynamic`) | ✅ 3-D shapes were flattened into the ground, see §3.3 |
| Train with 8, run inference with more drones | obs size fixed by `max_dynamic_neighbours` | ✅ Works whenever N−1 ≥ k |
| Checkpoint/resume, training plots | `runner/shared/pybullet_drone_runner.py` | ✅ |

## 2. What is **not** implemented (claimed in the paper, absent in code)

| Paper section | Claim | Finding |
|---|---|---|
| §III-A, §IV-F, §V-A | *"drones are initialised on the ground … first construct the desired formation"* | **Not implemented.** `MultiHoverAviary._generate_formation_positions` spawns drones **already in formation** at z ∈ [0.2, 1.0] m with 5 cm noise. The paper's own margin note (Yash, p. 3) says the same. |
| §IV-F | Staged reward (w_form = 0 until formation error < threshold, then enabled) | **Not implemented.** There is no staging logic. |
| §IV-A | Formation reward is part of the objective | **Weight is 0.** `PyBulletDroneWrapper(w_form=0)` (`pybullet_drone_env.py:36`), and no script changes it. "Formation keeping" only emerges because each drone flies to its own absolute slot at the target formation. That is close to the rigid per-drone straight-line approach that §III-A argues against. |
| §IV-E, Table III | Waypoint-guided long-distance execution (1 m waypoints; 5/10/20 m tests) | **Not implemented.** There is no waypoint code anywhere. Also, the observation contains the **absolute position** p_i (arena ±2.5 m during training), so distant targets are out of distribution even with waypoints. §III-C says "all positional information is relative", which contradicts the observation definition just above it. |
| §I contributions, §IV-C.1 (Eq. 10-13) | ANIM / optimal neighbour selection, rigidity bound, communication-aware reward (`+λΣ\|N_i\|`) | **Not implemented.** Neighbour choice is purely "within radius, closest k". There is no communication cost in the reward. Eq. 12 is a DMPC cost that nothing in the code uses. |
| §I contributions | "Dynamic changing of shape on the fly" | **Not implemented.** The shape is fixed per episode. |
| §I contributions, §VI | Static and dynamic obstacles | **Not implemented.** Listed as future work in §VI but claimed as validated in §I. |
| §V-A…§V-C, Tables I–III, Fig. 6 | Results for 32/64/128 drones, 600-run averages, waypoint ablation, neighbour sweep | **No evaluation scripts in the repo.** Only an interactive GUI render loop exists, so the tables cannot be reproduced from this code. The formation-error definition changed (sum → mean), so any numbers produced with the old code are N× larger than the mean. |
| §V-D | INT8 post-training quantisation for Crazyflie | Not implemented (the paper says so). The parameter counts in Table V are correct (541,802). The text formula `4(nh+h²+h)×2` is wrong; it should be `4(nh+h²+2h)`. |
| §V-B | Training: 25 M steps, target distance 1–3 m | README says 8 M, `command.txt` 100 K. `target_distance_range` defaults to **0.5–2.0 m**, and the target centre is clipped to ±2.5 m. |
| — | Hold position at the target | Episodes **terminate** as soon as every drone is within 0.2 m with v < 0.1 m/s (+100 bonus), so holding is never trained. |

Other design limitations (not bugs, but worth knowing):

* Zero-padded neighbour slots are indistinguishable from a neighbour at exactly the same position
  and velocity, i.e. a collision. A validity-mask bit per slot would remove the ambiguity.
* Formation yaw is always 0 (random yaw is commented out), so the policy never sees rotated
  formations.
* `collision_dist = 0.1 m` is about the CF2X diagonal, so it detects physical contact, not a
  safety margin.
* `onpolicy/controllers/pid_controller.py` is unused. The environment uses DSLPIDControl.
* Time-limit truncation is treated as terminal (no bootstrapping). This is common in MAPPO, and
  it is harmless here because `episode_length` equals the horizon.

## 3. Bugs found and fixed

All fixes are covered by `my-mappo/tests/` (run: `cd my-mappo && PYTHONPATH=. pytest tests -q`).
On the original code, 6 of the 7 tests fail.

### 3.1 Recurrent (LSTM) training was incorrect — *critical*

1. **Hidden states were given to the wrong drones during rollouts.**
   The policy returns hidden states shaped `(batch, 2, H)`, and `PyBulletDroneRunner.collect`
   (`pybullet_drone_runner.py:332`) reshaped them as `(2, threads, agents, H)`. So at every
   step each drone received another drone's h/c state (h and c interleaved across agents).
   The test measured 87.5 % of stored states wrong.
   *Fix:* split along the batch axis as every other tensor does.
2. **No backprop through time, and wrong memory in the PPO ratio.**
   `SharedReplayBuffer.recurrent_generator` (`shared_buffer.py:594`) tiled each chunk's
   *initial* hidden state over all 10 steps. `LSTMActor.forward` then ran each step
   independently from that state. So the log-probs used in the PPO ratio were computed with a
   different memory than at collection time: 74 % of ratios ≠ 1 before any update, even for a
   freshly initialised policy. The LSTM also never learned temporal dependencies.
   *Fix:* the buffer yields one state per chunk (upstream behaviour), and `run_lstm()` unrolls
   the actor and critic over the chunk, resetting at episode boundaries (`masks == 0`).

Any model trained with the original code (and so every number in the paper) was trained with
these two bugs.

### 3.2 Evaluation / rendering could not run

3. `eval()` unpacked 2 values from `reset()` (returns 3) and 5 from `step()` (returns 6), so it
   crashed whenever `--use_eval` was set. Fixed; it now also logs the eval formation error.
4. The render script built the environment **without** the training flags: `neighbour_radius=0`
   and `formation_type=polygon`. It also used `hidden_size=64` (training forces 256) and
   `episode_length=384` (the env runs for 240 steps). A model trained with `command.txt`
   (obs 51, hidden 256) therefore could not be loaded.
   Train, eval and render now share `env_kwargs_from_args()`, and the render scripts pass the
   training flags.
5. `Runner.restore` called `torch.load` without `map_location`, so GPU checkpoints failed to
   load on CPU machines.

### 3.3 Environment

6. **3-D formations were flattened into the ground.** `_apply_formation` clamps z ≥ 0.05. With a
   formation centre at z ∈ [0.2, 1.0] m, the lower layers of cube, sphere and pyramid shapes
   (which extend 0.5–1 m below their centroid) were clamped to 0.05 m. This changed the target
   shape (e.g. `Should reach: [.., +0.05]` for 3 of 8 cube drones in the training log) and put
   targets on the floor. The formation is now lifted as a whole.
7. **Formation error returned the sum over drones**, although the docstring, the comments and
   the paper describe the mean. It now returns the per-drone mean.
8. **Inconsistent "reached" threshold**: 0.2 m for the reward and termination, 0.5 m for the
   logged `reached_target`. There is now one `--success_dist` value (default 0.2).

### 3.4 Usability

9. Reward weights (`--w_form`, `--w_nav`, `--w_avoid`, `--w_tilt`, `--w_smooth`), thresholds and
   the target-distance range are now CLI flags. The defaults are unchanged, so the paper's
   ablations can be run without editing code. The no-op `--use_formation_reward` flag (a
   store_true that defaulted to True and did nothing) was removed.
10. `boxplot(labels=)` crashed the final plots on matplotlib ≥ 3.11 (renamed to `tick_labels`).
11. `render_drones.sh/.bat` had hard-coded Windows paths and wrong defaults.

## 4. Recommendations for the paper

* Retrain after the LSTM fixes before reporting any numbers. The previous runs trained a
  "recurrent" policy that could not use its memory correctly.
* Either implement or remove: ground take-off and formation construction, staged reward,
  waypoints, communication-aware selection, obstacles, and on-the-fly shape change. The
  OmniDrones environment in `omnidrones_formation/` implements take-off → formation → waypoint
  navigation → hold, with static/dynamic obstacles and a staged formation reward.
* Add an evaluation script that produces Tables I–III (fixed seeds, N runs, mean ± std).
* Remove absolute position from the observation, or justify it; it conflicts with the
  long-distance / waypoint claim.
