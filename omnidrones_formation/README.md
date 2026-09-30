# FormationNav for OmniDrones

A UAV swarm **takes off from the ground**, **builds a given formation**, flies it to a
goal **through static and/or moving obstacles (or none)**, and **holds the formation at
the goal**. Training uses the paper's **FC-LSTM-FC MAPPO** (shared decentralised actor,
centralised critic). The drones are **Crazyflies** by default (the paper's target platform);
OmniDrones' Hummingbird is available as a preset.

**Installation:** see [docs/INSTALL.md](docs/INSTALL.md). Check the GPU first: Isaac Sim 4.1,
and therefore OmniDrones, does not run on Blackwell GPUs (RTX 50xx / RTX PRO Blackwell).

```
 FORM (take off + build formation) ──► NAV (waypoints, avoid obstacles) ──► HOLD (at goal)
```

| Scenario | Obstacles |
|---|---|
| `none` | empty route: the baseline behaviour |
| `static` | 2–4 vertical pillars (r = 0.3–0.6 m) in a 5 m corridor along the route |
| `dynamic` | 1–3 spheres crossing the route back and forth (0.3–1.0 m/s) |
| `mixed` | both |
| `train_mix` | a random scenario per episode (default for training) |

## How it maps to the paper

| Paper | Here |
|---|---|
| Start on the ground, build the formation first (§III-A, §IV-F) | FORM phase: planar take-off grid, crossing-free drone→slot assignment, **staged take-off** (the formation rises as a block, top layer first, see below), NAV starts when the staging is complete and every drone has been within 0.35 m of its slot for 15 steps |
| Staged reward: formation term off until built (§IV-F) | `w_form` is applied only from NAV on |
| Formation pool, one policy (§IV-D) | `formation: dynamic` samples cube / sphere / pyramid / plane per episode; also line, column, v, circle, or `custom` offsets |
| Procrustes formation error (§IV-D) | batched Kabsch in `core.procrustes_error`, normalised by the max squared slot distance |
| Adaptive radius-based neighbours, max k, zero padding (§IV-C) | `neighbour_radius`, `max_neighbours`, plus a validity flag per slot (removes the padding/collision ambiguity of the old code) |
| Receding-horizon waypoints (§IV-E) | `waypoint_mode: discrete`, 1 m spacing; next waypoint when the swarm centroid is within 0.6 m. `carrot` = continuously moving reference |
| (new) Route around static obstacles | `waypoint_planner: dp` (default): the waypoints follow a route planned around the known pillars by a vectorised dynamic program, inflated by the formation's half-width. `straight` = the paper's straight line. Moving obstacles are always handled by the policy. See the results below for why this matters |
| FC-LSTM-FC actor + centralised critic, CTDE (§IV-B) | `formation_nav/mappo_lstm.py`, with proper BPTT (see below) |
| Rewards: nav, formation, avoidance, tilt, smoothness, reaching (§IV-A) | the same terms, plus obstacle clearance, a hold bonus and a crash penalty |
| 4-D action [direction, speed] → velocity → low-level controller (§III-D) | `action_mode: dir_speed` → Lee geometric controller (`controller.py`, OmniDrones' maths) at 62.5 Hz, with feed-forward of the simulated downwash |
| Crazyflie 2.1 as the target platform (§V, Table IV) | `drone_model: Crazyflie` (OmniDrones' asset); controller gains added for it |
| Scale to larger swarms with the same policy (§V-A) | The actor input is independent of the swarm size. `evaluate.py task.num_drones=16` loads the actor only |
| Static and dynamic obstacles (§VI future work) | pillars and moving spheres, observed as the 4 closest obstacles within 4 m (vector to the closest surface point, distance, relative velocity) |
| Hold position at the target | HOLD phase until the episode ends, with a hold bonus (in the old code the episode ended on arrival) |

Not included: the communication-cost reward / optimal neighbour selection (§IV-C.1), on-the-fly
shape changes, and INT8 quantisation.

**Observation (per drone, 128-D with the defaults):** velocity, heading, up vector, angular
velocity, altitude, phase one-hot, vector to its own slot (clipped to 3 m), 7 neighbour slots
× (relative position, relative velocity, distance, desired relative offset, valid) and
4 obstacle slots × (vector to the closest surface point, distance, relative velocity, valid).
There is no absolute x/y position, so routes of any length stay in the training distribution
(a unit test checks translation invariance).

**Critic state:** every drone's position relative to the formation reference, its velocity
and slot error, every obstacle slot, the phase and the vector to the goal.

## Drone model and low-level control

**Crazyflie by default.** OmniDrones ships the Crazyflie model (`cf2x_pybullet.usd` +
`crazyflie.yaml`) but controller gains only for the Firefly, Hummingbird and Neo-11. FormationNav
therefore brings its own velocity controller (`formation_nav/controller.py`):

* **The same Lee geometric controller maths as OmniDrones.** A test checks that its rotor
  commands match OmniDrones' `LeePositionController` for all three shipped drones.
* **Crazyflie gains** are the Hummingbird's scaled by the inertia ratio, so the attitude loop
  closes the same way: ω_n = 10 rad/s, ζ = 0.71. On the rigid-body model a 1.5 m/s velocity step
  settles in 0.8 s with no overshoot and at most 17.5° of tilt. The drone recovers from 30° tilt
  combined with a 3 rad/s spin.
* **The simulated mass is used:** 0.0274 kg for the base plus the rotor links, read from the
  articulation. The yaml says 0.028 kg, and that 2 % error gives a steady 0.1 m/s climb at zero
  command.

**Downwash.** In every environment with more than one drone, OmniDrones pushes each drone along
the thrust of every drone above it, by `exp(-0.5 (2r/z)^2) / (1 + 0.3 z)^2` times that thrust.
The model does not scale with drone size: 1 m below a hovering drone the push is 59 % of its
weight, and the lateral spread is ±z/2. FormationNav handles this in three ways:

1. **Feed-forward (`controller.downwash_feedforward`):** the controller cancels the downwash
   force OmniDrones applied in the previous physics step. Without it, the velocity loop cannot
   hold a drone under another one, and 3-D formations are impossible.
2. **Staged take-off (`staged_takeoff`, `takeoff_speed`):** the formation rises as a block at
   1 m/s. The top layer lifts off first, and each lower layer joins when the block has risen
   past it. No drone ever climbs beneath a drone that is still climbing. Such a drone can take
   up to 1.7 times its own weight in downwash, and a Crazyflie's thrust is only 2.3 times its
   weight.
3. **Crossing-free slot assignment:** drones in the same layer take off together, so their
   paths must not cross.

Scripted slot-seeking policy (`scripts/smoke_test.py` logic), 8 Crazyflies, 64 episodes each,
pure-PyTorch model of OmniDrones' physics:

| episodes that formed (crashes) | cube | sphere | pyramid | plane |
|---|---|---|---|---|
| no feed-forward, simultaneous take-off (the Isaac env as first committed) | 0 % (64) | 0 % (64) | 0 % (64) | 100 % |
| feed-forward only | 94 % (4 collisions) | 92 % (5 pinned to the ground) | 100 % | 100 % |
| feed-forward + staged take-off + crossing-free assignment (**default**) | **100 %** (0) | **100 %** (0) | **100 %** (0) | **100 %** (0) |

Line, column, v and circle also form in 100 % of episodes, and so does the Hummingbird preset.
The Hummingbird version as first committed failed the same way: its thrust/weight of 3.4 does
not help, because the velocity loop, not the thrust, runs out (0 % of cube / sphere / pyramid
episodes formed). The drones sank to the ground.

**Limit: tall formations of many Crazyflies.** Under OmniDrones' downwash, the bottom drones
of a 16-drone cube, pyramid or sphere at 1 m spacing need 3.05, 2.45 and 2.30 times their
weight in thrust just to hover. The Crazyflie has 2.30, so these formations are not flyable.
For 16+ Crazyflies, use `task.downwash_scale=0.5` (then every formation forms), planar
formations, or the Hummingbird (thrust/weight 3.4).

| task setting | default | meaning |
|---|---|---|
| `drone_model.name` | `Crazyflie` | any OmniDrones multirotor with gains in `controller.LEE_GAINS` (crazyflie, hummingbird, firefly, neo11) or `controller.gains` |
| `controller.gains` | `null` | `null` = built-in gains for the drone |
| `controller.mass` | `null` | `null` = simulated total mass |
| `controller.downwash_feedforward` | `true` | cancel OmniDrones' downwash force |
| `controller.integral_gain`, `integral_limit` | 0 | optional velocity integral (e.g. `[0,0,1]`, `[0,0,2]`), for mass randomisation |
| `downwash_scale` | `1.0` | scales OmniDrones' downwash model (0 = off) |
| `staged_takeoff`, `takeoff_speed` | `true`, 1.0 m/s | see above |
| `task=FormationNavHummingbird` | | Hummingbird with its sizes (1.2 m spacing, 0.25 m radius, ...) |

Sizes for the Crazyflie: 1.0 m formation and take-off spacing, drone radius 0.07 m,
collision distance 0.15 m, avoidance starts at 0.4 m. The spawn height is 0.05 m and the
viewer is 4 m from the swarm.

## Layout

```
formation_nav/
  core.py            task logic in pure PyTorch: formations, assignment, staged take-off,
                     obstacles, waypoints/phases, rewards, observations, stats (no Isaac imports)
  env.py             OmniDrones IsaacEnv "FormationNav" (spawns drones + obstacle prims,
                     velocity control with downwash feed-forward, calls core)
  controller.py      Lee geometric velocity controller (OmniDrones' maths) + gains per drone
  quadrotor.py       pure-PyTorch rigid-body multirotor with OmniDrones' rotor / downwash models
  assets/            OmniDrones' Crazyflie and Hummingbird parameter files (MIT)
  mappo_lstm.py      FC-LSTM-FC MAPPO (TorchRL), registered as algo "mappo_lstm"
  pointmass_env.py   the task without Isaac Sim: point-mass or quadrotor dynamics
  scripted.py        scripted slot-seeking policy for smoke tests
  evaluation.py      per-scenario evaluation, results table, trajectory plots
cfg/                 Hydra configs: task/FormationNav.yaml (Crazyflie), task/FormationNavHummingbird.yaml,
                     algo/mappo_lstm.yaml, train / eval / smoke
scripts/
  train.py           Isaac Sim training (OmniDrones loop + scenario evaluation + videos)
  evaluate.py        Isaac Sim evaluation: none/static/dynamic/mixed → json/md tables, plots, mp4
  smoke_test.py      first-flight check with the scripted policy (Isaac Sim, or lite=true)
  train_lite.py      training without Isaac Sim (--dynamics pointmass|quadrotor)
  eval_lite.py       evaluation without Isaac Sim
docs/INSTALL.md      how to install Isaac Sim 4.1 + OmniDrones (and what to do on Blackwell GPUs)
tests/               103 unit / integration tests (run without Isaac Sim)
```

## Install

Step by step, with troubleshooting: [docs/INSTALL.md](docs/INSTALL.md). In short:

1. Use a supported GPU (RTX 30/40-class; **not** Blackwell).
2. Get Isaac Sim 4.1.0. It is no longer on NVIDIA's download page; use the NGC container
   `nvcr.io/nvidia/isaac-sim:4.1.0` (recommended) or the pip wheels `isaacsim==4.1.0.0`.
3. `pip install -e .` in OmniDrones (`main`, developed against commit `9ce7c20`) with Isaac Sim's
   Python. Isaac Lab is **not** needed, but OmniDrones needs a two-line patch to import without it
   (docs/INSTALL.md, section 2).
4. `cd omnidrones_formation && pip install -e .`
5. Run `python scripts/smoke_test.py` (first-flight check, below).

## First-flight check (no training needed)

```bash
python scripts/smoke_test.py                 # Isaac Sim, headless; headless=false to watch
python scripts/smoke_test.py lite=true       # the same on the pure-PyTorch quadrotor model
```

A scripted policy flies each drone straight to its slot. It avoids nothing, but the planned route
goes around the pillars. The pure-PyTorch run gives this for 8 Crazyflies, 16 episodes per cell;
the Isaac Sim numbers should be close:

| success rate (crashes) | cube | sphere | pyramid | plane |
|---|---|---|---|---|
| `none` | 100 % (0) | 100 % (0) | 100 % (0) | 100 % (0) |
| `static` | 100 % (0) | 100 % (0) | 81 % (3 pillar hits) | 94 % (1 pillar hit) |

Time to form is 2.8–5.6 s (the sphere is 2.65 m tall). Without obstacles, every drone is
within 0.3 m of its slot for 92–94 % of the hold phase.

## Train (Isaac Sim, NVIDIA RTX GPU)

```bash
cd omnidrones_formation
# default: 8 drones, formation pool, random obstacle scenario per episode, MAPPO-LSTM
python scripts/train.py wandb.mode=online
# no obstacles at all
python scripts/train.py task.scenario=none
# one fixed formation / a custom one (+x = direction of travel, metres)
python scripts/train.py task.formation=pyramid
python scripts/train.py task.formation=custom task.num_drones=4 \
    'task.custom_formation=[[0,0,0],[-1.2,1.2,0],[-1.2,-1.2,0],[-2.4,0,0.8]]'
# the paper's continuous waypoint alternative, or the MLP baseline
python scripts/train.py task.waypoint_mode=carrot
python scripts/train.py task.waypoint_planner=straight    # the paper's straight route
python scripts/train.py task.obstacle_curriculum=true     # start with 1 obstacle, add more as success rises
python scripts/train.py algo=mappo
# warm start from a point-mass pre-trained actor (same observation/action spaces)
python scripts/train.py checkpoint_path=runs/lite/checkpoint.pt
# the Hummingbird instead of the Crazyflie
python scripts/train.py task=FormationNavHummingbird viewer.eye=[-7.,-7.,5.]
```

Everything is written to `output_dir`, by default
`runs/FormationNav_<date>_<time>/` under the directory you launch from:

```
config.yaml                    the full config of the run
checkpoint_<frames>.pt         every save_interval iterations (300 = ~9.8 M frames)
checkpoint_final.pt
eval_<frames>/, eval_final/    results.md / results.json (one column per scenario),
                               trajectory_<scenario>.png and video_<scenario>.mp4
```

Every `eval_interval` iterations it runs all four scenarios and saves the metric table, a
trajectory plot and a video per scenario there (and to wandb unless `wandb.mode=disabled`).
Videos are recorded headless too; the camera follows one environment's swarm. The default of 512 envs × 8 drones with an
LSTM of 256 needs about 1.5 GB of GPU memory for the rollout buffer (hidden states are stored
per step), on top of Isaac Sim; reduce `task.env.num_envs` if needed.

## Evaluate: behaviour with vs. without obstacles

```bash
python scripts/evaluate.py checkpoint_path=/path/to/checkpoint_final.pt
python scripts/evaluate.py checkpoint_path=... headless=false eval_scenarios=[none]   # watch live
python scripts/evaluate.py checkpoint_path=... 'eval_formations=[cube,sphere,pyramid,plane]'
python scripts/evaluate.py checkpoint_path=... task.num_drones=16 task.downwash_scale=0.5   # scale-up (see "Limit")
```

The script writes `eval_results/results_<formation>.md|json` with one column per scenario:
success rate, hold ratio, formation error, slot error, route progress, drone-drone /
drone-obstacle collisions, minimum clearance and separation, action smoothness, time to form,
time to goal and crash rate. It also writes `trajectory_*.png` (top view + altitude) and
`video_*.mp4`. Set `task.terminate_on_collision=false` to count collisions instead of ending
the episode.

## Without Isaac Sim

Same task, observation, action and reward, with two dynamics backends:

* `--dynamics pointmass` (default): first-order velocity tracking. Fast; use it to debug the
  reward, check learnability, or pre-train.
* `--dynamics quadrotor`: the rigid-body Crazyflie (or `--drone hummingbird`) as OmniDrones
  simulates it: rotor lag, thrust and moment model, the asset's mass and arm length, and the
  downwash. It is flown by the same controller and feed-forward as the Isaac env, at the same
  62.5 Hz. It runs at about half the speed of the point mass (5.6k vs 10k env-steps/s for
  128 envs × 8 drones on 2 CPU threads).

```bash
python scripts/train_lite.py --num_envs 128 --num_drones 8 --iters 400 --out runs/lite
python scripts/train_lite.py --dynamics quadrotor --drone crazyflie --out runs/lite_cf
python scripts/eval_lite.py --checkpoint runs/lite/checkpoint.pt --num_drones 8
pytest tests -q            # 103 tests, about 40 s on CPU (4 more need the OmniDrones source: OMNIDRONES_DIR=...)
```

The point-mass results in [docs/POINTMASS_RESULTS.md](docs/POINTMASS_RESULTS.md) were obtained
before the switch to the Crazyflie. They used the Hummingbird sizes, simultaneous take-off and
the plain greedy assignment.

## Correctness notes

* **Recurrent PPO:** rollouts store the LSTM state per step. Updates unroll the LSTM over
  16-step chunks from the stored state and reset at `is_init`, so the PPO ratio is exactly 1
  before the first gradient step (tested). This was the main bug in the old code.
* **Truncation:** each step bootstraps from the critic value of its own next observation.
  Only `terminated` (crash) cuts the bootstrap, so the 800-step time limit is not treated as
  failure.
* **Shared weights:** TorchRL's collector deep-copies the policy but shares parameter storage.
  A test checks that the collector acts with the updated weights after `train_op`.
* **Input normalisation:** per-feature running mean/std, updated after each PPO update
  (`algo.input_norm: running`). LayerNorm across the raw observation, as in the old code, makes
  a distant obstacle rescale a drone's own velocity features. It is still available as
  `layernorm`. The learning rate decays linearly over the run.

## What has and has not been verified

Verified in a CPU-only container, without Isaac Sim:

* **Core logic:** unit tests for formation geometry, the crossing-free assignment, the staged
  take-off, Kabsch, obstacle geometry and motion, the phase machine, termination and
  observation invariances; also for the MAPPO algorithm.
* **Against OmniDrones' own source:** the controller's rotor commands equal OmniDrones'
  `LeePositionController`, and the downwash model equals `MultirotorBase.downwash`.
* **Flight on the rigid-body model:** Crazyflie hover, velocity steps, take-off and tilt/spin
  recovery; 3-D formations holding under the downwash only with the feed-forward; the scripted
  first-flight check above.
* **Wiring:** `env.py` runs against fakes of the Isaac/OmniDrones classes whose drone is the
  rigid-body model. This covers the Hydra configs (both drone presets), the simulated mass,
  the feed-forward path and a short train + evaluate.
* **Learning on the point-mass surrogate** ([docs/POINTMASS_RESULTS.md](docs/POINTMASS_RESULTS.md)),
  with the pre-Crazyflie defaults:
  * without obstacles, 88–97 % of episodes take off, form, navigate and hold;
  * with straight waypoints, static pillars stay unsolved (0–3 %);
  * the planned route raises static success to 50 %.
* **Learning with the Crazyflie defaults, no obstacles, 1.6 M steps:**
  * point mass: 100 % success, hold ratio 0.75;
  * rigid-body Crazyflie: 91 % reach the goal, but the policy does not yet stop precisely
    enough to hold (hold ratio ≈ 0). It needs longer training (same document).

**Not yet run in Isaac Sim itself** (no GPU in the development container). Things to check on
the first run (`scripts/smoke_test.py`):

* the start-up line reports a simulated mass of 0.0274 kg for the Crazyflie;
* the smoke-test table is close to the `lite=true` one above. The pure-PyTorch model already
  includes the asset's mass and 0.0396 m rotor arm (the controller assumes the yaml's 0.043 m,
  about 8 % less roll/pitch authority). It does not model PhysX ground contact or the
  articulated rotor links;
* the obstacle prims move and render as expected.
