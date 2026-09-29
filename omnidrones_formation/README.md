# FormationNav for OmniDrones

A UAV swarm **takes off from the ground**, **builds a given formation**, flies it to a
goal **through static and/or moving obstacles (or none)**, and **holds the formation at
the goal**. Training uses the paper's **FC-LSTM-FC MAPPO** (shared decentralised actor,
centralised critic).

```
 FORM (take off + build formation) ──► NAV (waypoints, avoid obstacles) ──► HOLD (at goal)
```

| Scenario | Obstacles |
|---|---|
| `none` | empty route: the baseline behaviour |
| `static` | 3–6 vertical pillars (r = 0.3–0.6 m) in a 5 m corridor along the route |
| `dynamic` | 1–3 spheres crossing the route back and forth (0.3–1.0 m/s) |
| `mixed` | both |
| `train_mix` | a random scenario per episode (default for training) |

## How it maps to the paper

| Paper | Here |
|---|---|
| Start on the ground, build the formation first (§III-A, §IV-F) | FORM phase: planar take-off grid, greedy drone→slot assignment, NAV starts after every drone has been within 0.35 m of its slot for 15 steps |
| Staged reward: formation term off until built (§IV-F) | `w_form` is applied only from NAV on |
| Formation pool, one policy (§IV-D) | `formation: dynamic` samples cube / sphere / pyramid / plane per episode; also line, column, v, circle, or `custom` offsets |
| Procrustes formation error (§IV-D) | batched Kabsch in `core.procrustes_error`, normalised by the max squared slot distance |
| Adaptive radius-based neighbours, max k, zero padding (§IV-C) | `neighbour_radius`, `max_neighbours`, plus a validity flag per slot (removes the padding/collision ambiguity of the old code) |
| Receding-horizon waypoints (§IV-E) | `waypoint_mode: discrete`, 1 m spacing; next waypoint when the swarm centroid is within 0.6 m. `carrot` = continuously moving reference |
| FC-LSTM-FC actor + centralised critic, CTDE (§IV-B) | `formation_nav/mappo_lstm.py`, with proper BPTT (see below) |
| Rewards: nav, formation, avoidance, tilt, smoothness, reaching (§IV-A) | the same terms, plus obstacle clearance, a hold bonus and a crash penalty |
| 4-D action [direction, speed] → velocity → low-level controller (§III-D) | `action_mode: dir_speed` → Lee position controller (velocity tracking) at 62.5 Hz |
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

## Layout

```
formation_nav/
  core.py            task logic in pure PyTorch: formations, assignment, obstacles,
                     waypoints/phases, rewards, observations, stats   (no Isaac imports)
  env.py             OmniDrones IsaacEnv "FormationNav" (spawns drones + obstacle prims,
                     Lee velocity control, calls core)
  mappo_lstm.py      FC-LSTM-FC MAPPO (TorchRL), registered as algo "mappo_lstm"
  pointmass_env.py   same task with point-mass drones (CPU, no Isaac Sim)
  evaluation.py      per-scenario evaluation, results table, trajectory plots
cfg/                 Hydra configs (task/FormationNav.yaml, algo/mappo_lstm.yaml, train/eval)
scripts/
  train.py           Isaac Sim training (OmniDrones loop + scenario evaluation + videos)
  evaluate.py        Isaac Sim evaluation: none/static/dynamic/mixed → json/md tables, plots, mp4
  train_lite.py      point-mass training on CPU/GPU
  eval_lite.py       point-mass evaluation
tests/               62 unit / integration tests (run without Isaac Sim)
```

## Install

1. Install Isaac Sim 4.1 and OmniDrones (`main` branch; developed against commit
   `9ce7c20`) following the
   [OmniDrones docs](https://omnidrones.readthedocs.io/en/latest/). Isaac Lab is **not**
   needed; obstacles are sensed analytically, not with the ray-caster.
2. In the same Python environment:
   ```bash
   cd omnidrones_formation
   pip install -e .
   ```

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
python scripts/train.py algo=mappo
# warm start from a point-mass pre-trained actor (same observation/action spaces)
python scripts/train.py checkpoint_path=runs/lite/checkpoint.pt
```

Every `eval_interval` iterations it runs all four scenarios and logs the metric table, a
trajectory plot and a video per scenario to wandb. The default of 512 envs × 8 drones with an
LSTM of 256 needs about 1.5 GB of GPU memory for the rollout buffer (hidden states are stored
per step), on top of Isaac Sim; reduce `task.env.num_envs` if needed.

## Evaluate: behaviour with vs. without obstacles

```bash
python scripts/evaluate.py checkpoint_path=/path/to/checkpoint_final.pt
python scripts/evaluate.py checkpoint_path=... headless=false eval_scenarios=[none]   # watch live
python scripts/evaluate.py checkpoint_path=... 'eval_formations=[cube,sphere,pyramid,plane]'
python scripts/evaluate.py checkpoint_path=... task.num_drones=16                      # scale-up
```

The script writes `eval_results/results_<formation>.md|json` with one column per scenario:
success rate, hold ratio, formation error, slot error, route progress, drone-drone /
drone-obstacle collisions, minimum clearance and separation, action smoothness, time to form,
time to goal and crash rate. It also writes `trajectory_*.png` (top view + altitude) and
`video_*.mp4`. Set `task.terminate_on_collision=false` to count collisions instead of ending
the episode.

## Without Isaac Sim (point-mass surrogate)

Same task, observation, action and reward, with first-order point-mass dynamics. Use it to
debug the reward, check learnability, or pre-train.

```bash
python scripts/train_lite.py --num_envs 128 --num_drones 8 --iters 400 --out runs/lite
python scripts/eval_lite.py --checkpoint runs/lite/checkpoint.pt --num_drones 8
pytest tests -q            # 62 tests, about 1 min on CPU
```

## Correctness notes

* **Recurrent PPO:** rollouts store the LSTM state per step. Updates unroll the LSTM over
  16-step chunks from the stored state and reset at `is_init`, so the PPO ratio is exactly 1
  before the first gradient step (tested). This was the main bug in the old code.
* **Truncation:** each step bootstraps from the critic value of its own next observation.
  Only `terminated` (crash) cuts the bootstrap, so the 800-step time limit is not treated as
  failure.
* **Shared weights:** TorchRL's collector deep-copies the policy but shares parameter storage.
  A test checks that the collector acts with the updated weights after `train_op`.

## What has and has not been verified

Verified in a CPU-only container, without Isaac Sim:

* the unit tests for the core logic (formation geometry, assignment, Kabsch, obstacle
  geometry and motion, phase machine, termination, observation invariances) and the algorithm;
* a wiring test that runs `env.py` against fakes of the Isaac/OmniDrones classes, including
  the Hydra configs and a short train + evaluate;
* learning on the point-mass surrogate (see the repository README).

**Not yet run in Isaac Sim itself** (no GPU in the development container). Things to check on
the first run:

* `spawn_height` / `crash_height` against the Hummingbird's resting height;
* the Lee controller's velocity tracking with the chosen `max_speed`;
* that the obstacle prims move and render as expected.
