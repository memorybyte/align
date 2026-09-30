# ALiGn: decentralised MARL formation control for UAV swarms

This repository contains:

| Path | What |
|---|---|
| [`docs/AUDIT.md`](docs/AUDIT.md) | **Paper vs. code audit**: what the previous student implemented, what the draft paper claims but the code does not do, and every bug that was fixed |
| [`my-mappo/`](my-mappo) | The previous student's PyBullet MAPPO code (commit `47f202e` = unchanged import), **with the bugs fixed** in the following commit and regression tests in `my-mappo/tests/` |
| [`omnidrones_formation/`](omnidrones_formation) | **New OmniDrones (Isaac Sim) task**: take off from the ground → build a given formation → fly to the goal through static / dynamic obstacles or none → hold the formation, trained with the paper's FC-LSTM-FC MAPPO |

## Most important findings (details in the audit)

1. **The LSTM policy was never trained correctly.**
   * During rollouts, drones received other drones' hidden states (87.5 % of stored states wrong).
   * During PPO updates, each 10-step chunk reused its first hidden state for every step, with
     no backprop through time.

   Both are fixed and covered by tests. Any existing results need to be regenerated.
2. **Several paper claims are not implemented:**
   * ground take-off / formation construction (drones spawned already in formation);
   * the staged formation reward (and the formation reward weight was 0);
   * waypoint navigation;
   * communication-aware neighbour selection;
   * obstacles;
   * shape change on the fly;
   * evaluation scripts for Tables I–III.
3. **Cube / sphere / pyramid targets were flattened onto the ground** (bottom layer clamped to
   z = 0.05 m). Evaluation and rendering could not run.

## OmniDrones task: Crazyflie, downwash and installation

* **Drone:** the task now flies OmniDrones' **Crazyflie** (the paper's hardware target) by
  default. OmniDrones has no controller for it, so the task brings the same Lee controller
  maths (tested to match OmniDrones' own) with Crazyflie gains and the simulated mass. The
  Hummingbird remains available as `task=FormationNavHummingbird`.
* **Downwash fix:** OmniDrones pushes every drone down with the downwash of the drones above
  it (59 % of a drone's weight at 1 m). The first version of the task did not compensate this,
  so cube / sphere / pyramid formations could not hold: the lower layer sank to the ground in
  every test episode. The controller now cancels the force (feed-forward), the take-off is
  staged top layer first, and slot paths do not cross. A scripted policy then forms all eight
  formation types with 8 Crazyflies, without crashes.
  [Details](omnidrones_formation/README.md#drone-model-and-low-level-control)
* **Install:** [omnidrones_formation/docs/INSTALL.md](omnidrones_formation/docs/INSTALL.md).
  Isaac Sim 4.1, which OmniDrones needs, **does not run on Blackwell GPUs** such as the RTX PRO
  4000 Blackwell listed in the paper; the guide lists the options.

## OmniDrones task: what the CPU experiments showed

Measured with the point-mass version of the same task
([details](omnidrones_formation/docs/POINTMASS_RESULTS.md)), before the Crazyflie switch; the
Isaac Sim version has not been run yet because there was no GPU here.

* **Without obstacles:** 88–97 % of episodes take off, build the 3-D formation, fly 8–12 m and
  hold at the goal.
* **Static pillars with the paper's straight-line waypoints:** 0–3 % success. The straight
  route cuts through a pillar for the formation in 93–98 % of layouts.
* **With a route planned around the pillars** (`waypoint_planner: dp`, now the default):
  50 % static and 28 % mixed success after only about 2 M steps.
* **Moving obstacles:** avoided by the learned policy alone (about 38 % success so far); they
  need Isaac-scale training.

## Quick start

```bash
# fixed PyBullet code (unchanged training command, see my-mappo/README.md)
cd my-mappo && pip install -r requirements.txt && PYTHONPATH=. pytest tests -q

# OmniDrones task (needs Isaac Sim 4.1 + OmniDrones, see omnidrones_formation/docs/INSTALL.md)
cd omnidrones_formation && pip install -e .
python scripts/smoke_test.py                                   # first flight: take-off, form, fly, hold (scripted)
python scripts/train.py                                        # obstacles: random per episode
python scripts/evaluate.py checkpoint_path=...                 # none vs static vs dynamic vs mixed

# same task without Isaac Sim (CPU): point-mass or rigid-body Crazyflie dynamics
python scripts/smoke_test.py lite=true
python scripts/train_lite.py --out runs/lite && python scripts/eval_lite.py --checkpoint runs/lite/checkpoint.pt
python scripts/train_lite.py --dynamics quadrotor --out runs/lite_cf
```
