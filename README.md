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

## OmniDrones task: what the CPU experiments showed

Measured with the point-mass version of the same task
([details](omnidrones_formation/docs/POINTMASS_RESULTS.md)); the Isaac Sim version has not
been run yet because there was no GPU here.

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

# OmniDrones task (needs Isaac Sim 4.1 + OmniDrones, see omnidrones_formation/README.md)
cd omnidrones_formation && pip install -e .
python scripts/train.py                                        # obstacles: random per episode
python scripts/evaluate.py checkpoint_path=...                 # none vs static vs dynamic vs mixed

# same task without Isaac Sim (point-mass drones, CPU)
python scripts/train_lite.py --out runs/lite && python scripts/eval_lite.py --checkpoint runs/lite/checkpoint.pt
```
