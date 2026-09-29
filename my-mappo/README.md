# Think Locally for Global Harmony: A Distributed Framework for Synchrony in UAV Swarm

This repository contains a MAPPO-style multi-agent reinforcement learning setup for drone swarm control in PyBullet, with an MA-LSTM policy and centralized critic.

The project focuses on three simultaneous objectives:

- maintain a desired formation
- navigate toward per-drone targets
- avoid collisions and unstable flight

For the full training call flow and detailed metrics definitions, see [docs/Documentation.md](docs/Documentation.md).

## What is in this repo

- `onpolicy/`: core algorithms, runner, models, environment wrappers, utilities
- `onpolicy/scripts/train/train_pybullet_drones.py`: main training entrypoint
- `docs/Documentation.md`: detailed training workflow and implementation notes

## Quick Start

### 1. Create and activate a Python environment

Windows (PowerShell):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Linux/macOS:

```bash
python -m venv .venv
source .venv/bin/activate
```

### 2. Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

Alternative: use `environment.yaml` if you prefer conda-based setup.

## Training

Run training from repo root:

```bash
python onpolicy/scripts/train/train_pybullet_drones.py \
	--num_drones 8 \
	--num_env_steps 8000000 \
	--n_rollout_threads 8 \
	--n_training_threads 6 \
	--num_mini_batch 2 \
	--save_interval 25 \
	--experiment_name "malstm_gpu_8M" \
	--formation_type "dynamic" \
	--neighbour_radius 1.0 \
	--min_dynamic_neighbours 1 \
	--max_dynamic_neighbours 7
```

### Common training arguments

- `--num_drones`: number of agents (drones)
- `--num_env_steps`: total environment interaction steps
- `--n_rollout_threads`: number of parallel env workers
- `--n_training_threads`: PyTorch training threads
- `--formation_type`: one of `polygon`, `line`, `plane`, `cube`, `sphere`, `pyramid`, `dynamic`
- `--neighbour_radius`: dynamic-neighbor radius in meters (`0` disables dynamic neighbors)
- `--min_dynamic_neighbours`: minimum filled neighbor slots
- `--max_dynamic_neighbours`: maximum neighbor slots / fixed observation capacity
- `--episode_len_sec`: environment episode duration in seconds

### Resume options

Full-state resume (weights + optimizers + RNG + progress):

```bash
python onpolicy/scripts/train/train_pybullet_drones.py \
	--resume_checkpoint onpolicy/scripts/results/pybullet-drones/drones_8/rmappo/exp_name/run1/models/resume_checkpoint.pt
```

Weights-only resume (fresh optimizer/progress):

```bash
python onpolicy/scripts/train/train_pybullet_drones.py \
	--resume_partial_checkpoint onpolicy/scripts/results/pybullet-drones/drones_8/rmappo/exp_name/run1/models
```


## Outputs and artifacts

Training outputs are saved under:

```text
onpolicy/scripts/results/pybullet-drones/drones_{num_drones}/rmappo/{experiment_name}/run{N}/
```

Typical contents:

- `models/actor.pt`, `models/critic.pt`
- `models/resume_checkpoint.pt`
- `plots/training_summary.png`
- `plots/reward_components.png`
- `plots/penalty_bonus_components.png`
- `plots/distance_distribution.png`
- `plots/last20_metrics.json`