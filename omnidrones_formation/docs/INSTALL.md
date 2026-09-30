# Installing OmniDrones and FormationNav

OmniDrones runs on **Isaac Sim 4.1** with **Python 3.10**, **PyTorch 2.2.2 (CUDA 11.8)**,
**torchrl 0.3.1** and **tensordict 0.3.2**. FormationNav was developed against OmniDrones
`main` at commit `9ce7c20`.

## 0. Check the GPU first

| GPU | OmniDrones (Isaac Sim 4.1) |
|---|---|
| RTX 30xx / 40xx, RTX A-series / Ada workstation, A10G, L4, L40(S) | supported |
| **Blackwell**: RTX 50xx, **RTX PRO Blackwell** (e.g. the RTX PRO 4000 Blackwell listed in the paper) | **not supported** |
| GTX / no RTX | not supported (Isaac Sim needs RTX) |

Blackwell GPUs need Isaac Sim 5.1 or later (Linux driver 580.65.06 or later) and PyTorch 2.7 or
later with CUDA 12.8. OmniDrones pins Isaac Sim 4.1 and PyTorch 2.2.2, and it has no Isaac Sim 5
version. On a Blackwell machine you can:

1. run OmniDrones on another machine with a supported GPU (a local RTX 30/40 card, or a cloud
   instance with an A10G, L4 or L40S);
2. still use everything that does not need Isaac Sim on the Blackwell machine (section 6): the
   tests, the CPU quadrotor model and lite training;
3. port FormationNav to Isaac Lab 2.x on Isaac Sim 5.1, which supports Blackwell and ships a
   Crazyflie asset. `formation_nav/core.py` (the task) and `mappo_lstm.py` do not depend on the
   simulator; only `env.py` (about 350 lines) would need rewriting.

## 1. Requirements for Isaac Sim 4.1

* Ubuntu 22.04 or 20.04, x86_64. OmniDrones does not support Windows.
* An NVIDIA RTX GPU (see above) with 8 GB of VRAM or more. The default 512 environments × 8
  drones need more; lower `task.env.num_envs` on smaller cards.
* NVIDIA driver 535 or later. Check it with `nvidia-smi`.
* 32 GB RAM and about 50 GB of free disk.
* Miniconda or Anaconda.

## 2. Install Isaac Sim 4.1.0

OmniDrones' docs assume the Omniverse Launcher, which NVIDIA retired on 1 October 2025. Use the
standalone package instead:

1. In the Isaac Sim documentation, open **Download Isaac Sim → Download Archive** and get
   **Isaac Sim 4.1.0 for Linux** (a zip of about 7.8 GB).
2. Unpack it and run the post-install step:
   ```bash
   mkdir -p ~/isaacsim/isaac-sim-4.1.0 && cd ~/isaacsim/isaac-sim-4.1.0
   unzip ~/Downloads/<the downloaded isaac-sim 4.1.0 zip>
   ./post_install.sh                  # if present in the package
   ./isaac-sim.selector.sh            # optional: start the app once; the first start compiles shaders (several minutes)
   ```
3. Point OmniDrones to it (add the line to `~/.bashrc`, then run `source ~/.bashrc`):
   ```bash
   export ISAACSIM_PATH="$HOME/isaacsim/isaac-sim-4.1.0"
   ```
4. Check: `$ISAACSIM_PATH/python.sh -c "from isaacsim import SimulationApp; print('ok')"`

> **Alternative (not tested with OmniDrones): pip wheels**, Ubuntu 22.04 only (needs GLIBC 2.34 or later):
> `pip install isaacsim==4.1.0.0 isaacsim-extscache-physics==4.1.0.0 isaacsim-extscache-kit==4.1.0.0 isaacsim-extscache-kit-sdk==4.1.0.0 --extra-index-url https://pypi.nvidia.com`
> in the Python 3.10 environment of step 3, instead of steps 1–3 above and the `conda_setup` copy.
> OmniDrones' `init_simulation_app` reads `$EXP_PATH/omni.isaac.sim.python.kit`, which the
> binary package's setup script sets. With pip you must point `EXP_PATH` yourself at the `apps`
> folder of the installed `isaacsim` package.

## 3. Python environment and OmniDrones

```bash
conda create -n sim python=3.10 -y
conda activate sim

git clone https://github.com/btx0424/OmniDrones.git
cd OmniDrones
git checkout 9ce7c20            # optional: the commit FormationNav was developed against

# hook Isaac Sim into the conda env: activating it now sources $ISAACSIM_PATH/setup_conda_env.sh
cp -r conda_setup/etc $CONDA_PREFIX
conda deactivate && conda activate sim

python -c "from isaacsim import SimulationApp"                  # must not fail
python -c "import torch; print(torch.__version__, torch.__path__)"   # 2.2.2, from Isaac Sim's bundled packages

pip install -e .                # OmniDrones (+ hydra, wandb, torchrl 0.3.1, ...)
```

Do not `pip install torch` into this environment. PyTorch comes with Isaac Sim, and a second
copy breaks it.

**Isaac Lab is not needed.** OmniDrones' guide installs it next, but only its `Forest` and
`Pinball` tasks use it; OmniDrones prints a notice and continues without it. FormationNav senses
obstacles analytically.

Check OmniDrones on its own (`wandb.mode=disabled` avoids needing a wandb account):

```bash
cd scripts
python train.py algo=ppo headless=true wandb.mode=disabled total_frames=100000
```

## 4. FormationNav

```bash
cd /path/to/align/omnidrones_formation
pip install -e .

python -m pytest tests -q                              # CPU only, no Isaac Sim needed
python scripts/smoke_test.py lite=true                 # expected numbers, pure-PyTorch quadrotor model
python scripts/smoke_test.py                           # the same check in Isaac Sim (headless)
python scripts/smoke_test.py headless=false task.env.num_envs=4   # watch it
```

The smoke test flies a scripted policy (each drone goes straight to its slot) and needs no
training. It prints the masses the controller uses:
`crazyflie: mass in the parameter yaml 0.0280 kg, simulated 0.0274 kg`.

In `none` every formation should reach `success` ≈ 1 with no crashes, as in the `lite=true`
run. If the Isaac Sim numbers are much worse, open the plots in `smoke_results/<formation>/`
and see the troubleshooting section below.

Then train:

```bash
python scripts/train.py wandb.mode=disabled            # or set wandb.entity=... and keep it online
python scripts/train.py task=FormationNavHummingbird viewer.eye=[-7.,-7.,5.]   # the larger drone
```

## 5. Troubleshooting

* **`TypeError: ArticulationView.get_world_poses() got an unexpected keyword argument 'usd'`**,
  raised from `$ISAACSIM_PATH/exts/omni.isaac.core/omni/isaac/core/prims/xform_prim_view.py`:
  this is a known Isaac Sim 4.x issue (OmniDrones' troubleshooting page). At line 189 of that
  file, change `self.get_world_poses(usd=usd)` to `self.get_world_poses()`.
* **The first start hangs at "Waiting for compilation of ray tracing shaders"**: this is a
  one-time shader compilation of several minutes. If it happens on every start, see OmniDrones'
  troubleshooting page.
* **Slow start with `libcurl error ... localhost:8891`**: Isaac Sim is trying to reach a
  Nucleus/asset server. FormationNav needs none: the drone assets ship with OmniDrones and the
  ground plane and obstacles are created locally. The troubleshooting page shows how to skip the
  wait.
* **`KeyError: 'EXP_PATH'`**: the conda hook did not run. Check `echo $ISAACSIM_PATH`, check that
  `$CONDA_PREFIX/etc/conda/activate.d/env_vars.sh` exists, and re-activate the environment.
* **CUDA / `no kernel image is available` errors**: the GPU is too new (Blackwell, see section 0)
  or the driver is older than 535.
* **The lower drones of cube / sphere / pyramid formations sink**: check that
  `task.controller.downwash_feedforward=true` (the default) and read the controller line printed
  at start-up. For 16 or more Crazyflies in 3-D formations, use `task.downwash_scale=0.5` or the
  Hummingbird (see the README, "Drone model").

## 6. Without Isaac Sim: lite version, any machine (CPU is enough)

The tests, the pure-PyTorch quadrotor model and the lite training need only PyTorch. They were
tested with PyTorch 2.2.2, torchrl 0.3.1, tensordict 0.3.2 and Hydra 1.3:

```bash
python3.10 -m venv ~/venv-fn && source ~/venv-fn/bin/activate     # 3.10 or 3.11
pip install torch==2.2.2 torchrl==0.3.1 tensordict==0.3.2 hydra-core matplotlib pyyaml pytest
cd /path/to/align/omnidrones_formation && pip install -e .
python -m pytest tests -q
python scripts/smoke_test.py lite=true
python scripts/train_lite.py --dynamics quadrotor --drone crazyflie --out runs/lite_cf
```

PyTorch 2.2.2 has no Blackwell kernels, so on a Blackwell machine run these on the CPU
(`--device cpu`).
