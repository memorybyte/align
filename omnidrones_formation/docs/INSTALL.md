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

* Ubuntu 22.04 (20.04 works for the container route only). OmniDrones does not support Windows.
* An NVIDIA RTX GPU (see above) with 8 GB of VRAM or more. The default 512 environments × 8
  drones need more; lower `task.env.num_envs` on smaller cards.
* NVIDIA driver 535 or later. Check it with `nvidia-smi`.
* 32 GB RAM and about 50 GB of free disk.

## 2. Get Isaac Sim 4.1.0

Isaac Sim 4.1.0 is **no longer on the download page**. OmniDrones' docs assume the Omniverse
Launcher, which NVIDIA retired on 1 October 2025. NVIDIA's current advice for old releases is
to use either the **NGC container** or the **pip wheels**:

| Route | Pros | Cons |
|---|---|---|
| **A. Docker container `nvcr.io/nvidia/isaac-sim:4.1.0`** (recommended) | the exact 4.1.0 build, all files OmniDrones expects, works on Ubuntu 20.04 and 22.04 | needs Docker + NVIDIA Container Toolkit + a free NGC account; headless (training, videos and the smoke test all work headless) |
| **B. pip wheels `isaacsim==4.1.0.0`** | no Docker, the GUI works (`headless=false`) | Ubuntu 22.04 only (GLIBC ≥ 2.34), one extra environment variable for OmniDrones |

### Route A: container (recommended)

One-time setup:

1. Install [Docker](https://docs.docker.com/engine/install/ubuntu/) and the
   [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).
   Check it with `docker run --rm --gpus all ubuntu nvidia-smi`.
2. Create a free account at [ngc.nvidia.com](https://ngc.nvidia.com), then open
   **Setup → Generate API Key**.
3. Pull the image and clone the code on the host:

```bash
docker login nvcr.io                  # username: $oauthtoken   password: <your NGC API key>
docker pull nvcr.io/nvidia/isaac-sim:4.1.0

git clone https://github.com/btx0424/OmniDrones.git ~/OmniDrones
cd ~/OmniDrones && git checkout 9ce7c20 && cd -     # optional: the commit FormationNav was developed against
# ~/align = this repository

mkdir -p ~/docker/isaac-sim/{cache/kit,cache/ov,cache/pip,cache/glcache,cache/computecache,logs,data}
docker run --name omnidrones --entrypoint bash -it --gpus all --network=host \
  -e "ACCEPT_EULA=Y" -e "PRIVACY_CONSENT=Y" \
  -v ~/OmniDrones:/workspace/OmniDrones:rw -v ~/align:/workspace/align:rw \
  -v ~/docker/isaac-sim/cache/kit:/isaac-sim/kit/cache:rw \
  -v ~/docker/isaac-sim/cache/ov:/root/.cache/ov:rw \
  -v ~/docker/isaac-sim/cache/pip:/root/.cache/pip:rw \
  -v ~/docker/isaac-sim/cache/glcache:/root/.cache/nvidia/GLCache:rw \
  -v ~/docker/isaac-sim/cache/computecache:/root/.nv/ComputeCache:rw \
  -v ~/docker/isaac-sim/logs:/root/.nvidia-omniverse/logs:rw \
  -v ~/docker/isaac-sim/data:/root/.local/share/ov/data:rw \
  nvcr.io/nvidia/isaac-sim:4.1.0
```

Inside the container, Isaac Sim lives in `/isaac-sim`. Use its bundled Python (`/isaac-sim/python.sh`)
for everything. It already has Python 3.10 and PyTorch 2.2.2, and it sets `EXP_PATH`, which
OmniDrones needs:

```bash
# known Isaac Sim 4.x issue with OmniDrones (see Troubleshooting)
sed -i 's/self.get_world_poses(usd=usd)/self.get_world_poses()/' \
    /isaac-sim/exts/omni.isaac.core/omni/isaac/core/prims/xform_prim_view.py

alias pysim=/isaac-sim/python.sh
cd /workspace/OmniDrones && pysim -m pip install -e .
pysim -m pip install tensordict==0.3.2   # keep the version torchrl 0.3.1 was built for
cd /workspace/align/omnidrones_formation && pysim -m pip install -e . pytest
pysim -m pytest tests -q
pysim scripts/smoke_test.py                 # first flight in Isaac Sim (headless)
```

Afterwards, `docker start -ai omnidrones` reopens the same container with everything installed.
In the rest of this guide, read `python` as `/isaac-sim/python.sh` when you are inside the container.

### Route B: pip wheels (Ubuntu 22.04)

```bash
conda create -n sim python=3.10 -y && conda activate sim
pip install --upgrade pip
pip install torch==2.2.2 --index-url https://download.pytorch.org/whl/cu118
pip install isaacsim==4.1.0.0 isaacsim-extscache-physics==4.1.0.0 \
    isaacsim-extscache-kit==4.1.0.0 isaacsim-extscache-kit-sdk==4.1.0.0 \
    --extra-index-url https://pypi.nvidia.com
# if pip finds no 4.1.0.0, list what exists: pip index versions isaacsim --extra-index-url https://pypi.nvidia.com

# OmniDrones reads $EXP_PATH/omni.isaac.sim.python.kit; the binary package's scripts set it, pip does not
SITE=$(python -c "import importlib.util, os; print(os.path.dirname(importlib.util.find_spec('isaacsim').origin))")
export EXP_PATH="$SITE/apps"                 # add both exports to ~/.bashrc or $CONDA_PREFIX/etc/conda/activate.d/
export OMNI_KIT_ACCEPT_EULA=YES
ls "$EXP_PATH/omni.isaac.sim.python.kit" || find "$SITE" -name omni.isaac.sim.python.kit   # adjust EXP_PATH if needed

# same Isaac Sim 4.x fix as in route A
sed -i 's/self.get_world_poses(usd=usd)/self.get_world_poses()/' \
    $(find "$SITE" -path "*omni/isaac/core/prims/xform_prim_view.py")

python -c "from isaacsim import SimulationApp; print('ok')"
git clone https://github.com/btx0424/OmniDrones.git && cd OmniDrones
git checkout 9ce7c20        # optional
pip install -e .            # do NOT copy conda_setup/etc: that hook is for the binary package
pip install tensordict==0.3.2   # torchrl 0.3.1 does not cap tensordict; newer versions break it (MemmapTensor ImportError)
python -c "import torch, torchrl, tensordict; print(torch.__version__, torchrl.__version__, tensordict.__version__)"   # 2.2.2+cu118 0.3.1 0.3.2
```

The first start of Isaac Sim, by either route, compiles shaders and downloads extensions, which
can take 5–10 minutes. Later starts are much faster.

> I could not run either route here (no GPU, and NVIDIA's sites are blocked from my sandbox).
> The container tag and the pip naming scheme come from NVIDIA's NGC catalog and Isaac Lab's
> install docs. If a command fails, send me the error.

**Not recommended: Isaac Sim 4.2 / 4.5.** They ship PyTorch 2.4 / 2.5, but OmniDrones pins
torchrl 0.3.1, which is built for PyTorch 2.2. Using them would mean upgrading torchrl and
tensordict and fixing OmniDrones (untested).

**Isaac Lab is not needed.** OmniDrones' guide installs it, but only its `Forest` and `Pinball`
tasks use it; OmniDrones prints a notice and continues without it. FormationNav senses obstacles
analytically.

## 3. Check OmniDrones on its own

`wandb.mode=disabled` avoids needing a wandb account:

```bash
cd OmniDrones/scripts
python train.py algo=ppo headless=true wandb.mode=disabled total_frames=100000
```

## 4. FormationNav

```bash
cd align/omnidrones_formation
pip install -e .

python -m pytest tests -q                              # CPU only, no Isaac Sim needed
python scripts/smoke_test.py lite=true                 # expected numbers, pure-PyTorch quadrotor model
python scripts/smoke_test.py                           # the same check in Isaac Sim (headless)
python scripts/smoke_test.py headless=false task.env.num_envs=4   # watch it (route B)
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
  raised from `.../omni/isaac/core/prims/xform_prim_view.py`: this is a known Isaac Sim 4.x issue
  (OmniDrones' troubleshooting page). The `sed` line in section 2 fixes it. By hand, change
  `self.get_world_poses(usd=usd)` to `self.get_world_poses()` at line 189.
* **The first start hangs at "Waiting for compilation of ray tracing shaders"**: this is a
  one-time shader compilation of several minutes. If it happens on every start, see OmniDrones'
  troubleshooting page.
* **Slow start with `libcurl error ... localhost:8891`**: Isaac Sim is trying to reach a
  Nucleus/asset server. FormationNav needs none: the drone assets ship with OmniDrones and the
  ground plane and obstacles are created locally. The troubleshooting page shows how to skip the
  wait.
* **`KeyError: 'EXP_PATH'`**: in the container, you started Python without `/isaac-sim/python.sh`.
  With pip, `EXP_PATH` is not exported (route B).
* **`could not select device driver "" with capabilities: [[gpu]]`** (Docker): install the NVIDIA
  Container Toolkit, then run `sudo nvidia-ctk runtime configure --runtime=docker && sudo systemctl restart docker`.
* **`unauthorized` when pulling from nvcr.io**: log in with the user name `$oauthtoken`, spelled
  exactly like that, and your NGC API key as the password.
* **`ImportError: cannot import name 'MemmapTensor' from 'tensordict.memmap'`**: pip pulled a
  tensordict that is too new for torchrl 0.3.1. Run `pip install tensordict==0.3.2`.
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
