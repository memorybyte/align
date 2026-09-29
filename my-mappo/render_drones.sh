#!/bin/bash
# Visualization script for trained MA-LSTM-PPO models

# Configuration
PYTHON="C:/Users/DELL/my-mappo/.venv/Scripts/python.exe"
export PYTHONPATH="C:\Users\DELL\my-mappo"

# Default parameters
MODEL_DIR=${1:-"C:\Users\DELL\my-mappo\onpolicy\scripts\results\pybullet-drones\drones_4\rmappo\check\run2\models"}
NUM_DRONES=${2:-3}
EPISODE_LENGTH=${3:-384}
RECURRENT_N=${4:-1}
RENDER_EPISODES=${5:-3}

echo "========================================="
echo "MA-LSTM-PPO Visualization"
echo "========================================="
echo "Model directory: $MODEL_DIR"
echo "Number of drones: $NUM_DRONES"
echo "Episode length: $EPISODE_LENGTH"
echo "Recurrent layers: $RECURRENT_N"
echo "Episodes per loop: $RENDER_EPISODES"
echo "========================================="
echo ""
echo "Press Ctrl+C to stop visualization"
echo ""

$PYTHON onpolicy/scripts/render/render_pybullet_drones.py \
    --model ma_lstm \
    --use_render \
    --model_dir "$MODEL_DIR" \
    --num_drones $NUM_DRONES \
    --n_rollout_threads 1 \
    --episode_length $EPISODE_LENGTH \
    --render_episodes $RENDER_EPISODES \
    --recurrent_N $RECURRENT_N
