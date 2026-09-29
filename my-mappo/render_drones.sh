#!/bin/bash
# Visualization script for trained MA-LSTM-PPO models
#
# Usage:
#   ./render_drones.sh <model_dir> [num_drones] [render_episodes] [extra args...]
#
# The environment flags (formation type, neighbour settings, ...) must match the ones
# used for training, otherwise the observation size differs and the model cannot be
# loaded. The defaults below match command.txt; pass different ones as extra args.

# Configuration
PYTHON=${PYTHON:-python}
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="$SCRIPT_DIR${PYTHONPATH:+:$PYTHONPATH}"

# Default parameters
MODEL_DIR=${1:?"usage: $0 <model_dir> [num_drones] [render_episodes] [extra args...]"}
NUM_DRONES=${2:-8}
RENDER_EPISODES=${3:-3}
shift $(( $# < 3 ? $# : 3 ))

echo "========================================="
echo "MA-LSTM-PPO Visualization"
echo "========================================="
echo "Model directory: $MODEL_DIR"
echo "Number of drones: $NUM_DRONES"
echo "Episodes per loop: $RENDER_EPISODES"
echo "========================================="
echo ""
echo "Press Ctrl+C to stop visualization"
echo ""

$PYTHON "$SCRIPT_DIR/onpolicy/scripts/render/render_pybullet_drones.py" \
    --model ma_lstm \
    --use_render \
    --model_dir "$MODEL_DIR" \
    --num_drones $NUM_DRONES \
    --n_rollout_threads 1 \
    --render_episodes $RENDER_EPISODES \
    --formation_type dynamic \
    --neighbour_radius 1.0 \
    --min_dynamic_neighbours 1 \
    --max_dynamic_neighbours 7 \
    "$@"
