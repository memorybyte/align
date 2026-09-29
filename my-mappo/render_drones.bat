@echo off
REM Visualization script for trained MA-LSTM-PPO models (Windows Batch)
REM
REM Usage: render_drones.bat <model_dir> [num_drones] [render_episodes]
REM
REM The environment flags below must match the ones used for training, otherwise the
REM observation size differs and the model cannot be loaded (defaults match command.txt).
REM Episode length, recurrent_N and hidden size are fixed by the render script.

REM Configuration
if "%PYTHON%"=="" set PYTHON=python
set PYTHONPATH=%~dp0

REM Default parameters
set MODEL_DIR=%1
set NUM_DRONES=%2
set RENDER_EPISODES=%3

if "%MODEL_DIR%"=="" (
    echo usage: render_drones.bat ^<model_dir^> [num_drones] [render_episodes]
    exit /b 1
)
if "%NUM_DRONES%"=="" set NUM_DRONES=8
if "%RENDER_EPISODES%"=="" set RENDER_EPISODES=3

echo =========================================
echo MA-LSTM-PPO Visualization
echo =========================================
echo Model directory: %MODEL_DIR%
echo Number of drones: %NUM_DRONES%
echo Episodes per loop: %RENDER_EPISODES%
echo =========================================
echo.
echo Press Ctrl+C to stop visualization
echo.

%PYTHON% "%~dp0onpolicy\scripts\render\render_pybullet_drones.py" ^
    --model ma_lstm ^
    --use_render ^
    --model_dir "%MODEL_DIR%" ^
    --num_drones %NUM_DRONES% ^
    --n_rollout_threads 1 ^
    --render_episodes %RENDER_EPISODES% ^
    --formation_type dynamic ^
    --neighbour_radius 1.0 ^
    --min_dynamic_neighbours 1 ^
    --max_dynamic_neighbours 7
