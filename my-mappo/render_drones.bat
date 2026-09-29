@echo off
REM Visualization script for trained MA-LSTM-PPO models (Windows Batch)

REM Configuration
set PYTHON=C:\Users\DELL\my-mappo\.venv\Scripts\python.exe
set PYTHONPATH=C:\Users\DELL\my-mappo

REM Default parameters
set MODEL_DIR=%1
set NUM_DRONES=%2
set EPISODE_LENGTH=%3
set RECURRENT_N=%4

if "%MODEL_DIR%"=="" set MODEL_DIR=C:\Users\DELL\my-mappo\onpolicy\scripts\results\pybullet-drones\drones_3\rmappo\check\run26\models
if "%NUM_DRONES%"=="" set NUM_DRONES=3
if "%EPISODE_LENGTH%"=="" set EPISODE_LENGTH=384
if "%RECURRENT_N%"=="" set RECURRENT_N=1

echo =========================================
echo MA-LSTM-PPO Visualization
echo =========================================
echo Model directory: %MODEL_DIR%
echo Number of drones: %NUM_DRONES%
echo Episode length: %EPISODE_LENGTH%
echo Recurrent layers: %RECURRENT_N%
echo =========================================
echo.
echo Press Ctrl+C to stop visualization
echo.

%PYTHON% onpolicy/scripts/render/render_pybullet_drones.py ^
    --model ma_lstm ^
    --use_render ^
    --model_dir "%MODEL_DIR%" ^
    --num_drones %NUM_DRONES% ^
    --n_rollout_threads 1 ^
    --episode_length %EPISODE_LENGTH% ^
    --render_episodes 3 ^
    --recurrent_N %RECURRENT_N%
