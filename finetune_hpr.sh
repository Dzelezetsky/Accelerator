#!/bin/bash

# SEED 1



python MUJOCO/pomdp_lstm_stage2.py  --env "Hopper-v4"  --seed 3  --eval_freq 10000  --max_timesteps 500000
python MUJOCO/pomdp_lstm_stage2.py  --env "Hopper-v4"  --seed 2  --eval_freq 10000  --max_timesteps 500000