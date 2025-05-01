#!/bin/bash

# SEED 1


python MUJOCO/pomdp_lstm_sh_trainer.py  --env "Hopper-v4"  --seed 3 --max_timesteps 500000
python MUJOCO/pomdp_lstm_sh_trainer.py  --env "Hopper-v4"  --seed 2 --max_timesteps 500000