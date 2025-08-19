#!/bin/bash

# SEED 1



#python MUJOCO/stage2_sac.py  --env "Ant-v4"  --seed 1  --eval_freq 10000
# python MUJOCO/sh_trainer_sac.py  --env "Ant-v4"  --seed 2  --eval_freq 10000

python MUJOCO/stage2_sac.py  --env "Ant-v4"  --seed 1  --eval_freq 10000  --max_timesteps 500000
python MUJOCO/stage2_sac.py  --env "Ant-v4"  --seed 2  --eval_freq 10000  --max_timesteps 500000