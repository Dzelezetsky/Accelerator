#!/bin/bash

# SEED 1

# HERE WE WILL TRAIN PRETAINRED MODEL AND DON'T FORGET ABOUT 15K RB FOR UNTRAINED MODEL





python MUJOCO/pomdp_lstm_stage2.py --env "HalfCheetah-v4"  --rb_size 75000 --seed 1 --max_timesteps 200000 --eval_freq 100000

python MUJOCO/pomdp_lstm_stage2.py --env "HalfCheetah-v4" --rb_size 75000 --seed 2 --max_timesteps 200000 --eval_freq 100000

python MUJOCO/pomdp_lstm_stage2.py --env "Ant-v4" --rb_size 75000 --seed 1 --max_timesteps 200000 --eval_freq 100000

python MUJOCO/pomdp_lstm_stage2.py --env "Ant-v4" --rb_size 75000 --seed 2 --max_timesteps 200000 --eval_freq 100000

python MUJOCO/pomdp_lstm_stage2.py --env "Hopper-v4" --rb_size 75000 --seed 1 --max_timesteps 200000 --eval_freq 100000

python MUJOCO/pomdp_lstm_stage2.py --env "Hopper-v4" --rb_size 75000 --seed 2 --max_timesteps 200000 --eval_freq 100000




python MUJOCO/pomdp_lstm_stage2.py --env "HalfCheetah-v4"  --rb_size 55000 --seed 1 --max_timesteps 200000 --eval_freq 100000

python MUJOCO/pomdp_lstm_stage2.py --env "HalfCheetah-v4" --rb_size 55000 --seed 2 --max_timesteps 200000 --eval_freq 100000

python MUJOCO/pomdp_lstm_stage2.py --env "Ant-v4" --rb_size 55000 --seed 1 --max_timesteps 200000 --eval_freq 100000

python MUJOCO/pomdp_lstm_stage2.py --env "Ant-v4" --rb_size 55000 --seed 2 --max_timesteps 200000 --eval_freq 100000

python MUJOCO/pomdp_lstm_stage2.py --env "Hopper-v4" --rb_size 55000 --seed 1 --max_timesteps 200000 --eval_freq 100000

python MUJOCO/pomdp_lstm_stage2.py --env "Hopper-v4" --rb_size 55000 --seed 2 --max_timesteps 200000 --eval_freq 100000



python MUJOCO/pomdp_lstm_stage2.py --env "HalfCheetah-v4"  --rb_size 35000 --seed 1 --max_timesteps 200000 --eval_freq 100000

python MUJOCO/pomdp_lstm_stage2.py --env "HalfCheetah-v4" --rb_size 35000 --seed 2 --max_timesteps 200000 --eval_freq 100000

python MUJOCO/pomdp_lstm_stage2.py --env "Ant-v4" --rb_size 35000 --seed 1 --max_timesteps 200000 --eval_freq 100000

python MUJOCO/pomdp_lstm_stage2.py --env "Ant-v4" --rb_size 35000 --seed 2 --max_timesteps 200000 --eval_freq 100000

python MUJOCO/pomdp_lstm_stage2.py --env "Hopper-v4" --rb_size 35000 --seed 1 --max_timesteps 200000 --eval_freq 100000

python MUJOCO/pomdp_lstm_stage2.py --env "Hopper-v4" --rb_size 35000 --seed 2 --max_timesteps 200000 --eval_freq 100000





python MUJOCO/pomdp_lstm_stage2.py --env "HalfCheetah-v4"  --rb_size 15000 --seed 1 --max_timesteps 200000 --eval_freq 100000

python MUJOCO/pomdp_lstm_stage2.py --env "HalfCheetah-v4" --rb_size 15000 --seed 2 --max_timesteps 200000 --eval_freq 100000

python MUJOCO/pomdp_lstm_stage2.py --env "Ant-v4" --rb_size 15000 --seed 1 --max_timesteps 200000 --eval_freq 100000

python MUJOCO/pomdp_lstm_stage2.py --env "Ant-v4" --rb_size 15000 --seed 2 --max_timesteps 200000 --eval_freq 100000

python MUJOCO/pomdp_lstm_stage2.py --env "Hopper-v4" --rb_size 15000 --seed 1 --max_timesteps 200000 --eval_freq 100000

python MUJOCO/pomdp_lstm_stage2.py --env "Hopper-v4" --rb_size 15000 --seed 2 --max_timesteps 200000 --eval_freq 100000

