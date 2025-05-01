#!/bin/bash

# SEED 1



python MUJOCO/sh_trainer.py  --env "HalfCheetah-v4"  --seed 1  --eval_freq 10000
python MUJOCO/sh_trainer.py  --env "HalfCheetah-v4"  --seed 2  --eval_freq 10000