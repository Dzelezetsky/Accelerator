#!/bin/bash

# SEED 1



python MUJOCO/sh_trainer.py  --env "Ant-v4"  --seed 3  --eval_freq 10000
python MUJOCO/sh_trainer.py  --env "Ant-v4"  --seed 4  --eval_freq 10000