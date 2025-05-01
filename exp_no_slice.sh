#!/bin/bash

seed=(1 3 4)

for seed in ${seed[@]}
do

    python MANISKILL/state/State_Workshop_stage2.py --env="PickCube-v1" --seed $seed

  done