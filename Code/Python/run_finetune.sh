#!/bin/bash

for seed in $(seq 0 4); do
python ./finetune_multiling_bert.py --experiment_name "2021-08-31_finetune_eng_e1_b6_lr5e-5_run$seed" --seed $seed
done
