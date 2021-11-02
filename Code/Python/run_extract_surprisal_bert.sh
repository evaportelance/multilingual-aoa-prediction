#!/bin/bash

# for seed in $(seq 0 4); do
#  python ./extract_surprisal_values_bert.py --experiment_dir "../../Results/experiments/2021-08-31_finetune_eng_e1_b6_lr5e-5_run$seed" --gpu_run --batch_size 6 
#  python ./extract_surprisal_values_bert.py --experiment_dir "../../Results/experiments/2021-08-31_finetune_eng_e1_b6_lr5e-5_run$seed" --gpu_run --batch_size 6 --data_path "../../Data/model_datasets/eng/train.txt" --split "train"
# done

python ./extract_surprisal_values_bert.py --experiment_dir "../../Results/experiments/2021-08-31_finetune_eng_e1_b6_lr5e-5_run0" --gpu_run --batch_size 6 