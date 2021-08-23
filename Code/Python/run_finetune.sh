#!/bin/bash

#python ./finetune_multiling_bert.py --train_path '../../Data/model_datasets/eng/train.txt' --val_path '../../Data/model_datasets/eng/validation.txt' --result_dir '../../Results/experiments/' --experiment_name '2021-08-20_fintune_eng_e30_b256_lr1e-3_run0' --batch_size 64 --n_epochs 30 --lr 1e-3

#python ./finetune_multiling_bert.py --train_path '../../Data/model_datasets/eng/train.txt' --val_path '../../Data/model_datasets/eng/validation.txt' --result_dir '../../Results/experiments/' --experiment_name '2021-08-20_fintune_eng_e30_b256_lr1e-3_run0' --batch_size 256 --n_epochs 30 --lr 1e-4

python ./finetune_multiling_bert.py --train_path '../../Data/model_datasets/eng/train.txt' --val_path '../../Data/model_datasets/eng/validation.txt' --result_dir '../../Results/experiments/' --experiment_name '2021-08-20_fintune_eng_e30_b32_lr5e-5_run0' --batch_size 32 --n_epochs 30
