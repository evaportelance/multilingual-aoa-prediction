#!/bin/bash

#for seed in $(seq 0 4); do
python ./finetune_multiling_bert.py --experiment_name "2021-09-22_finetune_eng_e3_b6_lr5e-5_run0" --seed 0 --train_path '../../Data/model_datasets/eng/train.txt' --val_path '../../Data/model_datasets/eng/validation.txt' --n_epochs 3

python ./finetune_multiling_bert.py --experiment_name "2021-09-22_finetune_fra_e3_b6_lr5e-5_run0" --seed 0 --train_path '../../Data/model_datasets/fra/train.txt' --val_path '../../Data/model_datasets/fra/validation.txt' --n_epochs 3

python ./finetune_multiling_bert.py --experiment_name "2021-09-22_finetune_deu_e3_b6_lr5e-5_run0" --seed 0 --train_path '../../Data/model_datasets/deu/train.txt' --val_path '../../Data/model_datasets/deu/validation.txt' --n_epochs 3

python ./finetune_multiling_bert.py --experiment_name "2021-09-22_finetune_spa_e3_b6_lr5e-5_run0" --seed 0 --train_path '../../Data/model_datasets/spa/train.txt' --val_path '../../Data/model_datasets/spa/validation.txt' --n_epochs 3

python ./finetune_multiling_bert.py --experiment_name "2021-09-22_finetune_zho_e3_b6_lr5e-5_run0" --seed 0 --train_path '../../Data/model_datasets/zho/train.txt' --val_path '../../Data/model_datasets/zho/validation.txt' --n_epochs 3
#done
