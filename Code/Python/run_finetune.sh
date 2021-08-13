#!/bin/bash

python ./multiling_bert_finetune.py --train_path '../../Data/model-sets/toy_train.txt' --val_path '../../Data/model-sets/toy_validation.txt' --result_dir '../../Results/experiments/' --experiment_name 'test-finetune' --batch_size 10 --n_epochs 5
