#!/bin/bash

for seed in $(seq 0 4); do
 python ./extract_surprisal_values.py --experiment_dir "../../Results/experiments/2021-08-20_lstm_eng_5e_256b_em100_hd100_v5000_run$seed" --gpu_run --batch_size 500 
 python ./extract_surprisal_values.py --experiment_dir "../../Results/experiments/2021-08-20_lstm_eng_5e_256b_em100_hd100_v5000_run$seed" --gpu_run --batch_size 500 --data_path "../../Data/model_datasets/eng/train_vocab_size_5000.pkl" --split "train"
done
