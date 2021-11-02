#!/bin/bash

# python ./train_lstm.py --experiment_name "2021-08-20_lstm_eng_5e_256b_em100_hd100_v5000_run0" --gpu_run --batch_size 256 --embedding_dim 100 --hidden_dim 100 --seed 0 --num_epochs 5
#
# python ./train_lstm.py --experiment_name "2021-08-20_lstm_eng_5e_256b_em100_hd100_v5000_run1" --gpu_run --batch_size 256 --embedding_dim 100 --hidden_dim 100 --seed 1 --num_epochs 5
#
# python ./train_lstm.py --experiment_name "2021-08-20_lstm_eng_5e_256b_em100_hd100_v5000_run2" --gpu_run --batch_size 256 --embedding_dim 100 --hidden_dim 100 --seed 2 --num_epochs 5
#
# python ./train_lstm.py --experiment_name "2021-08-20_lstm_eng_5e_256b_em100_hd100_v5000_run3" --gpu_run --batch_size 256 --embedding_dim 100 --hidden_dim 100 --seed 3 --num_epochs 5
#
# python ./train_lstm.py --experiment_name "2021-08-20_lstm_eng_5e_256b_em100_hd100_v5000_run4" --gpu_run --batch_size 256 --embedding_dim 100 --hidden_dim 100 --seed 4 --num_epochs 5

for seed in $(seq 0 4); do
 python ./train_lstm.py --experiment_name "2021-09-21_lstm_zho_20e_256b_em100_hd100_v5000_run$seed" --gpu_run --batch_size 256 --embedding_dim 100 --hidden_dim 100 --num_epochs 20 --seed $seed --training_data_path "../../Data/model_datasets/zho/train_vocab_size_5000.pkl" --validation_data_path "../../Data/model_datasets/zho/validation_vocab_size_5000.pkl" --test_data_path "../../Data/model_datasets/zho/test_child_data_vocab_size_5000.pkl"
done
