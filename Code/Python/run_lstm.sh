#!/bin/bash

python ./train_lstm.py --experiment_name "2021-08-20_lstm_eng_5e_256b_em100_hd100_v5000_run0" --gpu_run --batch_size 256 --embedding_dim 100 --hidden_dim 100 --seed 0 --num_epochs 5

python ./train_lstm.py --experiment_name "2021-08-20_lstm_eng_5e_256b_em100_hd100_v5000_run1" --gpu_run --batch_size 256 --embedding_dim 100 --hidden_dim 100 --seed 1 --num_epochs 5

python ./train_lstm.py --experiment_name "2021-08-20_lstm_eng_5e_256b_em100_hd100_v5000_run2" --gpu_run --batch_size 256 --embedding_dim 100 --hidden_dim 100 --seed 2 --num_epochs 5

python ./train_lstm.py --experiment_name "2021-08-20_lstm_eng_5e_256b_em100_hd100_v5000_run3" --gpu_run --batch_size 256 --embedding_dim 100 --hidden_dim 100 --seed 3 --num_epochs 5

python ./train_lstm.py --experiment_name "2021-08-20_lstm_eng_5e_256b_em100_hd100_v5000_run4" --gpu_run --batch_size 256 --embedding_dim 100 --hidden_dim 100 --seed 4 --num_epochs 5
