#!/bin/bash

python ./train_lstm.py --training_data_path "../../Data/model-sets/train.txt" --validation_data_path "../../Data/model-sets/validation.txt" --test_data_path "../../Data/model-sets/test.txt" --experiment_name "2021-08-13_lstm_eng_50e_200b_em150_hd100_v2000" --gpu_run --batch_size 200 --num_epochs 50 --embedding_dim 150 --hidden_dim 100 --vocab_size 2000 --patience 50
