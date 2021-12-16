#!/bin/bash

python ./data_preprocessing.py --dataset_path "../../Data/model_datasets/fra/" --vocab_size 5000 --existing_encoding_dict "encoding_dictionary_vocab_size_5000.pkl"

python ./data_preprocessing.py --dataset_path "../../Data/model_datasets/deu/" --vocab_size 5000 --existing_encoding_dict "encoding_dictionary_vocab_size_5000.pkl"

python ./data_preprocessing.py --dataset_path "../../Data/model_datasets/zho/" --vocab_size 5000 --existing_encoding_dict "encoding_dictionary_vocab_size_5000.pkl"

python ./data_preprocessing.py --dataset_path "../../Data/model_datasets/spa/" --vocab_size 5000 --existing_encoding_dict "encoding_dictionary_vocab_size_5000.pkl"

for seed in $(seq 0 2); do

python ./train_lstm.py --experiment_name "2021-12-10_lstm_fra_20e_256b_em100_hd100_v5000_run$seed" --gpu_run --batch_size 256 --embedding_dim 100 --hidden_dim 100 --num_epochs 20 --seed $seed --training_data_path "../../Data/model_datasets/fra/all_child_directed_data_vocab_size_5000.pkl" --validation_data_path "../../Data/model_datasets/fra/validation_vocab_size_5000.pkl" --test_data_path "../../Data/model_datasets/fra/test_child_data_vocab_size_5000.pkl"

python ./train_lstm.py --experiment_name "2021-12-10_lstm_deu_20e_256b_em100_hd100_v5000_run$seed" --gpu_run --batch_size 256 --embedding_dim 100 --hidden_dim 100 --num_epochs 20 --seed $seed --training_data_path "../../Data/model_datasets/deu/all_child_directed_data_vocab_size_5000.pkl" --validation_data_path "../../Data/model_datasets/deu/validation_vocab_size_5000.pkl" --test_data_path "../../Data/model_datasets/deu/test_child_data_vocab_size_5000.pkl"

python ./train_lstm.py --experiment_name "2021-12-10_lstm_spa_20e_256b_em100_hd100_v5000_run$seed" --gpu_run --batch_size 256 --embedding_dim 100 --hidden_dim 100 --num_epochs 20 --seed $seed --training_data_path "../../Data/model_datasets/spa/all_child_directed_data_vocab_size_5000.pkl" --validation_data_path "../../Data/model_datasets/spa/validation_vocab_size_5000.pkl" --test_data_path "../../Data/model_datasets/spa/test_child_data_vocab_size_5000.pkl"

python ./train_lstm.py --experiment_name "2021-12-10_lstm_zho_20e_256b_em100_hd100_v5000_run$seed" --gpu_run --batch_size 256 --embedding_dim 100 --hidden_dim 100 --num_epochs 20 --seed $seed --training_data_path "../../Data/model_datasets/zho/all_child_directed_data_vocab_size_5000.pkl" --validation_data_path "../../Data/model_datasets/zho/validation_vocab_size_5000.pkl" --test_data_path "../../Data/model_datasets/zho/test_child_data_vocab_size_5000.pkl"

done
