#!/bin/bash

for seed in $(seq 1 4); do
  python ./extract_surprisal_values_lstm.py --experiment_dir "../../Results/experiments/keepers/2021-09-21_lstm_deu_15e_256b_em100_hd100_v5000_run$seed" --gpu_run --batch_size 500 --data_path "../../Data/model_datasets/deu/validation_vocab_size_5000.pkl" --encoding_dictionary_path "../../Data/model_datasets/deu/encoding_dictionary_vocab_size_5000.pkl" --aoa_word_list "../../Data/word-lists/deu/word_list_german_clean.csv"

  python ./extract_surprisal_values_lstm.py --experiment_dir "../../Results/experiments/keepers/2021-09-21_lstm_spa_15e_256b_em100_hd100_v5000_run$seed" --gpu_run --batch_size 500 --data_path "../../Data/model_datasets/spa/validation_vocab_size_5000.pkl" --encoding_dictionary_path "../../Data/model_datasets/spa/encoding_dictionary_vocab_size_5000.pkl" --aoa_word_list "../../Data/word-lists/spa/word_list_spanish_(mexican)_clean.csv"

  python ./extract_surprisal_values_lstm.py --experiment_dir "../../Results/experiments/keepers/2021-09-21_lstm_zho_20e_256b_em100_hd100_v5000_run$seed" --gpu_run --batch_size 500 --data_path "../../Data/model_datasets/zho/validation_vocab_size_5000.pkl" --encoding_dictionary_path "../../Data/model_datasets/zho/encoding_dictionary_vocab_size_5000.pkl" --aoa_word_list "../../Data/word-lists/zho/word_list_mandarin_(beijing)_clean.csv"

  python ./extract_surprisal_values_lstm.py --experiment_dir "../../Results/experiments/keepers/2021-08-20_lstm_eng_5e_256b_em100_hd100_v5000_run$seed" --gpu_run --batch_size 500 --data_path "../../Data/model_datasets/eng/validation_vocab_size_5000.pkl" --encoding_dictionary_path "../../Data/model_datasets/eng/encoding_dictionary_vocab_size_5000.pkl" --aoa_word_list "../../Data/word-lists/eng/word_list_english_(american)_clean.csv"
done

# python ./extract_surprisal_values.py --experiment_dir "../../Results/experiments/2021-08-20_lstm_eng_5e_256b_em100_hd100_v5000_run$seed" --gpu_run --batch_size 500 --data_path "../../Data/model_datasets/eng/train_vocab_size_5000.pkl" --split "train"
