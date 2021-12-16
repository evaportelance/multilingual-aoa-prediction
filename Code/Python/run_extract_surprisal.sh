#!/bin/bash

for seed in $(seq 0 2); do
  python ./extract_singleword_surprisals_lstm.py --data_path "../../Data/model_datasets/eng/all_child_directed_data_vocab_size_5000.pkl" --encoding_dictionary_path "../../Data/model_datasets/eng/encoding_dictionary_vocab_size_5000.pkl" --gpu_run --batch_size 500 --aoa_word_list "../../Data/word-lists/eng/word_list_english_(australian)_clean.csv" --split "all_child_directed_data" --experiment_dir "../../Results/experiments/2021-12-10_lstm_eng_20e_256b_em100_hd100_v5000_run$seed/"
  echo "English $seed done"
done
