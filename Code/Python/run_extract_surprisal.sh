#!/bin/bash

for seed in $(seq 0 2); do
  python ./extract_singleword_surprisals_lstm.py --language "eng" --gpu_run --batch_size 500 --aoa_word_list "word_list_english_(american)_clean.csv" --experiment_dir "../../Results/experiments/2021-12-10_lstm_eng_20e_256b_em100_hd100_v5000_run$seed/"

  python ./extract_singleword_surprisals_lstm.py --language "eng" --gpu_run --batch_size 500 --aoa_word_list "word_list_english_(british)_clean.csv" --experiment_dir "../../Results/experiments/2021-12-10_lstm_eng_20e_256b_em100_hd100_v5000_run$seed/"

  python ./extract_singleword_surprisals_lstm.py --language "eng" --gpu_run --batch_size 500 --aoa_word_list "word_list_english_(australian)_clean.csv" --experiment_dir "../../Results/experiments/2021-12-10_lstm_eng_20e_256b_em100_hd100_v5000_run$seed/"

  echo "English $seed done"

  python ./extract_singleword_surprisals_lstm.py --language "deu" --gpu_run --batch_size 500 --aoa_word_list "word_list_german_clean_caps.csv" --experiment_dir "../../Results/experiments/2021-12-10_lstm_deu_20e_256b_em100_hd100_v5000_run$seed/"

  echo "German $seed done"

  python ./extract_singleword_surprisals_lstm.py --language "fra" --gpu_run --batch_size 500 --aoa_word_list "word_list_french_(french)_clean.csv" --experiment_dir "../../Results/experiments/2021-12-10_lstm_fra_20e_256b_em100_hd100_v5000_run$seed/"

  python ./extract_singleword_surprisals_lstm.py --language "fra" --gpu_run --batch_size 500 --aoa_word_list "word_list_french_(quebecois)_clean.csv" --experiment_dir "../../Results/experiments/2021-12-10_lstm_fra_20e_256b_em100_hd100_v5000_run$seed/"

  echo "French $seed done"

  python ./extract_singleword_surprisals_lstm.py --language "spa" --gpu_run --batch_size 500 --aoa_word_list "word_list_spanish_(european)_clean.csv" --experiment_dir "../../Results/experiments/2021-12-10_lstm_spa_20e_256b_em100_hd100_v5000_run$seed/"

  python ./extract_singleword_surprisals_lstm.py --language "spa" --gpu_run --batch_size 500 --aoa_word_list "word_list_spanish_(mexican)_clean.csv" --experiment_dir "../../Results/experiments/2021-12-10_lstm_spa_20e_256b_em100_hd100_v5000_run$seed/"

  echo "Spanish $seed done"

  python ./extract_singleword_surprisals_lstm.py --language "zho" --gpu_run --batch_size 500 --aoa_word_list "word_list_mandarin_(beijing)_clean.csv" --experiment_dir "../../Results/experiments/2021-12-10_lstm_zho_20e_256b_em100_hd100_v5000_run$seed/"

  python ./extract_singleword_surprisals_lstm.py --language "zho" --gpu_run --batch_size 500 --aoa_word_list "word_list_mandarin_(taiwanese)_clean.csv" --experiment_dir "../../Results/experiments/2021-12-10_lstm_zho_20e_256b_em100_hd100_v5000_run$seed/"

  echo "Mandarin $seed done"
done
