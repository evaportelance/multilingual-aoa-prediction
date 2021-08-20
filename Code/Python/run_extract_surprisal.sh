#!/bin/bash

for seed in $(seq 0 4); do
  python ./extract_surprisal_values.py --experiment_dir "../../Results/experiments/2021-08-20_lstm_eng_5e_256b_em100_hd100_v5000_run$seed"
done
