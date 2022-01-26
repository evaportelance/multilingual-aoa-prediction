#!/bin/bash

python ./get_token_frequencies.py --language "eng" --gpu_run --batch_size 500

python ./get_token_frequencies.py --language "fra" --gpu_run --batch_size 500

python ./get_token_frequencies.py --language "deu" --gpu_run --batch_size 500

python ./get_token_frequencies.py --language "spa" --gpu_run --batch_size 500

python ./get_token_frequencies.py --language "zho" --gpu_run --batch_size 500
