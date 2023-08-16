# Predicting age of acquisition for children’s early vocabulary in five languages using language model surprisal

This repository contains all the data, code, and analyses presented in the paper entitled _Predicting age of acquisition for children’s early vocabulary in five languages using language model surprisal_. 

Please cite the following paper if you choose to use any of the cleaned data, code, or results from this repository:
```
@article{portelance2023predicting,
  author  = {Portelance, Eva and Duan, Yuguang and Frank, Michael C. and Lupyan, Gary},
  title   = {Predicting age of acquisition for children’s early vocabulary in five languages using language model surprisal},
  journal = {Cognitive Science},
  year    = {To Appear}
}
```
Additionally, please cite the CHILDES and Wordbank repositories, where the original data is stored, as well as the original studies in each language used:
```
@book{macwhinney2000CHILDES,
  title={The {CHILDES} {P}roject: {T}ools for analyzing talk. {T}hird {E}dition.},
  author={MacWhinney, B.},
  year={2000},
  publisher={Erlbaum}
}

@article{frank2017wordbank,
  title={Wordbank: An open repository for developmental vocabulary data},
  author={Frank, Michael C and Braginsky, Mika and Yurovsky, Daniel and Marchman, Virginia A},
  journal={Journal of child language},
  volume={44},
  number={3},
  pages={677--694},
  year={2017},
  publisher={Cambridge University Press}
}
```
## How to

### Where to find the corpora of child-directed utterances
All of the child-directed utterances are collected in the directory *./Data/model_datasets/* . There are subdirectories for each language (English - eng, French - fra, Spanish - spa, German - deu, Mandarin - zho). The file named 'all_child_directed_data.txt' in each of these directories contains all of the cleaned child-directed utterances. The file 'test_child_data.txt' contains all of the cleaned child-produced utterances. The files 'train.txt' and 'validation.txt' contain respectively a mutually exclusive 80% and 20% random set of all the child-directed utterances.

You can also find all of the original transcripts separated by corpus in CHILDES and by child in the directory *./Data/transcripts*. These again are separated into language specific subdirectories.

### Training LSTMs on child-directed corpora and extracting surprisal and frequency values

(Note: all of the model average surprisal values have already been cashed for the analyses, so if you would like you can skip this step. Only run this step if you'd like to train the language models again and extract surprisal values again.)

To train all of the LSTM models used in the paper you simply have to run the following executable file. Note that it expects that you have one gpu (by default 'gpu:0') available. For each language takes about one hour to train the models on an RTX 3080.

`source ./Code/Python/run_lstm.sh`

Once you have trained the models, you can extract the average surprisal of the words for which AoA estimates are available in each language. To do so, simply run the following executable, again it will expect one gpu to be available.

`source ./Code/Python/run_extract_surprisal.sh`

To extract token frequency values separately, run :

`source ./Code/Python/run_get_frequencies.sh`


All of the results will be saved to the experiment directory, by default *./Results/experiments/[NAME OF EXPERIMENT]*.

### Running the analyses and experiments

If you have retrained and extracted your one models, you'll need to collect the average surprisal results from each model run from the directory indicated above and then you can run the code in *./Analyses/surprisal_frequency_data_wrangling.Rmd* to average data across each random seed by language. Otherwise SKIP THIS STEP.

The code in the */Analyses/* section uses a lot of the base functions provided by the **AoA-pipeline** project (https://github.com/mikabr/aoa-pipeline), a joint initiative to create a library of functions for analyzing and predicting AoA using wordbank data.

To run the experiments presented in the manuscript:
For experiment 1, simply run the code in *./Analyses/approach1.Rmd* ;
For experiment 2, simply run the code in *./Analyses/approach2.Rmd* .


To reproduce the plots used in the manuscript, run the code in *./Analyses/plots/plots.Rmd*
