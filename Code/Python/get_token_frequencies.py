'''
Get word counts in corpus for word list.
'''
import os
import torch
import sys
import argparse
import functools
import operator
from data_loader import Dataset
from torch.utils.data import DataLoader
import utils
import os

'''
Gets arguments from the command-line.

Returns:
    params: a dictionary of command-line arguments
'''
def get_parameters():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", default="../../Data/")
    parser.add_argument("--language_code", default="eng")
    parser.add_argument("--gpu_run", action="store_true")
    parser.add_argument("--batch_size", default=1000, type=int)
    parser.add_argument("--result_dir", default="../../Results/frequencies/")
    params = parser.parse_args()
    return params

def get_total_child_directed_word_count(text_data_path):
    total_word_count = 0
    lines = utils.open_txt(text_data_path)
    for line in lines:
        total_word_count += len(line)
    return total_word_count


def make_token_word_pairs(aoa_word_list, vocabulary):
    word_pairs = []
    for word in aoa_word_list:
        word_seq = word["word_clean"].split(" ")
        seq_complete = True
        token_seq = []
        for w in word_seq:
            if w in vocabulary:
                token = vocabulary[w]
                if token != 0:
                    token_seq.append(token)
                else:
                    #remove words that arent in top 5000 tokens
                    seq_complete = False
                    break
            else:
                # remove words that aren't in Childes corpus
                seq_complete = False
                break
        if seq_complete:
            token_seq = torch.Tensor(token_seq).long()
            word_pairs.append((token_seq, word['word_clean'], len(token_seq)))
    return word_pairs

def get_token_counts(dataloader, word_pairs, device):
    tokens_count = {}
    for index, word, n_tokens in word_pairs:
        tokens_count[word] = 0
    batch_size = dataloader.batch_size
    for n, batch in enumerate(dataloader):
        batch = batch.to(device)
        labels_split = torch.tensor_split(batch, batch_size)
        for indexes, word, n_tokens in word_pairs:
            indexes = indexes.to(device)
            if n_tokens == 1:
                index_matches = (batch == indexes).nonzero(as_tuple=False)
                tokens_count[word] += len(index_matches)
            else:
                match_list = list(map(lambda x: indexes_in_sequence(indexes, x), enumerate(labels_split)))
                index_matches = functools.reduce(operator.iconcat, match_list)
                tokens_count[word] += len(index_matches)
    return tokens_count


def main():
    params = get_parameters()
    dict_path = params.data_path + "model_datasets/" + params.language_code + "/encoding_dictionary_vocab_size_5000.pkl"
    train_data_path = params.data_path + "model_datasets/" + params.language_code + "/train_vocab_size_5000.pkl"
    val_data_path = params.data_path + "model_datasets/" + params.language_code + "/validation_vocab_size_5000.pkl"
    text_data_path = params.data_path + "model_datasets/" + params.language_code + "/all_child_directed_data.txt"
    word_list_path = params.data_path + "word-lists/" + params.language_code + "/unique_clean_words.csv"
    print("GET TOTAL WORD COUNT...")
    total_count = get_total_child_directed_word_count(text_data_path)
    print(total_count)
    print("GET BY WORD COUNT...")
    device = torch.device('cuda') if params.gpu_run == True else torch.device('cpu')
    vocabulary = utils.open_pkl(dict_path)
    word_list = utils.open_word_list_csv(word_list_path)
    word_pairs = make_token_word_pairs(word_list, vocabulary)
    print("FOR TRAIN SET...")
    data = Dataset(train_data_path)
    dataloader = DataLoader(data, batch_size=params.batch_size)
    train_word_counts = get_token_counts(dataloader, word_pairs, device)
    print("FOR VALIDATION SET...")
    data = Dataset(val_data_path)
    dataloader = DataLoader(data, batch_size=params.batch_size)
    val_word_counts = get_token_counts(dataloader, word_pairs, device)
    file_name = params.language_code + "_frequency_counts.csv"
    utils.save_frequencies_as_csv(train_word_counts, val_word_counts, total_count, params.result_dir, file_name)
if __name__=="__main__":
    main()
