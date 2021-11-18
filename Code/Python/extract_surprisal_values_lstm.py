'''
Calculates the surprisal values for words in a list of words.
'''
import os
import torch
import torch.nn.functional as F
import sys
import argparse
import functools
import operator
import math
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
    parser.add_argument("--data_path", default="../../Data/model_datasets/eng/validation_vocab_size_5000.pkl")
    parser.add_argument("--encoding_dictionary_path", default="../../Data/model_datasets/eng/encoding_dictionary_vocab_size_5000.pkl")
    parser.add_argument("--gpu_run", action="store_true")
    parser.add_argument("--batch_size", default=1000, type=int)
    parser.add_argument("--aoa_word_list", default="../../Data/word-lists/eng/word_list_english_(american)_clean.csv")
    parser.add_argument("--experiment_dir", default="../../Results/experiments/")
    parser.add_argument("--model", default="model.pt")
    parser.add_argument("--split", default="validation")
    params = parser.parse_args()
    return params

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
            word_pairs.append((token_seq, word['word_clean'], len(token_seq), word['language']))
    return word_pairs

def indexes_in_sequence(query, base):
    id_, label = base[0], base[1]
    label = label.squeeze()
    l = len(query)
    locations = []
    for i in range((len(label)-l)):
        if torch.all(label[i:i+l] == query):
            locations.append([id_, i])
    return locations

def get_batched_surprisal_perplexity(model, dataloader, word_pairs, device):
    model.eval()
    tokens_surprisals_perplexity_n = {}
    for index, word, n_tokens, language in word_pairs:
        tokens_surprisals_perplexity_n[word] = [n_tokens, sys.float_info.epsilon, sys.float_info.epsilon, 0, language]
    batch_size = dataloader.batch_size
    for n, batch in enumerate(dataloader):
        batch = batch.to(device)
        outputs = model(batch)
        surprisals = -F.log_softmax(outputs, dim=2)
        labels_split = torch.tensor_split(batch, batch_size)
        for indexes, word, n_tokens, language in word_pairs:
            indexes = indexes.to(device)
            if n_tokens == 1:
                index_matches = (batch == indexes).nonzero(as_tuple=False)
                if len(index_matches) > 0:
                    for i in index_matches:
                        match = surprisals[tuple(i)]
                        surprisal = match[indexes].item()
                        perplexity = 1/math.exp(-surprisal)
                        tokens_surprisals_perplexity_n[word][1] += surprisal
                        tokens_surprisals_perplexity_n[word][2] += perplexity
                        tokens_surprisals_perplexity_n[word][3] += 1
            else:
                match_list = list(map(lambda x: indexes_in_sequence(indexes, x), enumerate(labels_split)))
                index_matches = functools.reduce(operator.iconcat, match_list)
                if len(index_matches) > 0:
                    for i in index_matches:
                        surprisal = 0.0
                        for j, index in enumerate(indexes):
                            id_ = i
                            id_[1] += j
                            match = surprisals[tuple(id_)]
                            sub_surprisal = match[index].item()
                            surprisal += sub_surprisal
                        perplexity = 1/math.exp(((-surprisal)/n_tokens))
                        tokens_surprisals_perplexity_n[word][1] += surprisal
                        tokens_surprisals_perplexity_n[word][2] += perplexity
                        tokens_surprisals_perplexity_n[word][3] += 1
    return tokens_surprisals_perplexity_n

def main():
    params = get_parameters()
    device = torch.device('cuda') if params.gpu_run == True else torch.device('cpu')
    vocabulary = utils.open_pkl(params.encoding_dictionary_path)
    model = torch.load(os.path.join(params.experiment_dir, params.model))
    model = model.to(device)
    data = Dataset(params.data_path)
    dataloader = DataLoader(data, batch_size=params.batch_size)
    word_list = utils.open_word_list_csv(params.aoa_word_list)
    word_pairs = make_token_word_pairs(word_list, vocabulary)
    word_surprisals = get_batched_surprisal_perplexity(model, dataloader, word_pairs, device)
    file_name = params.split + "_average_surprisal_perplexity.csv"
    utils.save_surprisals_as_csv(word_surprisals, params.experiment_dir, file_name)
if __name__=="__main__":
    main()
