import argparse
import os
import sys
import torch
import torch.nn.functional as F
from bert_custom_dataset import CHILDESDataset
import operator
import functools
import math
from torch.utils.data import DataLoader
import utils

def get_parameters():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", default="../../Data/model_datasets/eng/validation.txt")
    parser.add_argument("--gpu_run", action="store_true")
    parser.add_argument("--batch_size", default=6, type=int)
    parser.add_argument("--aoa_word_list", default="../../Data/word-lists/eng/word_list_english_(american)_clean.csv")
    parser.add_argument("--experiment_dir", default="../../Results/experiments/")
    parser.add_argument("--model", default="model.pt")
    parser.add_argument("--split", default="validation")
    params = parser.parse_args()
    return params


def make_token_word_pairs(word_list_dict, dataset):
    tokenizer = dataset.tokenizer
    word_pairs = []
    for word in word_list_dict:
        seq = tokenizer(word['word_clean'])['input_ids']
        token = torch.Tensor(seq[1:-1]).long()
        word_pairs.append((token, word['word_clean'], len(token), word['language']))
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
        if n % 100 == 0:
            print(n)
        for key in batch:
            batch[key] = batch[key].to(device)
        outputs = model(**batch)
        surprisals = -F.log_softmax(outputs.logits, -1)
        labels = batch['labels']
        labels_split = torch.tensor_split(labels, batch_size)
        for indexes, word, n_tokens, language in word_pairs:
            indexes = indexes.to(device)
            if n_tokens == 1:
                index_matches = (labels == indexes).nonzero(as_tuple=False)
                if len(index_matches) > 0:
                    for i in index_matches:
                        match = surprisals[tuple(i)]
                        surprisal = match[indexes].item()
                        perplexity = 1/math.exp(-surprisal)
                        word_surprisals_perplexity_n[word][1] += surprisal
                        word_surprisals_perplexity_n[word][2] += perplexity
                        word_surprisals_perplexity_n[word][3] += 1
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
                        word_surprisals_perplexity_n[word][1] += surprisal
                        word_surprisals_perplexity_n[word][2] += perplexity
                        word_surprisals_perplexity_n[word][3] += 1
    return word_surprisals_perplexity_n

def main():
    params = get_parameters()
    device = torch.device('cuda') if params.gpu_run == True else torch.device('cpu')
    if params.gpu_run == True:
        torch.cuda.empty_cache()
    model = torch.load(os.path.join(params.experiment_dir, params.model))
    model = model.to(device)
    data = CHILDESDataset(params.data_path)
    dataloader = DataLoader(data, batch_size=params.batch_size)
    word_list_dict = utils.open_word_list_csv(params.aoa_word_list)
    word_pairs = make_token_word_pairs(word_list_dict, data)
    word_surprisals_perplexity_n = get_batched_surprisal_perplexity(model, dataloader, word_pairs, device)
    file_name = params.split + "_average_surprisal_perplexity.csv"
    utils.save_surprisals_as_csv(word_surprisals_perplexity_n, params.experiment_dir, file_name)

if __name__=="__main__":
    main()
