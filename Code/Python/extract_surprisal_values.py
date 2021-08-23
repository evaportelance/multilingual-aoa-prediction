'''
Calculates the surprisal values for words in a list of words.
'''
import os
import torch
import torch.nn.functional as F
import sys
import argparse
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
    parser.add_argument("--aoa_word_list", default="../../Data/model_datasets/eng/aoa_words.csv")
    parser.add_argument("--experiment_dir", default="../../Results/experiments/")
    parser.add_argument("--model", default="model.pt")
    parser.add_argument("--split", default="validation")
    params = parser.parse_args()
    return params

def make_word_dict(aoa_word_list, vocabulary):
    word_dict = {}
    not_in_vocab = []
    for word in aoa_word_list:
        if word in vocabulary:
            token = vocabulary[word]
            word_dict[token] = word
        else:
            not_in_vocab.append(word)
    return word_dict, not_in_vocab


def get_surprisals(model, all_data, word_dict, device):
    model.eval()
    word_surprisals = {}
    for index in word_dict.keys():
        word_surprisals[word_dict[index]] = [0.0, 0]
    for utt in all_data:
        utt_tensor = torch.tensor(utt)[None, :]
        utt_tensor = utt_tensor.to(device)
        outputs = model(utt_tensor)
        surprisals = -F.log_softmax(outputs, dim=2)
        utt_tensor = torch.squeeze(utt_tensor)
        for word_index in word_dict:
            index_matches = (utt_tensor == word_index).nonzero(as_tuple=False)
            if len(index_matches) > 0:
                for i in index_matches:
                    match = i.item()
                    surprisal = (surprisals[0][match][word_index] + sys.float_info.epsilon).item()
                    word = word_dict[word_index]
                    word_surprisals[word][0] += surprisal
                    word_surprisals[word][1] += 1

    return word_surprisals

def get_batched_surprisals(model, dataloader, word_dict, device):
    model.eval()
    word_surprisals = {}
    for index in word_dict.keys():
        word_surprisals[word_dict[index]] = [0.0, 0]
    for i, batch in enumerate(dataloader):
        print(i)
        batch = batch.to(device)
        outputs = model(batch)
        surprisals = -F.log_softmax(outputs, dim=2)
        for word_index in word_dict:
            index_matches = (batch == word_index).nonzero(as_tuple=False)
            if len(index_matches) > 0:
                for i in index_matches:
                    match = surprisals[tuple(i)]
                    surprisal = match[word_index].item() + sys.float_info.epsilon
                    word = word_dict[word_index]
                    word_surprisals[word][0] += surprisal
                    word_surprisals[word][1] += 1
    return word_surprisals

def main():
    params = get_parameters()
    device = torch.device('cuda') if params.gpu_run == True else torch.device('cpu')
    vocabulary = utils.open_pkl(params.encoding_dictionary_path)
    model = torch.load(os.path.join(params.experiment_dir, params.model))
    model = model.to(device)
    data = Dataset(params.data_path)
    dataloader = DataLoader(data, batch_size=params.batch_size)
    word_list = utils.open_word_list_csv(params.aoa_word_list)
    word_dict, not_in_vocab_list = make_word_dict(word_list, vocabulary)
    #word_surprisals = get_surprisals(model, all_data, word_dict, device)
    word_surprisals = get_batched_surprisals(model, dataloader, word_dict, device)
    for word in not_in_vocab_list:
        word_surprisals[word] = [0.0, 0]
    file_name = params.split + "_average_surprisals.csv"
    utils.save_surprisals_as_csv(word_surprisals, params.experiment_dir, file_name)
if __name__=="__main__":
    main()
