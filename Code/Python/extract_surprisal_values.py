'''
Calculates the surprisal values for words in a list of words.
'''

import torch
import torch.nn.functional as F
import sys
import argparse
import utils
import os

'''
Gets arguments from the command-line.

Returns:
    params: a dictionary of command-line arguments
'''
def get_parameters():
    parser = argparse.ArgumentParser()
    parser.add_argument("--all_child_directed_data_path", default="../../data/model-sets/toy_datasets/toy_all.pkl")
    parser.add_argument("--encoding_dictionary_path", default="../../data/model-sets/toy_datasets/encoding_dictionary_vocab_size_10")
    #parser.add_argument("--gpu_run", action="store_true")
    parser.add_argument("--aoa_word_list", default="../../data/model-sets/aoa_word_list.csv")
    parser.add_argument("--experiment_dir", default="../../results/experiments/2021-12-08T22-59-15/")
    parser.add_argument("--model", default="model")
    params = parser.parse_args()
    return params

'''
Make surprisal values run independently

Finds the surprisal of each word in the word_list averaged across the words surprisal for every place where it
occurs in the data.

Parameters:
    word_list: a dictionary of words to their numerical encodings
    model: a trained lstm
    all_data: a list containing all utterances in the dataset, where every utterance list is a list of words

Returns:
    average_surprisals: a dictionary of word encodings to average surprisal values
'''
def find_surprisal_values(word_list, model, all_data, vocabulary):
    surprisal_info = {}
    for word in vocabulary.values():
        surprisal_info[word] = [0.0, 0]
    for utt in all_data:
        utt_tensor = torch.tensor(utt)[None, :]
        output_weights = model(utt_tensor)
        surprisals = -F.log_softmax(output_weights, dim=2)
        i = 0
        for word in utt:
            if word != 0:
                surprisal_info[word][0] += (surprisals[0][i][word] + sys.float_info.epsilon).item()
                surprisal_info[word][1] += 1
            i += 1
    average_surprisals = {}
    for word in word_list:
        word_index = vocabulary[word]
        if surprisal_info[word_index][1] != 0:
            average_surprisals[word] = surprisal_info[word_index][0]/surprisal_info[word_index][1]
        else:
            average_surprisals[word] = "NA"
    return average_surprisals

def main():
    params = get_parameters()
    #May add batching, optimize, and cuda support
    #device = torch.device('cuda') if params["gpu_run"] == True else torch.device('cpu')
    vocabulary = utils.open_pkl(params.encoding_dictionary_path)
    word_list = set(utils.open_word_list_csv(params.aoa_word_list))
    in_word_list_not_vocab = word_list - set(vocabulary.keys())
    vocab_word_list_intersection = word_list - in_word_list_not_vocab
    model = torch.load(os.path.join(params.experiment_dir, params.model))
    all_data = utils.open_pkl(params.all_child_directed_data_path)
    average_surprisals = find_surprisal_values(vocab_word_list_intersection, model, all_data, vocabulary)
    utils.save_surprisals_as_csv(average_surprisals, params.experiment_dir)
    print(average_surprisals)
if __name__=="__main__":
    main()
