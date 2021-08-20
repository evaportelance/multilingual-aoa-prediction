'''
Calculates the surprisal values for words in list. [MORE DETAIL NEEDED]
'''
import os
import torch
import torch.nn.functional as F
import sys
import argparse
import utils

'''
to-do:
add word_list functionality
save as csv
write header comment for the train script

Gets arguments from the command-line.

Returns:
    params: a dictionary of command-line arguments
'''
def get_parameters():
    parser = argparse.ArgumentParser()
    parser.add_argument("--all_child_directed_data_path", default="../../Data/model_datasets/eng/all_child_directed_data_vocab_size_5000.pkl")
    parser.add_argument("--encoding_dictionary_path", default="../../Data/model_datasets/eng/encoding_dictionary_vocab_size_5000.pkl")
    #parser.add_argument("--gpu_run", action="store_true")
    parser.add_argument("--aoa_word_list", default="../../Data/model_datasets/eng/aoa_words.csv")
    parser.add_argument("--experiment_dir", default="../../Results/experiments/")
    parser.add_argument("--model", default="model.pt")
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
def find_surprisal_values(word_list, model, all_data):
    surprisal_info = {}
    for word in word_list:
        surprisal_info[word] = [0.0, 0]
    for utt in all_data:
        for word in word_list:
            if word in utt:
                #word = word.item().to(device)
                indexes_in_utt = [i for i, token in enumerate(utt) if token == word]
                utt_tensor = torch.tensor(utt)[None, :]
                output_weights = model(utt_tensor)
                surprisals = -F.log_softmax(output_weights, dim=2)
                word_surprisals = 0
                for index in indexes_in_utt:
                    word_surprisals += (surprisals[0][index][word] + sys.float_info.epsilon).item()
                surprisal_info[word][0] += word_surprisals
                surprisal_info[word][1] += len(indexes_in_utt)
    average_surpisals = {}
    for word in surprisal_info:
        if surprisal_info[word][1] != 0:
            average_surpisals[word] = surprisal_info[word][0]/surprisal_info[word][1]
        else:
            average_surpisals[word] = "NA"
    return average_surpisals

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
    average_surprisals = find_surprisal_values(vocab_word_list_intersection, model, all_data)
    print(average_surprisals)
    #[SAVE AS SOMETHING] save as csv sum number of cases word names not indexes
if __name__=="__main__":
    main()
