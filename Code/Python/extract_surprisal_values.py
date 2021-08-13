'''
Calculates the surprisal values for words in list. [MORE DETAIL NEEDED]
'''

import torch
import torch.nn.functional as F
import sys
import argparse
import utils

'''
Gets arguments from the command-line.

Returns:
    params: a dictionary of command-line arguments
'''
def get_parameters():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu_run", action="store_true")
    parser.add_argument("-word_list", default="word_list.pkl")
    parser.add_argument("-experiment_dir", default="../../results/experiments/2021-12-08T22-59-15/")
    parser.add_argument("-model", default="model")
    parser.add_argument("-all_data", default="all_data.pkl")
    params = vars(parser.parse_args()) #converts namespace to dictionary
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
    for word in word_list.values():
        surprisal_info[word] = [0.0, 0]
    for utt in all_data:
        for word in word_list.values():
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
        average_surpisals[word] = surprisal_info[word][0]/surprisal_info[word][1]
    return average_surpisals

def main():
    params = get_parameters()
    #May add batching, optimize, and cuda support
    #device = torch.device('cuda') if params["gpu_run"] == True else torch.device('cpu')
    word_list = utils.open_pkl(params["experiment_dir"] + params["word_list"])
    model = torch.load(params["experiment_dir"] + params["model"])
    all_data = utils.open_pkl(params["experiment_dir"] + params["all_data"])
    average_surprisals = find_surprisal_values(word_list, model, all_data)
    print(average_surprisals)
    #[SAVE AS SOMETHING]
if __name__=="__main__":
    main()


