import torch
import torch.nn.functional as F
import sys

'''
Finds the surprisal of each word in the vocabulary averaged across the words surprisal for every place where it 
occurs in the data.

Parameters:
    vocabulary: a dictionary of words to their numerical encodings
    model: a trained lstm
    all_data: a list containing all utterances in the dataset, where every utterance list is a list of words
    
Returns:
    average_surprisals: a dictionary of word encodings to average surprisal values
'''
def find_surprisal_values(vocabulary, model, all_data):
    surprisal_info = {}
    for word in vocabulary.values():
        surprisal_info[word] = [0.0, 0]
    for utt in all_data:
        for word in vocabulary.values():
            if word in utt:
                indexes_in_utt = [i for i, token in enumerate(utt) if token == word]
                print(indexes_in_utt)
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




