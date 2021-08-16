
import utils
import random
import argparse
from operator import itemgetter
import os


def get_parameters():
    parser = argparse.ArgumentParser()
    parser.add_argument("-training_dataset_name", default="toy_train")
    parser.add_argument("-validation_dataset_name", default="toy_validation")
    parser.add_argument("-test_dataset_name", default="toy_test")
    parser.add_argument("-all_dataset_name", default="toy_all")
    parser.add_argument("-encoding_dictionary_name", default="encoding_dictionary")
    parser.add_argument("-dataset_path", default="../../data/model-sets/toy_datasets/")
    parser.add_argument("-vocab_size", default=10, type=int)
    params = vars(parser.parse_args())  # converts namespace to dictionary
    return params


'''
    Creates a dictionary from a word list where the keys are words and the values are
    the number of occurrences of the word in the word list for the number of words in
    VOCAB_SIZE that have the most occurrences in the word list. All other word-keys
    have value 0.
    Parameters:
        vocabulary: one-dimensional list of words
    Returns:
        encoding_dictionary: a dictionary of words to word counts
'''


def create_encoding_dictionary(all_data, vocab_size):
    flattened_list = [word for utterance in all_data for word in utterance]
    word_set = set(flattened_list)
    word_index_dict = {}
    encoding_dictionary = word_index_dict.fromkeys(word_set, 0)
    for word in flattened_list:
        encoding_dictionary[word] = encoding_dictionary[word] + 1
    vocabulary = dict(sorted(encoding_dictionary.items(), key=itemgetter(1), reverse=True)[:vocab_size])

    # Set word dict values equal to indices
    i = 1
    for word in word_set:
        if word in vocabulary:
            encoding_dictionary[word] = i
            vocabulary[word] = i
            i += 1
        else:
            encoding_dictionary[word] = 0
    return encoding_dictionary


'''
Replaces all the words in dataset in a tuple of datasets with their encoding in a dictionary.
Parameters:
    datasets: a tuple of datasets
    encoding_dict: a dictionary of words to indexes
'''


def replace_words_with_indexes(datasets, encoding_dict):
    max_utterance_lens = []
    max_utterance_len = 0
    for dataset in datasets:
        for i in range(len(dataset)):
            if len(dataset[i]) > max_utterance_len:
                max_utterance_len = len(dataset[i])
            for j in range(len(dataset[i])):
                # assert dataset[i][j] in encoding_dict[dataset[i][j]].keys, \
                #    "Encoding_dict was not created using these datasets."
                dataset[i][j] = encoding_dict[dataset[i][j]]
        max_utterance_lens.append(max_utterance_len)
        max_utterance_len = 0
    return datasets, max_utterance_lens


'''
Pads of all the utterances in a tuple of datasets with 0's so that all of the utterances are the same length as the 
longest unpadded utterance.
Parameters:
    datasets: a tuple of datasets
    encoding_dict: a dictionary of words to indexes
    max_utterance_len: the length of the longest utterance 
'''


def pad_utterances_to_same_length(datasets, encoding_dict, max_utterance_len):
    j = 0
    for dataset in datasets:
        for i in range(len(dataset)):
            zeros = 0 * random.choice(list(encoding_dict.values()))
            padding = [zeros] * (max_utterance_len[j] - len(dataset[i]))
            dataset[i] = padding + dataset[i]
        j += 1
    return datasets


def main():
    params = get_parameters()
    if not os.path.isdir(params["dataset_path"]):
        os.mkdir(params["dataset_path"])

    # Open files
    train_data = utils.open_txt(params["dataset_path"] + params["training_dataset_name"] + ".txt")
    validation_data = utils.open_txt(params["dataset_path"] + params["validation_dataset_name"] + ".txt")
    test_data = utils.open_txt(params["dataset_path"] + params["test_dataset_name"] + ".txt")
    datasets = (train_data, validation_data, test_data)

    encoding_dictionary = create_encoding_dictionary(train_data + validation_data + test_data,
                                                     params["vocab_size"])

    # Process data
    datasets, max_utterance_lens = replace_words_with_indexes(datasets, encoding_dictionary)
    datasets = pad_utterances_to_same_length(datasets, encoding_dictionary, max_utterance_lens)

    # Save_files
    utils.save_pkl(params["dataset_path"], datasets[0], params["training_dataset_name"] + ".pkl")
    utils.save_pkl(params["dataset_path"], datasets[1], params["validation_dataset_name"] + ".pkl")
    utils.save_pkl(params["dataset_path"], datasets[2], params["test_dataset_name"] + ".pkl")
    utils.save_pkl(params["dataset_path"], datasets[0] + datasets[1] + datasets[2], params["all_dataset_name"] + ".pkl")
    utils.save_pkl(params["dataset_path"], encoding_dictionary,
                   params["encoding_dictionary_name"] + "_vocab_size_" + str(params["vocab_size"]))

if __name__ == "__main__":
    main()