'''
Opens text files and creates dataloader objects for training a model.
'''
import torch
from torch.utils.data import Dataset, DataLoader
from operator import itemgetter
import random

'''
    Opens a text file and creates a list whose elements are lists that 
    correspond to file lines. The elements of each file line list are
    the words in the line.
    
    Parameters:
        file_path: a relative path to a text file
        
    Returns:
        file_list: a list of lines in the file at file_path
'''
def open_file(file_path):
    lines = []
    with open(file_path, "r") as f:
        for line in f:
            line = line.strip('\n')
            line_as_list = line.split(" ")
            lines.append(line_as_list)
    return lines

'''
    Reduces n-dimensional lists to a single dimension.
    
    Parameters: 
        list: any list
    
    Returns: a flattened version of list
'''
def flatten_list(list):
    return [item for sublist in list for item in sublist]

'''
    Creates a dictionary from a word list where the keys are words and the values are
    the number of occurrences of the word in the word list for the number of words in
    VOCAB_SIZE that have the most occurrences in the word list. All other word-keys
    have value 0.
    
    Parameters:
        word_list: one-dimensional list of words
        
    Returns:
        word_dict: a dictionary of words to word counts
'''
def create_word_dict(all_data, vocab_size):
    flattened_list = flatten_list(all_data)
    word_set = set(flattened_list)
    word_index_dict = {}
    word_dict = word_index_dict.fromkeys(word_set, 0)
    for word in flattened_list:
        word_dict[word] = word_dict[word] + 1
    lexicon = dict(sorted(word_dict.items(), key = itemgetter(1), reverse = True)[:VOCAB_SIZE])
    max_word_len = len(str(vocab_size))

    #Set word_dict values to indexes and pad words
    i = 1
    for word in word_set:
        if word in lexicon:
            word_dict[word] = int(str(i).zfill(max_word_len))
            i += 1
        else:
            word_dict[word] = int("0".zfill(max_word_len))
    return word_dict

class Dataset(Dataset):
    def __init__(self, data, encoding_dict, batch_size):
        self.data = data
        max_utterance_len = 0
        for i in range(len(data)):
            if len(data[i]) > max_utterance_len:
                max_utterance_len = len(data[i])
            for j in range(len(data[i])):
                self.data[i][j] = encoding_dict[data[i][j]]
        if max_utterance_len > batch_size:
            raise ValueError("Batch size must be at least " + str(max_utterance_len) + " items long. "
                            "Otherwise, utterance lists will be truncated by the dataloader.")
        for i in range(len(data)):
            zeros = 0 * random.choice(list(encoding_dict.values()))
            padding = [zeros] * (max_utterance_len - len(data[i]))
            #print("Unpadded data: " + str(self.data[i]))
            self.data[i] = padding + self.data[i]
            #print("Padded data: " + str(self.data[i]))
        self.data = torch.tensor(self.data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        #torch tensor list of lists

        return self.data[index]

'''
    Creates dataloaders for the training, validation, and test datasets.
    
    Parameters: none
    
    Returns:
        training_dataloader, validation_dataloader, test_dataloader
'''
def create_dataloaders(training_data_path, validation_data_path, test_data_path, vocab_size, batch_size):
    #Get file data
    train_data = open_file(training_data_path)
    validation_data = open_file(validation_data_path)
    test_data = open_file(test_data_path)

    #Create word to index dictionary
    all_data = train_data + validation_data + test_data
    word_dict = create_word_dict(all_data, vocab_size)

    #Initialize datasets, replacing words with indexes or '0,' as appropriate
    training_dataset = Dataset(train_data, word_dict, batch_size)
    validation_dataset = Dataset(validation_data, word_dict, batch_size)
    test_dataset = Dataset(test_data, word_dict, batch_size)

    #Create dataloaders
    training_dataloader = DataLoader(training_dataset, batch_size=batch_size,shuffle=True)
    validation_dataloader = DataLoader(validation_dataset, batch_size=batch_size,shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size,shuffle=True)
    return training_dataloader, validation_dataloader, test_dataloader

############TEMPORARY#################
TRAINING_DATA_PATH = "../../Data/model-sets/toy_train.txt"
VALIDATION_DATA_PATH = "../../Data/model-sets/toy_validation.txt"
TEST_DATA_PATH = "../../Data/model-sets/toy_test.txt"
VOCAB_SIZE = 20 #Actual 5000
BATCH_SIZE = 20

import time
start_time = time.time()
train_dl, validation_dl, test_dl = create_dataloaders(TRAINING_DATA_PATH, VALIDATION_DATA_PATH, TEST_DATA_PATH,
                                                         VOCAB_SIZE, BATCH_SIZE)
#print("--- %s seconds ---" % (time.time() - start_time))
#
#print("Dataloader output:")
#for batch in enumerate(test_dl):
#    print(batch)
#    for elem in batch[1]:
#        print(elem)

############TEMPORARY#################
