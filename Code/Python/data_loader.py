from torch.utils.data import Dataset, DataLoader
from operator import itemgetter

VOCAB_SIZE = 10 #Actual 5000
BATCH_SIZE = 10
TRAINING_DATA_PATH = "../../Data/model-sets/toy_train.txt" 
VALIDATION_DATA_PATH = "../../Data/model-sets/toy_validation.txt"
TEST_DATA_PATH = "../../Data/model-sets/toy_test.txt"

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
    file_list = []
    with open(file_path, "r") as f:
        for line in f:
            file_list.append(line.split())
    return file_list

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
def create_word_dict(word_list):
    word_list = flatten_list(word_list)
    word_set = set(word_list)
    word_index_dict = {}
    word_dict = word_index_dict.fromkeys(word_set, 0)
    for word in word_list:
        word_dict[word] = word_dict[word] + 1
    lexicon = dict(sorted(word_dict.items(), key = itemgetter(1), reverse = True)[:VOCAB_SIZE])
    not_lexicon = set(word_dict) - set(lexicon)
    for word in not_lexicon:
        word_dict[word] = 0
    return word_dict

class Dataset(Dataset):
    def __init__(self, data, encoding_dict):
        self.data = data
        for i in range(len(data)):
            for j in range(len(data[i])):
                self.data[i][j] = encoding_dict[data[i][j]]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

'''
    Creates dataloaders for the training, validation, and test datasets.
    
    Parameters: none
    
    Returns:
        training_dataloader, validation_dataloader, test_dataloader
'''
def create_dataloaders():
    #Get file data
    train_data = open_file(TRAINING_DATA_PATH)
    validation_data = open_file(VALIDATION_DATA_PATH)
    test_data = open_file(TEST_DATA_PATH)

    #Create word to index dictionary
    all_data = train_data + validation_data + test_data
    word_dict = create_word_dict(all_data)

    #Initialize datasets, replacing words with indexes or '0,' as appropriate
    training_dataset = Dataset(train_data, word_dict)
    validation_dataset = Dataset(validation_data, word_dict)
    test_dataset = Dataset(test_data, word_dict)

    #Create dataloaders
    training_dataloader = DataLoader(training_dataset, batch_size=BATCH_SIZE,shuffle=True)
    validation_dataloader = DataLoader(validation_dataset, batch_size=BATCH_SIZE,shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE,shuffle=True)
    return training_dataloader, validation_dataloader, test_dataloader