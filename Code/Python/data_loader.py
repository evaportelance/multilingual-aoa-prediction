'''
What am I doing today
steps:
    *set up pytorch
    *Set up Keras and PyTorch
    Create DataLoader
What does the dataloader do?
Loads data to model for training

Input: data as a txt
Output: batches of data


Pad data, length of longest
load 5000 most common words
'''


import tensorflow as tf
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd

TRAINING_DATA_PATH = "../../Data/model-sets/train.txt"
VALIDATION_DATA_PATH = "../../Data/model-sets/validation.txt"
TEST_DATA_PATH = "../../Data/model-sets/test.txt"


'''

'''
class TrainingSet(Dataset):
    def __init__(self):
        self.data = pd.read_csv(TRAINING_DATA_PATH, sep='\n')

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]


class ValidationSet(Dataset):
    def __init__(self):
        self.data = pd.read_csv(VALIDATION_DATA_PATH, sep='\n')

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

class TestSet(Dataset):
    def __init__(self):
        self.data = pd.read_csv(TEST_DATA_PATH, sep='\n')

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

train_max_len_line = df.col1.map(lambda TrainingSet.data: len(TrainingSet.data)).max()
TrainingSet.data = TrainingSet.data.str.pad(train_max_len_line, side ='left')
valid_max_len_line = df.col1.map(lambda ValidationSet.data: len(ValidationSet.data)).max()
TrainingSet.data = ValidationSet.data.str.pad(valid_max_len_line, side ='left')
test_max_len_line = df.col1.map(lambda TestSet.data: len(TestSet.data)).max()
TrainingSet.data = TestSet.data.str.pad(test_max_len_line, side ='left')

train_dataloader = DataLoader(transformed_dataset, batch_size=4,
                        shuffle=True)

valid_dataloader = DataLoader(transformed_dataset, batch_size=4,
                        shuffle=True)

test_dataloader = DataLoader(transformed_dataset, batch_size=4,
                        shuffle=True)





 
'''
train, val, test = get_data_from_files()

tokenizer = Tokenizer(num_words=vocab_size)
tokenizer.fit_on_texts(train+val+test)
vocab = tokenizer.word_index

# transform text strings into sequences of int (representing the word's
# index in vocab)
train_seqs = tokenizer.texts_to_sequences(train)
val_seqs = tokenizer.texts_to_sequences(val)
# get the maximum length of sequences - this is needed for data generator
maxlen = max([len(seq) for seq in (train_seqs+val_seqs)])

print('vocab_size = '+str(vocab_size))
print('train_maxlen = '+str(maxlen))
print('INITIALIZE DATA GENERATORS...\n')

# Create data generators for train and test sequences
train_generator = DataGenerator(seqs = train_seqs,
                                   vocab = vocab,
                                   vocab_size = vocab_size,
                                   maxlen = maxlen,
                                   batch_size = batch_size,
                                   shuffle = shuffle)
val_generator = DataGenerator(seqs = val_seqs,
                                   vocab = vocab,
                                   vocab_size = vocab_size,
                                   maxlen = maxlen,
                                   batch_size = batch_size,
                                   shuffle = shuffle)

print('BUILDING MODEL...\n')
'''
