'''
What am I doing today
steps:
    *set up pytorch
    *Set up Keras and PyTorch
    *Initialize data classes
    *Pad data
    *Create DataLoaders
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

TRAINING_DATA_PATH = "../../Data/model-sets/toy_train.txt"
VALIDATION_DATA_PATH = "../../Data/model-sets/toy_validation.txt"
TEST_DATA_PATH = "../../Data/model-sets/toy_test.txt"

'''
'''
# class FaceLandmarksDataset(Dataset):
#     """Face Landmarks dataset."""
#
#     def __init__(self, csv_file, root_dir, transform=None):
#         """
#         Args:
#             csv_file (string): Path to the csv file with annotations.
#             root_dir (string): Directory with all the images.
#             transform (callable, optional): Optional transform to be applied
#                 on a sample.
#         """
#         self.landmarks_frame = pd.read_csv(csv_file)
#         self.root_dir = root_dir
#         self.transform = transform
#
#     def __len__(self):
#         return len(self.landmarks_frame)
#
#     def __getitem__(self, idx):
#         if torch.is_tensor(idx):
#             idx = idx.tolist()
#
#         img_name = os.path.join(self.root_dir,
#                                 self.landmarks_frame.iloc[idx, 0])
#         image = io.imread(img_name)
#         landmarks = self.landmarks_frame.iloc[idx, 1:]
#         landmarks = np.array([landmarks])
#         landmarks = landmarks.astype('float').reshape(-1, 2)
#         sample = {'image': image, 'landmarks': landmarks}
#
#         if self.transform:
#             sample = self.transform(sample)
#
#         return sample

#File open function

train_data = []
with open(TRAINING_DATA_PATH, "r") as f:
    for line in f:
        train_data.append(line.split())

validation_data = []
with open(TRAINING_DATA_PATH, "r") as f:
    for line in f:
        validation_data.append(line.split())

test_data = []
with open(TRAINING_DATA_PATH, "r") as f:
    for line in f:
        test_data.append(line.split())

train_word_set = set(train_data)
word_index_dict = {}
train_set = word_index_dict.fromkeys(train_word_set[, 0])

class TrainingSet(Dataset):
    def __init__(self, path = TRAINING_DATA_PATH):
        self.data = test

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

class ValidationSet(Dataset):
    def __init__(self, path = VALIDATION_DATA_PATH):
        with open(path, "r") as f:
            keys = f.readlines()
            maxlen = len(max(keys, key=len))
            keys = [line.rjust(maxlen) for line in keys]
            values = range(len(keys))
            self.data = dict(zip(keys, values))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):

        return self.data[index]

class TestSet(Dataset):
    def __init__(self, path = TEST_DATA_PATH):
        with open(path, "r") as f:
            keys = f.readlines()
            maxlen = len(max(keys, key=len))
            keys = [line.rjust(maxlen) for line in keys]
            values = range(len(keys))
            self.data = dict(zip(keys, values))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

#Initialize a dataloader for each data set
training_dataloader = TrainingSet()
validation_dataloader = ValidationSet()
test_dataloader = TestSet()

'''


'''




# train_max_len_line = .col1.map(lambda TrainingSet.data: len(TrainingSet.data)).max()
# TrainingSet.data = TrainingSet.data.str.pad(train_max_len_line, side ='left')
#
# valid_max_len_line = df.col1.map(lambda ValidationSet.data: len(ValidationSet.data)).max()
# TrainingSet.data = ValidationSet.data.str.pad(valid_max_len_line, side ='left')
# test_max_len_line = df.col1.map(lambda TestSet.data: len(TestSet.data)).max()
# TrainingSet.data = TestSet.data.str.pad(test_max_len_line, side ='left')
#
# train_dataloader = DataLoader(transformed_dataset, batch_size=4,
#                         shuffle=True)
#
# valid_dataloader = DataLoader(transformed_dataset, batch_size=4,
#                         shuffle=True)
#
# test_dataloader = DataLoader(transformed_dataset, batch_size=4,
#                         shuffle=True)
#




 
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
