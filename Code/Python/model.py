from torch.utils.data import Dataset, DataLoader
from operator import itemgetter
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import data_loader as dl

torch.manual_seed(1)


from torch.utils.data import Dataset, DataLoader
from operator import itemgetter

VOCAB_SIZE = 10 #Actual 5000
BATCH_SIZE = 10
TRAINING_DATA_PATH = "../../Data/model-sets/toy_train.txt"
VALIDATION_DATA_PATH = "../../Data/model-sets/toy_validation.txt"
TEST_DATA_PATH = "../../Data/model-sets/toy_test.txt"
EMBEDDING_DIM = 30 #Hyperparameter
'''

'''
#Create dataloaders
train_dl, validation_dl, test_dl = dl.create_dataloaders()

lstm = nn.LSTM(2, 2)
hidden_dim = torch.zeros(EMBEDDING_DIM, dl.VOCAB_SIZE)
#Create the model
class LSTM(nn.Module):
    def __init__(self, hidden_dim):
        super(LSTM, self).__init__()
        self.hidden_dim = hidden_dim

        self.word_embeddings = nn.Embedding(dl.VOCAB_SIZE, EMBEDDING_DIM)

        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.lstm = nn.LSTM(EMBEDDING_DIM, hidden_dim)


    def forward(self, sentence):
        embeds = self.word_embeddings(sentence)
        lstm_out, _ = self.lstm(embeds.view(len(sentence), 1, -1))
        tag_space = self.hidden2tag(lstm_out.view(len(sentence), -1))
        tag_scores = F.log_softmax(tag_space, dim=1)
        return tag_scores