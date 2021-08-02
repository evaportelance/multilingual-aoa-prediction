import torch
import torch.nn as nn
from stats import StatTracker
import os

torch.manual_seed(1)


class LSTM(nn.Module):
    def __init__(self, vocab_size, batch_size, embedding_dim, hidden_dim, output_dir):
        super(LSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.vocab_size = vocab_size

        self.h_0 = torch.zeros(2, batch_size, hidden_dim)
        self.c_0 = torch.zeros(2, batch_size, embedding_dim)
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=2, batch_first=True)  # One should be hidden
        self.linear = nn.Linear(hidden_dim, vocab_size)

    def forward(self, batch):
        embeds = self.word_embeddings(batch)
        output, _ = self.lstm(embeds)
        # batch_size * max sequence length * 2 number of hidden states
        #output_reshape = torch.reshape(output, -1, self.hidden_dim)
        logits = self.linear(output)  # Run into errors, ask Eva
        #logits = logits.view(self.batch_size, self.vocab_size)
        return logits
