import torch
import torch.nn as nn

torch.manual_seed(1)
class LSTM(nn.Module):
    def __init__(self, vocab_size, batch_size, embedding_dim):
        hidden_dim = torch.zeros(embedding_dim, vocab_size)
        super(LSTM, self).__init__()
        self.h_0 = torch.zeros(2, batch_size, hidden_dim)
        self.c_0 = torch.zeros(2, batch_size, embedding_dim)
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)

        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=2, batch_first=True)
        self.linear = nn.Linear(hidden_dim, vocab_size)

    def forward(self, utterance):
        embeds = self.word_embeddings(utterance)
        _, (_,x1) = self.lstm(embeds,(self.h_0, self.c_0))
        logits = self.linear(x1[1]) #Run into errors, ask Eva
        return logits