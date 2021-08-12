import torch
import torch.nn as nn

torch.manual_seed(1)
'''
Creates an lstm that takes in a word embedding tensor of shape (batch_size X max_utterance_len X embedding dim)
as input and outputs a weight tensor of the form (batch_size X max_utterance_len X vocabulary size), where each
entry in the output is the surprisal (entropy) of a particular word in context.
'''
class LSTM(nn.Module):
    def __init__(self, vocab_size, batch_size, embedding_dim, hidden_dim):
        super(LSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.vocab_size = vocab_size
        self.h_0 = torch.zeros(2, batch_size, hidden_dim)
        self.c_0 = torch.zeros(2, batch_size, embedding_dim)
        self.word_embeddings = nn.Embedding(vocab_size + 1, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=2, batch_first=True)  # One should be hidden
        self.linear = nn.Linear(hidden_dim, vocab_size + 1)

    def forward(self, batch):
        embeds = self.word_embeddings(batch)
        output, _ = self.lstm(embeds)
        logits = self.linear(output)
        return logits
